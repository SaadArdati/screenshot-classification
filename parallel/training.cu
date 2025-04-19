#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

// Platform-specific directory handling
#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(dir) _mkdir(dir)
#define PATH_SEPARATOR "\\"
#else
#include <sys/stat.h>
#define CREATE_DIR(dir) mkdir(dir, 0777)
#define PATH_SEPARATOR "/"
#endif

// Simple CUDA kernel for converting RGB to grayscale
__global__ void rgbToGraySimple(unsigned char* rgb, unsigned char* gray, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int rgb_idx = idx * 3;
        gray[idx] = (rgb[rgb_idx] + rgb[rgb_idx + 1] + rgb[rgb_idx + 2]) / 3;
    }
}

// Simple CUDA kernel for computing histogram (no shared memory, no optimization)
__global__ void computeHistogramSimple(unsigned char* gray, float* histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bin = gray[idx] * NUM_BINS / 256;
        atomicAdd(&histogram[bin], 1.0f);
    }
}

// Simple CUDA kernel for distance calculation
__global__ void calculateDistances(Feature* trainSet, Feature* testFeature, float* distances, int trainSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trainSize) {
        float dist = 0.0f;
        for (int k = 0; k < NUM_BINS; k++) {
            float diff = testFeature->bins[k] - trainSet[idx].bins[k];
            dist += diff * diff;
        }
        distances[idx] = dist;
    }
}

// Feature extraction using basic GPU functions
void extractFeaturesBasicGPU(const char* imagePath, Feature* feature) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", imagePath);
        return;
    }
    
    // Initialize feature
    memset(feature, 0, sizeof(Feature));
    
    // Allocate GPU memory (not reused, allocated for each image)
    unsigned char* d_rgb;
    unsigned char* d_gray;
    float* d_histogram;
    cudaMalloc(&d_rgb, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_gray, width * height * sizeof(unsigned char));
    cudaMalloc(&d_histogram, NUM_BINS * sizeof(float));
    
    // Copy image to GPU
    cudaMemcpy(d_rgb, img, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Convert to grayscale (simple 1D grid)
    int size = width * height;
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    rgbToGraySimple<<<gridSize, blockSize>>>(d_rgb, d_gray, size);
    
    // Clear histogram
    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(float));
    
    // Compute histogram
    computeHistogramSimple<<<gridSize, blockSize>>>(d_gray, d_histogram, size);
    
    // Copy histogram to CPU
    float histogram[NUM_BINS];
    cudaMemcpy(histogram, d_histogram, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Normalize histogram on CPU
    float total = width * height;
    for (int i = 0; i < NUM_BINS; i++) {
        feature->bins[i] = histogram[i] / total;
    }
    
    // Cleanup (no memory reuse, freed after each image)
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_histogram);
    
    stbi_image_free(img);
}

// Load images from directory with basic GPU usage
int loadImagesFromDirBasicGPU(const char* dirPath, int label, Feature* features, int* count, int maxImages) {
    DIR* dir = opendir(dirPath);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", dirPath);
        return -1;
    }
    
    struct dirent* entry;
    int loaded = 0;
    
    while ((entry = readdir(dir)) != NULL && *count < maxImages) {
        if (entry->d_type != DT_REG) continue;
        
        char fullPath[512];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", dirPath, entry->d_name);
        
        // Extract features using basic GPU function
        extractFeaturesBasicGPU(fullPath, &features[*count]);
        features[*count].label = label;
        (*count)++;
        loaded++;
    }
    
    closedir(dir);
    printf("Loaded %d images from %s\n", loaded, dirPath);
    return loaded;
}

// Very basic GPU-based model evaluation (inefficient)
float evaluateModelBasicGPU(Feature* trainSet, int trainSize, Feature* testSet, int testSize) {
    if (trainSize == 0 || testSize == 0) return 0.0f;
    
    int correct = 0;
    
    // Allocate GPU memory once (reused for all test samples)
    Feature* d_trainSet;
    Feature* d_testFeature;
    float* d_distances;
    float* h_distances = (float*)malloc(trainSize * sizeof(float));
    
    CUDA_CHECK(cudaMalloc(&d_trainSet, trainSize * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_testFeature, sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_distances, trainSize * sizeof(float)));
    
    // Copy training set to GPU once
    CUDA_CHECK(cudaMemcpy(d_trainSet, trainSet, trainSize * sizeof(Feature), cudaMemcpyHostToDevice));
    
    // Process each test sample
    for (int i = 0; i < testSize; i++) {
        // Copy current test sample to GPU
        CUDA_CHECK(cudaMemcpy(d_testFeature, &testSet[i], sizeof(Feature), cudaMemcpyHostToDevice));
        
        // Calculate distances
        int blockSize = 256;
        int gridSize = (trainSize + blockSize - 1) / blockSize;
        calculateDistances<<<gridSize, blockSize>>>(d_trainSet, d_testFeature, d_distances, trainSize);
        
        // Synchronize to ensure kernel completion
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy distances back to CPU
        CUDA_CHECK(cudaMemcpy(h_distances, d_distances, trainSize * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Find minimum distance on CPU
        float minDist = INFINITY;
        int prediction = 0;
        for (int j = 0; j < trainSize; j++) {
            if (h_distances[j] < minDist) {
                minDist = h_distances[j];
                prediction = trainSet[j].label;
            }
        }
        
        if (prediction == testSet[i].label) {
            correct++;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_trainSet));
    CUDA_CHECK(cudaFree(d_testFeature));
    CUDA_CHECK(cudaFree(d_distances));
    free(h_distances);
    
    return (float)correct / testSize;
}

// Save model to file
int saveModel(const char* path, Feature* features, int count) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    
    fwrite(&count, sizeof(int), 1, f);
    fwrite(features, sizeof(Feature), count, f);
    fclose(f);
    return 0;
}

// Print performance metrics
void printPerformanceMetrics(double load_time, double eval_time, double save_time, double total_time, int trainCount, int testCount, float accuracy) {
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Data Loading Time: %.2f seconds\n", load_time);
    printf("Model Evaluation Time: %.2f seconds\n", eval_time);
    printf("Model Saving Time: %.2f seconds\n", save_time);
    printf("Total Processing Time: %.2f seconds\n", total_time);
    printf("Number of Training Examples: %d\n", trainCount);
    printf("Number of Test Examples: %d\n", testCount);
    printf("Classification Accuracy: %.2f%%\n", accuracy * 100);
}

int main(int argc, char** argv) {
    // Performance measurement variables
    clock_t start_time, end_time;
    double load_time = 0.0, evaluation_time = 0.0, save_time = 0.0, total_time = 0.0;
    
    // Check for CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA capable devices found!\n");
        return 1;
    }
    
    // Print CUDA device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nCUDA Device Information:\n");
    printf("------------------------\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("\n");
    
    // Paths to image directories
    const char* screenshots_train_dir = "../split_data/screenshots_256x256/train";
    const char* non_screenshots_train_dir = "../split_data/non_screenshot_256x256/train";
    const char* screenshots_test_dir = "../split_data/screenshots_256x256/test";
    const char* non_screenshots_test_dir = "../split_data/non_screenshot_256x256/test";
    const char* modelPath = "trained_model.bin";
    
    // Check command line args
    if (argc > 1) {
        modelPath = argv[1];
    }
    
    // Start total time measurement
    clock_t total_start = clock();
    
    // Allocate memory for features
    const int MAX_IMG_COUNT = 50000;
    Feature* trainFeatures = (Feature*)malloc(MAX_IMG_COUNT * sizeof(Feature));
    Feature* testFeatures = (Feature*)malloc(MAX_IMG_COUNT * sizeof(Feature));
    
    if (!trainFeatures || !testFeatures) {
        fprintf(stderr, "Failed to allocate memory for features\n");
        free(trainFeatures);
        free(testFeatures);
        return 1;
    }
    
    // Load training data
    int trainCount = 0;
    start_time = clock();
    loadImagesFromDirBasicGPU(screenshots_train_dir, 1, trainFeatures, &trainCount, MAX_IMG_COUNT);
    loadImagesFromDirBasicGPU(non_screenshots_train_dir, 0, trainFeatures, &trainCount, MAX_IMG_COUNT);
    end_time = clock();
    load_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Load test data
    int testCount = 0;
    start_time = clock();
    loadImagesFromDirBasicGPU(screenshots_test_dir, 1, testFeatures, &testCount, MAX_IMG_COUNT);
    loadImagesFromDirBasicGPU(non_screenshots_test_dir, 0, testFeatures, &testCount, MAX_IMG_COUNT);
    end_time = clock();
    load_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Evaluate model
    start_time = clock();
    float accuracy = evaluateModelBasicGPU(trainFeatures, trainCount, testFeatures, testCount);
    end_time = clock();
    evaluation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Save the model
    start_time = clock();
    if (saveModel(modelPath, trainFeatures, trainCount) != 0) {
        fprintf(stderr, "Failed to save model\n");
    } else {
        printf("Model saved successfully\n");
    }
    end_time = clock();
    save_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Calculate total time
    clock_t total_end = clock();
    total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    
    // Calculate memory usage
    size_t total_memory = (trainCount + testCount) * sizeof(Feature);
    
    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Data Loading Time: %.2f seconds\n", load_time);
    printf("Model Evaluation Time: %.2f seconds\n", evaluation_time);
    printf("Model Saving Time: %.2f seconds\n", save_time);
    printf("Total Processing Time: %.2f seconds\n", total_time);
    printf("Number of Training Examples: %d\n", trainCount);
    printf("Number of Test Examples: %d\n", testCount);
    printf("Classification Accuracy: %.2f%%\n", accuracy * 100);
    printf("Total Memory Usage: %.2f MB\n", (float)total_memory / (1024.0f * 1024.0f));
    
    // Cleanup
    free(trainFeatures);
    free(testFeatures);
    
    return 0;
}