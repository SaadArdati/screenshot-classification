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

// Constants for CUDA kernels in constant memory
__constant__ int c_numBins = NUM_BINS;

// Simple CUDA kernel for converting RGB to grayscale with coalesced memory access
__global__ void rgbToGraySimple(unsigned char* rgb, unsigned char* gray, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Use stride to process multiple pixels per thread for better occupancy
    for (int i = idx; i < size; i += stride) {
        int rgb_idx = i * 3;
        gray[i] = (rgb[rgb_idx] + rgb[rgb_idx + 1] + rgb[rgb_idx + 2]) / 3;
    }
}

// CUDA kernel for computing histogram with improved memory patterns
__global__ void computeHistogramSimple(unsigned char* gray, float* histogram, int size) {
    __shared__ unsigned int temp_hist[NUM_BINS];
    
    // Initialize shared memory histogram
    int tid = threadIdx.x;
    if (tid < NUM_BINS) {
        temp_hist[tid] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple pixels per thread
    for (int i = idx; i < size; i += stride) {
        int bin = gray[i] * c_numBins / 256;
        atomicAdd(&temp_hist[bin], 1);
    }
    
    __syncthreads();
    
    // Only threads with IDs < NUM_BINS write to global memory
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], (float)temp_hist[tid]);
    }
}

// Optimized CUDA kernel for distance calculation with coalesced memory access
__global__ void calculateDistances(Feature* trainSet, Feature* testFeature, float* distances, int trainSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Cache test feature in shared memory
    __shared__ float test_bins[NUM_BINS];
    
    // Collaborative loading of test feature
    if (threadIdx.x < NUM_BINS) {
        test_bins[threadIdx.x] = testFeature->bins[threadIdx.x];
    }
    __syncthreads();
    
    // Each thread processes multiple training examples for better occupancy
    for (int i = idx; i < trainSize; i += stride) {
        float dist = 0.0f;
        
        // Unroll small loops for better performance
        #pragma unroll
        for (int k = 0; k < NUM_BINS; k++) {
            float diff = test_bins[k] - trainSet[i].bins[k];
            dist += diff * diff;
        }
        
        distances[i] = dist;
    }
}

// Optimized CUDA kernel for converting RGB to grayscale using coalesced memory access
__global__ void rgbToGrayShared(unsigned char* rgb, unsigned char* gray, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = width * height;
    
    // Each thread processes multiple pixels for better occupancy
    for (int i = idx; i < size; i += stride) {
        int rgb_idx = i * 3;
        gray[i] = (rgb[rgb_idx] + rgb[rgb_idx + 1] + rgb[rgb_idx + 2]) / 3;
    }
}

// Optimized CUDA kernel for computing histogram with warp-level aggregation
__global__ void computeHistogramShared(unsigned char* gray, float* histogram, int size) {
    // Shared memory for per-block histogram
    __shared__ unsigned int temp_hist[NUM_BINS];
    
    int tid = threadIdx.x;
    
    // Clear shared memory histogram
    if (tid < NUM_BINS) {
        temp_hist[tid] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple pixels with a stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple pixels per thread
    for (int i = idx; i < size; i += stride) {
        int bin = gray[i] * c_numBins / 256;
        atomicAdd(&temp_hist[bin], 1);
    }
    
    __syncthreads();
    
    // Only threads with IDs < NUM_BINS write results back to global memory
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], (float)temp_hist[tid]);
    }
}

// Optimized CUDA kernel for distance calculation with improved memory access
__global__ void calculateDistancesShared(Feature* trainSet, Feature* testFeature, float* distances, int trainSize) {
    // Load test feature bins into shared memory for faster access
    __shared__ float test_bins[NUM_BINS];
    
    // Collaborative loading of test feature into shared memory
    if (threadIdx.x < NUM_BINS) {
        test_bins[threadIdx.x] = testFeature->bins[threadIdx.x];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple training examples
    for (int i = idx; i < trainSize; i += stride) {
        float dist = 0.0f;
        
        // Unroll small loops for better performance
        #pragma unroll
        for (int k = 0; k < NUM_BINS; k++) {
            float diff = test_bins[k] - trainSet[i].bins[k];
            dist += diff * diff;
        }
        
        distances[i] = dist;
    }
}

// Feature extraction using optimized GPU implementation
void extractFeaturesOptimizedGPU(const char* imagePath, Feature* feature) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", imagePath);
        return;
    }
    
    // Initialize feature
    memset(feature, 0, sizeof(Feature));
    
    // Calculate sizes for proper memory alignment
    int size = width * height;
    size_t rgb_pitch, gray_pitch;
    unsigned char* d_rgb;
    unsigned char* d_gray;
    float* d_histogram;
    
    // Allocate GPU memory with pitched allocation for better memory alignment
    CUDA_CHECK(cudaMallocPitch(&d_rgb, &rgb_pitch, width * 3 * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMallocPitch(&d_gray, &gray_pitch, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(float)));
    
    // Copy image to GPU with proper pitch
    CUDA_CHECK(cudaMemcpy2D(d_rgb, rgb_pitch, img, width * 3 * sizeof(unsigned char),
                           width * 3 * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    
    // Clear histogram memory
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(float)));
    
    // Convert to grayscale using optimized kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    rgbToGrayShared<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK_KERNEL();
    
    // Compute histogram using optimized kernel
    computeHistogramShared<<<gridSize, blockSize>>>(d_gray, d_histogram, size);
    CUDA_CHECK_KERNEL();
    
    // Copy histogram to CPU
    float histogram[NUM_BINS];
    CUDA_CHECK(cudaMemcpy(histogram, d_histogram, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Normalize histogram on CPU
    float total = width * height;
    for (int i = 0; i < NUM_BINS; i++) {
        feature->bins[i] = histogram[i] / total;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_histogram));
    
    stbi_image_free(img);
}

// Load images from directory with optimized GPU usage
int loadImagesFromDirOptimizedGPU(const char* dirPath, int label, Feature* features, int* count, int maxImages) {
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
        
        // Extract features using optimized GPU function
        extractFeaturesOptimizedGPU(fullPath, &features[*count]);
        features[*count].label = label;
        (*count)++;
        loaded++;
    }
    
    closedir(dir);
    printf("Loaded %d images from %s\n", loaded, dirPath);
    return loaded;
}

// Optimized GPU-based model evaluation with improved memory access patterns
float evaluateModelOptimizedGPU(Feature* trainSet, int trainSize, Feature* testSet, int testSize) {
    if (trainSize == 0 || testSize == 0) return 0.0f;
    
    int correct = 0;
    
    // Allocate pinned memory for faster host-device transfers
    Feature* d_trainSet;
    Feature* d_testFeature;
    float* d_distances;
    float* h_distances;
    
    CUDA_CHECK(cudaMalloc(&d_trainSet, trainSize * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_testFeature, sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_distances, trainSize * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_distances, trainSize * sizeof(float))); // Pinned memory
    
    // Copy training set to GPU once
    CUDA_CHECK(cudaMemcpy(d_trainSet, trainSet, trainSize * sizeof(Feature), cudaMemcpyHostToDevice));
    
    // Find optimal grid and block dimensions
    int blockSize = 256;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                               calculateDistancesShared, 0, 0));
    int gridSize = (trainSize + blockSize - 1) / blockSize;
    
    // Process each test sample
    for (int i = 0; i < testSize; i++) {
        // Copy current test sample to GPU
        CUDA_CHECK(cudaMemcpy(d_testFeature, &testSet[i], sizeof(Feature), cudaMemcpyHostToDevice));
        
        // Calculate distances using optimized kernel
        calculateDistancesShared<<<gridSize, blockSize>>>(d_trainSet, d_testFeature, d_distances, trainSize);
        CUDA_CHECK_KERNEL();
        
        // Copy distances back to pinned memory for faster transfers
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
    CUDA_CHECK(cudaFreeHost(h_distances));
    
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
    
    // Allocate pinned memory for features to improve data transfer performance
    Feature* trainFeatures;
    Feature* testFeatures;
    CUDA_CHECK(cudaMallocHost(&trainFeatures, MAX_IMAGES * sizeof(Feature)));
    CUDA_CHECK(cudaMallocHost(&testFeatures, MAX_IMAGES * sizeof(Feature)));
    
    if (!trainFeatures || !testFeatures) {
        fprintf(stderr, "Failed to allocate memory for features\n");
        cudaFreeHost(trainFeatures);
        cudaFreeHost(testFeatures);
        return 1;
    }
    
    // Load training data
    int trainCount = 0;
    start_time = clock();
    loadImagesFromDirOptimizedGPU(screenshots_train_dir, 1, trainFeatures, &trainCount, MAX_IMAGES);
    loadImagesFromDirOptimizedGPU(non_screenshots_train_dir, 0, trainFeatures, &trainCount, MAX_IMAGES);
    end_time = clock();
    load_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Load test data
    int testCount = 0;
    start_time = clock();
    loadImagesFromDirOptimizedGPU(screenshots_test_dir, 1, testFeatures, &testCount, MAX_IMAGES);
    loadImagesFromDirOptimizedGPU(non_screenshots_test_dir, 0, testFeatures, &testCount, MAX_IMAGES);
    end_time = clock();
    load_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Evaluate model
    start_time = clock();
    float accuracy = evaluateModelOptimizedGPU(trainFeatures, trainCount, testFeatures, testCount);
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
    cudaFreeHost(trainFeatures);
    cudaFreeHost(testFeatures);
    
    return 0;
}