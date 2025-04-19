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
#include "../include/feature_extraction.cuh"
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

// Load images from directory with optimized GPU usage
int loadImagesFromDirOptimizedGPU(const char* dirPath, int label, Feature* features, int* count, int maxImages) {
    DIR* dir = opendir(dirPath);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", dirPath);
        return -1;
    }
    
    struct dirent* entry;
    int loaded = 0;
    int image_count = 0;
    
    while ((entry = readdir(dir)) != NULL && *count < maxImages) {
        if (entry->d_type != DT_REG) continue;
        
        char fullPath[512];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", dirPath, entry->d_name);
        
        // Show progress every 100 images
        if (++image_count % 100 == 0) {
            printf("\rProcessing image %d from %s...", image_count, dirPath);
            fflush(stdout);
        }
        
        // Extract features using optimized GPU function
        extractFeaturesGPU(fullPath, &features[*count]);
        features[*count].label = label;
        (*count)++;
        loaded++;
    }
    
    closedir(dir);
    printf("\nLoaded %d images from %s\n", loaded, dirPath);
    return loaded;
}

// Optimized GPU-based model evaluation
float evaluateModelOptimizedGPU(Feature* trainSet, int trainSize, Feature* testSet, int testSize) {
    if (trainSize == 0 || testSize == 0) return 0.0f;
    
    printf("Evaluating model with %d training samples and %d test samples...\n", trainSize, testSize);
    
    int correct = 0;
    int* predictions = (int*)malloc(testSize * sizeof(int));
    
    if (!predictions) return 0.0f;
    
    // Use the batched classification function
    printf("Performing batch classification...\n");
    if (classifyBatchGPU(trainSet, trainSize, testSet, testSize, predictions) == 0) {
        printf("Classification completed. Counting correct predictions...\n");
        // Count correct predictions
        for (int i = 0; i < testSize; i++) {
            if (predictions[i] == testSet[i].label) {
                correct++;
            }
        }
    } else {
        printf("Batch classification failed!\n");
    }
    
    free(predictions);
    return (float)correct / testSize;
}

// Save model to file
int saveModel(const char* path, Feature* features, int count) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    
    printf("Saving model to %s with %d features...\n", path, count);
    fwrite(&count, sizeof(int), 1, f);
    fwrite(features, sizeof(Feature), count, f);
    fclose(f);
    return 0;
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
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
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
    
    // Load training data with optimized GPU functions
    int trainCount = 0;
    printf("\nLoading training data...\n");
    start_time = clock();
    loadImagesFromDirOptimizedGPU(screenshots_train_dir, 1, trainFeatures, &trainCount, MAX_IMG_COUNT);
    loadImagesFromDirOptimizedGPU(non_screenshots_train_dir, 0, trainFeatures, &trainCount, MAX_IMG_COUNT);
    end_time = clock();
    load_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training data loaded in %.2f seconds\n", load_time);
    
    // Load test data with optimized GPU functions
    int testCount = 0;
    printf("\nLoading test data...\n");
    start_time = clock();
    loadImagesFromDirOptimizedGPU(screenshots_test_dir, 1, testFeatures, &testCount, MAX_IMG_COUNT);
    loadImagesFromDirOptimizedGPU(non_screenshots_test_dir, 0, testFeatures, &testCount, MAX_IMG_COUNT);
    end_time = clock();
    load_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Test data loaded in %.2f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    
    // Evaluate model with optimized GPU functions
    printf("\nEvaluating model...\n");
    start_time = clock();
    float accuracy = evaluateModelOptimizedGPU(trainFeatures, trainCount, testFeatures, testCount);
    end_time = clock();
    evaluation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Model evaluation completed in %.2f seconds\n", evaluation_time);
    
    // Save the model
    printf("\nSaving model...\n");
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
    printf("\nOptimized Performance Metrics:\n");
    printf("-----------------------------\n");
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
    
    printf("\nProgram completed successfully!\n");
    return 0;
}
