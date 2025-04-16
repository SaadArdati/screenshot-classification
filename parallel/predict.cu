#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

// Function declarations - these are implemented in feature_extraction.cu and knn.cu
extern "C" void extractFeaturesGPU(const unsigned char* h_images, int batch_size,
                                   int width, int height, int channels, 
                                   Feature* h_features);
extern "C" void classifyBatchGPU(const Feature* train_features, int train_size,
                               const Feature* query_features, int query_size,
                               int* predictions, double* computation_times);

// Load model from file
Feature* loadModel(const char* filename, int* size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model file: %s\n", filename);
        return NULL;
    }
    
    // Read dataset size
    if (fread(size, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read model size\n");
        fclose(f);
        return NULL;
    }
    
    // Allocate memory for features
    Feature* model = (Feature*)malloc(*size * sizeof(Feature));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    // Read features
    if (fread(model, sizeof(Feature), *size, f) != *size) {
        fprintf(stderr, "Failed to read model data\n");
        free(model);
        fclose(f);
        return NULL;
    }
    
    fclose(f);
    return model;
}

// Print device information
void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("\nCUDA Device Information:\n");
    printf("------------------------\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model_file> <image_path>\n", argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    
    // Performance timing
    clock_t start_time, end_time;
    double load_model_time = 0.0, feature_time = 0.0, classification_time = 0.0;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    printDeviceInfo();
    
    // Load model
    start_time = clock();
    int model_size;
    Feature* model = loadModel(model_path, &model_size);
    end_time = clock();
    load_model_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    if (!model) {
        return 1;
    }
    
    printf("Model loaded with %d training examples\n", model_size);
    
    // Load query image
    int width, height, channels;
    unsigned char* img = stbi_load(image_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        free(model);
        return 1;
    }
    
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
    
    // Extract features
    start_time = clock();
    Feature query_feature;
    extractFeaturesGPU(img, 1, width, height, 3, &query_feature);
    end_time = clock();
    feature_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Check for statistical detection (label=2 indicates statistical detection)
    if (query_feature.label == 2) {
        printf("Classification result for %s: SCREENSHOT (Statistical analysis)\n", image_path);
        printf("This image was detected as a screenshot by analyzing UI patterns\n");
    } else {
        // Perform kNN classification on GPU
        start_time = clock();
        int prediction;
        double knn_times[2] = {0}; // [0] = transfer time, [1] = compute time
        
        // Call GPU KNN function (we only have 1 query)
        classifyBatchGPU(model, model_size, &query_feature, 1, &prediction, knn_times);
        
        end_time = clock();
        classification_time = knn_times[0] + knn_times[1];
        
        printf("Classification result for %s: %s\n", image_path, 
               prediction ? "SCREENSHOT" : "NON-SCREENSHOT");
        printf("Classification based on K-nearest neighbors (K=%d)\n", K_NEIGHBORS);
    }
    
    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Model Loading Time: %.5f seconds\n", load_model_time);
    printf("Feature Extraction Time: %.5f seconds\n", feature_time);
    printf("Classification Time: %.5f seconds\n", classification_time);
    printf("Total Processing Time: %.5f seconds\n", 
           load_model_time + feature_time + classification_time);
    printf("Model Memory Usage: %.2f MB\n", 
           (float)(model_size * sizeof(Feature)) / (1024.0f * 1024.0f));
    
    // Clean up
    stbi_image_free(img);
    free(model);
    
    return 0;
} 