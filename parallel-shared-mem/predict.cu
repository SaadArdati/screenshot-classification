#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"
#include "../include/screenshot_utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

// Function declarations - these are implemented in feature_extraction.cu and knn.cu
extern "C" void extractFeaturesGPU(const unsigned char* h_images, int batch_size,
                                  int width, int height, int channels, 
                                  Feature* h_features);
extern "C" void classifyBatchGPU(const Feature* train_features, int train_size,
                                const Feature* query_features, int query_size,
                                int* predictions, double* computation_times);

// CUDA kernel for computing screenshot statistics
__global__ void computeScreenshotStatsKernel(
    const unsigned char* d_img, 
    int w, int h, int channels,
    int* d_edge_pixels,
    int* d_regular_edge_pixels,
    int* d_uniform_color_pixels,
    int* d_horizontal_edge_counts) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= w-1 || y >= h-1 || x < 1 || y < 1)
        return;
    
    const int idx = (y * w + x) * channels;
    
    // Get grayscale of current and neighboring pixels
    const unsigned char gray = (d_img[idx] + d_img[idx+1] + d_img[idx+2]) / 3;
    const unsigned char gray_left = (d_img[(y * w + (x-1)) * channels] + 
                                  d_img[(y * w + (x-1)) * channels + 1] + 
                                  d_img[(y * w + (x-1)) * channels + 2]) / 3;
    const unsigned char gray_right = (d_img[(y * w + (x+1)) * channels] + 
                                   d_img[(y * w + (x+1)) * channels + 1] + 
                                   d_img[(y * w + (x+1)) * channels + 2]) / 3;
    const unsigned char gray_up = (d_img[((y-1) * w + x) * channels] + 
                                d_img[((y-1) * w + x) * channels + 1] + 
                                d_img[((y-1) * w + x) * channels + 2]) / 3;
    const unsigned char gray_down = (d_img[((y+1) * w + x) * channels] + 
                                  d_img[((y+1) * w + x) * channels + 1] + 
                                  d_img[((y+1) * w + x) * channels + 2]) / 3;
    
    // Calculate horizontal and vertical gradients
    const int h_gradient = abs(gray_right - gray_left);
    const int v_gradient = abs(gray_down - gray_up);
    
    // Detect edges
    if (h_gradient > EDGE_THRESHOLD || v_gradient > EDGE_THRESHOLD) {
        atomicAdd(d_edge_pixels, 1);
        atomicAdd(&d_horizontal_edge_counts[y], 1);
        
        // Check for regular edges (straight lines common in UI)
        if ((h_gradient > EDGE_THRESHOLD && v_gradient < EDGE_THRESHOLD/2) || 
            (v_gradient > EDGE_THRESHOLD && h_gradient < EDGE_THRESHOLD/2)) {
            atomicAdd(d_regular_edge_pixels, 1);
        }
    }
    
    // Check for uniform color regions (common in UI backgrounds/panels)
    int local_variance = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (y+dy >= 0 && y+dy < h && x+dx >= 0 && x+dx < w) {
                const int local_idx = ((y+dy) * w + (x+dx)) * channels;
                const unsigned char local_gray = (d_img[local_idx] + d_img[local_idx+1] + d_img[local_idx+2]) / 3;
                local_variance += abs(gray - local_gray);
            }
        }
    }
    
    // Low local variance indicates uniform color region
    if (local_variance < 20) {
        atomicAdd(d_uniform_color_pixels, 1);
    }
}

// CUDA kernel for analyzing horizontal alignments
__global__ void analyzeGridAlignmentKernel(
    const int* d_horizontal_edge_counts,
    int h, int w,
    int* d_aligned_rows) {
    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= h-3 || y < 1)
        return;
    
    // Check for similar edge patterns in consecutive rows (indicates UI grid)
    if (d_horizontal_edge_counts[y] > 0 && 
        abs(d_horizontal_edge_counts[y] - d_horizontal_edge_counts[y+1]) < w * 0.05) {
        atomicAdd(d_aligned_rows, 1);
    }
}

// Compute screenshot statistics with CUDA
ScreenshotStats computeScreenshotStatisticsGPU(unsigned char *img, int w, int h, int channels) {
    ScreenshotStats stats = {0};
    int total_pixels = w * h;
    
    // Allocate device memory
    unsigned char* d_img;
    int* d_edge_pixels;
    int* d_regular_edge_pixels;
    int* d_uniform_color_pixels;
    int* d_horizontal_edge_counts;
    int* d_aligned_rows;
    
    CUDA_CHECK(cudaMalloc(&d_img, w * h * channels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_edge_pixels, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_regular_edge_pixels, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_uniform_color_pixels, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_horizontal_edge_counts, h * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_aligned_rows, sizeof(int)));
    
    // Initialize counters to 0
    CUDA_CHECK(cudaMemset(d_edge_pixels, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_regular_edge_pixels, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_uniform_color_pixels, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_horizontal_edge_counts, 0, h * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_aligned_rows, 0, sizeof(int)));
    
    // Copy image to device
    CUDA_CHECK(cudaMemcpy(d_img, img, w * h * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Launch kernels
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    
    computeScreenshotStatsKernel<<<gridSize, blockSize>>>(
        d_img, w, h, channels,
        d_edge_pixels, d_regular_edge_pixels, d_uniform_color_pixels,
        d_horizontal_edge_counts
    );
    CUDA_CHECK_KERNEL();
    
    // Launch grid alignment kernel
    int blockSizeAlign = 256;
    int gridSizeAlign = (h + blockSizeAlign - 1) / blockSizeAlign;
    
    analyzeGridAlignmentKernel<<<gridSizeAlign, blockSizeAlign>>>(
        d_horizontal_edge_counts, h, w, d_aligned_rows
    );
    CUDA_CHECK_KERNEL();
    
    // Copy results back to host
    int edge_pixels = 0, regular_edge_pixels = 0, uniform_color_pixels = 0, aligned_rows = 0;
    CUDA_CHECK(cudaMemcpy(&edge_pixels, d_edge_pixels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&regular_edge_pixels, d_regular_edge_pixels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&uniform_color_pixels, d_uniform_color_pixels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&aligned_rows, d_aligned_rows, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate final statistics (normalized to [0,1] range)
    float edge_density = (float)edge_pixels / total_pixels;
    float edge_regularity = edge_pixels > 0 ? (float)regular_edge_pixels / edge_pixels : 0;
    float grid_alignment = (float)aligned_rows / h;
    float color_uniformity = (float)uniform_color_pixels / total_pixels;
    
    // Combine metrics into simplified scores
    stats.edge_score = (edge_regularity * 0.6) + (edge_density * 0.2) + (grid_alignment * 0.2);
    stats.color_score = color_uniformity;
    stats.ui_element_score = edge_density * 0.5 + grid_alignment * 0.5;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_edge_pixels));
    CUDA_CHECK(cudaFree(d_regular_edge_pixels));
    CUDA_CHECK(cudaFree(d_uniform_color_pixels));
    CUDA_CHECK(cudaFree(d_horizontal_edge_counts));
    CUDA_CHECK(cudaFree(d_aligned_rows));
    
    return stats;
}

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
    
    // Check with statistical analysis first
    start_time = clock();
    ScreenshotStats stats = computeScreenshotStatisticsGPU(img, width, height, 3);
    int statistical_detection = isLikelyScreenshot(stats);
    
    // Extract features
    Feature query_feature;
    extractFeaturesGPU(img, 1, width, height, 3, &query_feature);
    end_time = clock();
    feature_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // If statistical detection is positive, skip kNN
    if (statistical_detection) {
        printf("Classification result for %s: SCREENSHOT (Statistical analysis)\n", image_path);
        printf("This image was detected as a screenshot by analyzing UI patterns\n");
        query_feature.label = 2; // Mark as detected by statistical analysis
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