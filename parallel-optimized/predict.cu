#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"
#include "../include/feature_extraction.cuh"
#include "../include/stb_image.h"

// Threshold definitions for statistical analysis
#define EDGE_THRESHOLD 30
#define SCREENSHOT_THRESHOLD 0.5

// Function to check if image is likely a screenshot based on statistics
float computeFinalScore(const ScreenshotStats& stats) {
    return (stats.edge_score * 0.4 + stats.color_score * 0.3 + stats.ui_element_score * 0.3);
}

// Optimized CUDA kernel for computing screenshot statistics using shared memory
__global__ void computeScreenshotStatsKernel(
    const unsigned char* d_img, 
    int w, int h, int channels,
    int* d_edge_pixels,
    int* d_regular_edge_pixels,
    int* d_uniform_color_pixels,
    int* d_horizontal_edge_counts) {
    
    extern __shared__ unsigned char s_img[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    int s_height = blockDim.y + 2;
    
    if (x >= w || y >= h)
        return;
    
    // Load central pixels
    int globalIdx = (y * w + x) * channels;
    int sharedIdx = (s_y * s_width + s_x) * channels;
    
    if (x < w && y < h) {
        s_img[sharedIdx] = d_img[globalIdx];
        s_img[sharedIdx + 1] = d_img[globalIdx + 1];
        s_img[sharedIdx + 2] = d_img[globalIdx + 2];
    }
    
    // Load halo (boundary) pixels - only some threads load these
    if (threadIdx.y == 0 && y > 0) {
        int topIdx = ((y-1) * w + x) * channels;
        int sharedTopIdx = (0 * s_width + s_x) * channels;
        s_img[sharedTopIdx] = d_img[topIdx];
        s_img[sharedTopIdx + 1] = d_img[topIdx + 1];
        s_img[sharedTopIdx + 2] = d_img[topIdx + 2];
    }
    
    if (threadIdx.y == blockDim.y - 1 && y < h - 1) {
        int bottomIdx = ((y+1) * w + x) * channels;
        int sharedBottomIdx = ((blockDim.y + 1) * s_width + s_x) * channels;
        s_img[sharedBottomIdx] = d_img[bottomIdx];
        s_img[sharedBottomIdx + 1] = d_img[bottomIdx + 1];
        s_img[sharedBottomIdx + 2] = d_img[bottomIdx + 2];
    }
    
    if (threadIdx.x == 0 && x > 0) {
        int leftIdx = (y * w + (x-1)) * channels;
        int sharedLeftIdx = (s_y * s_width + 0) * channels;
        s_img[sharedLeftIdx] = d_img[leftIdx];
        s_img[sharedLeftIdx + 1] = d_img[leftIdx + 1];
        s_img[sharedLeftIdx + 2] = d_img[leftIdx + 2];
    }
    
    if (threadIdx.x == blockDim.x - 1 && x < w - 1) {
        int rightIdx = (y * w + (x+1)) * channels;
        int sharedRightIdx = (s_y * s_width + (blockDim.x + 1)) * channels;
        s_img[sharedRightIdx] = d_img[rightIdx];
        s_img[sharedRightIdx + 1] = d_img[rightIdx + 1];
        s_img[sharedRightIdx + 2] = d_img[rightIdx + 2];
    }
    
    __syncthreads();
    
    if (x >= w-1 || y >= h-1 || x < 1 || y < 1)
        return;
    
    // Compute grayscale and gradients using shared memory
    const unsigned char gray = (s_img[sharedIdx] + s_img[sharedIdx+1] + s_img[sharedIdx+2]) / 3;
    const unsigned char gray_left = (s_img[(s_y * s_width + (s_x-1)) * channels] + 
                                  s_img[(s_y * s_width + (s_x-1)) * channels + 1] + 
                                  s_img[(s_y * s_width + (s_x-1)) * channels + 2]) / 3;
    const unsigned char gray_right = (s_img[(s_y * s_width + (s_x+1)) * channels] + 
                                   s_img[(s_y * s_width + (s_x+1)) * channels + 1] + 
                                   s_img[(s_y * s_width + (s_x+1)) * channels + 2]) / 3;
    const unsigned char gray_up = (s_img[((s_y-1) * s_width + s_x) * channels] + 
                                s_img[((s_y-1) * s_width + s_x) * channels + 1] + 
                                s_img[((s_y-1) * s_width + s_x) * channels + 2]) / 3;
    const unsigned char gray_down = (s_img[((s_y+1) * s_width + s_x) * channels] + 
                                  s_img[((s_y+1) * s_width + s_x) * channels + 1] + 
                                  s_img[((s_y+1) * s_width + s_x) * channels + 2]) / 3;
    
    const int h_gradient = abs(gray_right - gray_left);
    const int v_gradient = abs(gray_down - gray_up);
    
    if (h_gradient > EDGE_THRESHOLD || v_gradient > EDGE_THRESHOLD) {
        atomicAdd(d_edge_pixels, 1);
        atomicAdd(&d_horizontal_edge_counts[y], 1);
        
        if ((h_gradient > EDGE_THRESHOLD && v_gradient < EDGE_THRESHOLD/2) || 
            (v_gradient > EDGE_THRESHOLD && h_gradient < EDGE_THRESHOLD/2)) {
            atomicAdd(d_regular_edge_pixels, 1);
        }
    }
    
    int local_variance = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int local_s_x = s_x + dx;
            int local_s_y = s_y + dy;
            const int local_idx = (local_s_y * s_width + local_s_x) * channels;
            const unsigned char local_gray = (s_img[local_idx] + s_img[local_idx+1] + s_img[local_idx+2]) / 3;
            local_variance += abs(gray - local_gray);
        }
    }
    
    if (local_variance < 20) {
        atomicAdd(d_uniform_color_pixels, 1);
    }
}

// Analyze grid alignment kernel
__global__ void analyzeGridAlignmentKernel(
    const int* d_horizontal_edge_counts,
    int h, int w,
    int* d_aligned_rows) {
    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= h-3 || y < 1)
        return;
    
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
    
    CUDA_CHECK(cudaMemset(d_edge_pixels, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_regular_edge_pixels, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_uniform_color_pixels, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_horizontal_edge_counts, 0, h * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_aligned_rows, 0, sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_img, img, w * h * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Launch kernels
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    
    int sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * channels * sizeof(unsigned char);
    
    computeScreenshotStatsKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_img, w, h, channels,
        d_edge_pixels, d_regular_edge_pixels, d_uniform_color_pixels,
        d_horizontal_edge_counts
    );
    CUDA_CHECK(cudaGetLastError());
    
    int blockSizeAlign = 256;
    int gridSizeAlign = (h + blockSizeAlign - 1) / blockSizeAlign;
    
    analyzeGridAlignmentKernel<<<gridSizeAlign, blockSizeAlign>>>(
        d_horizontal_edge_counts, h, w, d_aligned_rows
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy results back to host
    int edge_pixels = 0, regular_edge_pixels = 0, uniform_color_pixels = 0, aligned_rows = 0;
    CUDA_CHECK(cudaMemcpy(&edge_pixels, d_edge_pixels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&regular_edge_pixels, d_regular_edge_pixels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&uniform_color_pixels, d_uniform_color_pixels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&aligned_rows, d_aligned_rows, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate final statistics
    float edge_density = (float)edge_pixels / total_pixels;
    float edge_regularity = edge_pixels > 0 ? (float)regular_edge_pixels / edge_pixels : 0;
    float grid_alignment = (float)aligned_rows / h;
    float color_uniformity = (float)uniform_color_pixels / total_pixels;
    
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
int loadModel(const char* modelPath, Feature** trainSet, int* trainSize) {
    FILE* f = fopen(modelPath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model file: %s\n", modelPath);
        return -1;
    }
    
    if (fread(trainSize, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read model size\n");
        fclose(f);
        return -1;
    }
    
    *trainSet = (Feature*)malloc(*trainSize * sizeof(Feature));
    if (!*trainSet) {
        fprintf(stderr, "Failed to allocate memory for model\n");
        fclose(f);
        return -1;
    }
    
    if (fread(*trainSet, sizeof(Feature), *trainSize, f) != (size_t)*trainSize) {
        fprintf(stderr, "Failed to read model features\n");
        free(*trainSet);
        fclose(f);
        return -1;
    }
    
    fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_file> <image1> [<image2> ...]\n", argv[0]);
        return 1;
    }
    
    const char* modelPath = argv[1];
    
    // Performance timing
    clock_t start_time, end_time;
    double load_model_time = 0.0, feature_time = 0.0, classification_time = 0.0;
    
    // Initialize CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA capable devices found!\n");
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("CUDA Device Information:\n");
    printf("------------------------\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    
    // Load the trained model
    start_time = clock();
    Feature* trainSet = NULL;
    int trainSize = 0;
    
    if (loadModel(modelPath, &trainSet, &trainSize) != 0) {
        return 1;
    }
    end_time = clock();
    load_model_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Model loaded with %d training examples\n", trainSize);
    
    // Process each input image
    int numImages = argc - 2;
    
    for (int i = 0; i < numImages; i++) {
        const char* imagePath = argv[i + 2];
        
        // Load image
        int width, height, channels;
        unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 3);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", imagePath);
            continue;
        }
        
        printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
        
        // Feature extraction and statistical analysis
        start_time = clock();
        ScreenshotStats stats = computeScreenshotStatisticsGPU(img, width, height, 3);
        Feature testFeature;
        extractFeaturesGPU(imagePath, &testFeature);
        end_time = clock();
        feature_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        
        // Compute final score and threshold check
        float final_score = computeFinalScore(stats);
        
        // Display screenshot analysis
        printf("Screenshot Analysis:\n");
        printf("Edge Score: %.3f\n", stats.edge_score);
        printf("Color Score: %.3f\n", stats.color_score);
        printf("UI Element Score: %.3f\n", stats.ui_element_score);
        printf("Final Score: %.3f (Threshold: %.1f)\n", final_score, SCREENSHOT_THRESHOLD);
        
        // Classification
        int prediction = 0;
        start_time = clock();
        
        // Perform batch classification for one image
        if (classifyBatchGPU(trainSet, trainSize, &testFeature, 1, &prediction) != 0) {
            fprintf(stderr, "Classification failed for %s\n", imagePath);
            stbi_image_free(img);
            continue;
        }
        
        end_time = clock();
        classification_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        
        printf("Classification result for %s: %s\n", imagePath, 
               prediction ? "SCREENSHOT" : "NON-SCREENSHOT");
        printf("Classification based on K-nearest neighbors (K=%d)\n", 3);  // Assuming K=3 from your example
        
        // Print performance metrics
        printf("Performance Metrics:\n");
        printf("-------------------\n");
        printf("Model Loading Time: %.5f seconds\n", load_model_time);
        printf("Feature Extraction Time: %.5f seconds\n", feature_time);
        printf("Classification Time: %.5f seconds\n", classification_time);
        printf("Total Processing Time: %.5f seconds\n", 
               load_model_time + feature_time + classification_time);
        printf("Model Memory Usage: %.2f MB\n", 
               (float)(trainSize * sizeof(Feature)) / (1024.0f * 1024.0f));
        
        stbi_image_free(img);
    }
    
    // Cleanup
    free(trainSet);
    
    return 0;
}