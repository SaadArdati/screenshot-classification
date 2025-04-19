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

// Optimized CUDA kernel for computing screenshot statistics using shared memory
__global__ void computeScreenshotStatsKernel(
    const unsigned char* d_img, 
    int w, int h, int channels,
    int* d_edge_pixels,
    int* d_regular_edge_pixels,
    int* d_uniform_color_pixels,
    int* d_horizontal_edge_counts) {
    
    // Define shared memory - will cache part of the image
    extern __shared__ unsigned char s_img[];
    
    // Calculate global and local coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local coordinates within the shared memory block
    int s_x = threadIdx.x + 1;  // Add 1 for padding
    int s_y = threadIdx.y + 1;  // Add 1 for padding
    
    // Tile dimensions (including padding)
    int s_width = blockDim.x + 2;
    int s_height = blockDim.y + 2;
    
    // Boundary check for global coordinates
    if (x >= w || y >= h)
        return;
    
    // Load central pixels (main block)
    int globalIdx = (y * w + x) * channels;
    int sharedIdx = (s_y * s_width + s_x) * channels;
    
    if (x < w && y < h) {
        s_img[sharedIdx] = d_img[globalIdx];
        s_img[sharedIdx + 1] = d_img[globalIdx + 1];
        s_img[sharedIdx + 2] = d_img[globalIdx + 2];
    }
    
    // Load halo (boundary) pixels - top and bottom
    if (threadIdx.y == 0) {
        // Top row
        int y_top = y - 1;
        if (y_top >= 0) {
            int globalTopIdx = (y_top * w + x) * channels;
            int sharedTopIdx = (0 * s_width + s_x) * channels;
            s_img[sharedTopIdx] = d_img[globalTopIdx];
            s_img[sharedTopIdx + 1] = d_img[globalTopIdx + 1];
            s_img[sharedTopIdx + 2] = d_img[globalTopIdx + 2];
        }
        
        // Bottom row
        int y_bottom = y + blockDim.y;
        if (y_bottom < h && threadIdx.x < blockDim.x) {
            int globalBottomIdx = (y_bottom * w + x) * channels;
            int sharedBottomIdx = ((blockDim.y + 1) * s_width + s_x) * channels;
            s_img[sharedBottomIdx] = d_img[globalBottomIdx];
            s_img[sharedBottomIdx + 1] = d_img[globalBottomIdx + 1];
            s_img[sharedBottomIdx + 2] = d_img[globalBottomIdx + 2];
        }
    }
    
    // Load halo (boundary) pixels - left and right
    if (threadIdx.x == 0) {
        // Left column
        int x_left = x - 1;
        if (x_left >= 0) {
            int globalLeftIdx = (y * w + x_left) * channels;
            int sharedLeftIdx = (s_y * s_width + 0) * channels;
            s_img[sharedLeftIdx] = d_img[globalLeftIdx];
            s_img[sharedLeftIdx + 1] = d_img[globalLeftIdx + 1];
            s_img[sharedLeftIdx + 2] = d_img[globalLeftIdx + 2];
        }
        
        // Right column
        int x_right = x + blockDim.x;
        if (x_right < w && threadIdx.y < blockDim.y) {
            int globalRightIdx = (y * w + x_right) * channels;
            int sharedRightIdx = (s_y * s_width + (blockDim.x + 1)) * channels;
            s_img[sharedRightIdx] = d_img[globalRightIdx];
            s_img[sharedRightIdx + 1] = d_img[globalRightIdx + 1];
            s_img[sharedRightIdx + 2] = d_img[globalRightIdx + 2];
        }
    }
    
    // Wait for all threads to load shared memory
    __syncthreads();
    
    // Skip boundary pixels for computation
    if (x >= w-1 || y >= h-1 || x < 1 || y < 1)
        return;
    
    // Get grayscale of current and neighboring pixels from shared memory
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
            int local_s_x = s_x + dx;
            int local_s_y = s_y + dy;
            const int local_idx = (local_s_y * s_width + local_s_x) * channels;
            const unsigned char local_gray = (s_img[local_idx] + s_img[local_idx+1] + s_img[local_idx+2]) / 3;
            local_variance += abs(gray - local_gray);
        }
    }
    
    // Low local variance indicates uniform color region
    if (local_variance < 20) {
        atomicAdd(d_uniform_color_pixels, 1);
    }
}

// Optimized CUDA kernel for analyzing horizontal alignments with shared memory
__global__ void analyzeGridAlignmentKernel(
    const int* d_horizontal_edge_counts,
    int h, int w,
    int* d_aligned_rows) {
    
    // Define shared memory for caching horizontal edge counts
    extern __shared__ int s_edge_counts[];
    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory (each thread loads one value)
    if (y < h) {
        s_edge_counts[tid] = d_horizontal_edge_counts[y];
    }
    
    // Load additional data for block boundaries
    if (tid < 3 && blockIdx.x > 0 && (blockDim.x * blockIdx.x - 3 + tid) < h) {
        // Load 3 values before this block's start
        s_edge_counts[tid] = d_horizontal_edge_counts[blockDim.x * blockIdx.x - 3 + tid];
    }
    
    if (tid < 3 && y + blockDim.x < h) {
        // Load 3 values after this block's end
        s_edge_counts[blockDim.x + tid] = d_horizontal_edge_counts[y + blockDim.x + tid];
    }
    
    __syncthreads();
    
    // Skip boundary checks
    if (y >= h-3 || y < 1)
        return;
    
    // Check for similar edge patterns in consecutive rows (indicates UI grid)
    // Use shared memory for faster access
    int local_tid = tid;
    if (s_edge_counts[local_tid] > 0 && 
        abs(s_edge_counts[local_tid] - s_edge_counts[local_tid+1]) < w * 0.05) {
        atomicAdd(d_aligned_rows, 1);
    }
}

// Compute screenshot statistics with CUDA using shared memory
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
    
    // Launch kernels with shared memory
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    
    // Calculate shared memory size for image data - include padding for halo
    int sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * channels * sizeof(unsigned char);
    
    computeScreenshotStatsKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_img, w, h, channels,
        d_edge_pixels, d_regular_edge_pixels, d_uniform_color_pixels,
        d_horizontal_edge_counts
    );
    CUDA_CHECK_KERNEL();
    
    // Launch grid alignment kernel with shared memory
    int blockSizeAlign = 256;
    int gridSizeAlign = (h + blockSizeAlign - 1) / blockSizeAlign;
    
    // Shared memory size for edge counts (including padding)
    int alignSharedMemSize = (blockSizeAlign + 3) * sizeof(int);
    
    analyzeGridAlignmentKernel<<<gridSizeAlign, blockSizeAlign, alignSharedMemSize>>>(
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