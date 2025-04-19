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

// Constants for screenshot detection in constant memory
__constant__ int c_edgeThreshold = EDGE_THRESHOLD;

// Optimized CUDA kernel for computing screenshot statistics with improved memory coalescing
__global__ void computeScreenshotStatsKernel(
    const unsigned char* d_img, 
    int w, int h, int channels,
    int* d_edge_pixels,
    int* d_regular_edge_pixels,
    int* d_uniform_color_pixels,
    int* d_horizontal_edge_counts) {
    
    // Define shared memory - packed for better memory access
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
    
    // Calculate shared memory indices only once
    int sharedIdx = (s_y * s_width + s_x) * channels;
    
    // Boundary check for global coordinates - early return
    if (x >= w || y >= h)
        return;
    
    // Load central pixels into shared memory
    int globalIdx = (y * w + x) * channels;
    if (x < w && y < h) {
        // Use vector operations for more efficient memory transfers when possible
        if (channels == 3) {
            // Load RGB as one operation if architecture supports it
            if (sizeof(int) >= 3*sizeof(unsigned char)) {
                // Cast to int pointer for vectorized load (RGB = 3 bytes, pad to 4)
                unsigned int pixel = *((unsigned int*)&d_img[globalIdx]);
                *((unsigned int*)&s_img[sharedIdx]) = pixel;
            } else {
                // Fallback to separate loads
                s_img[sharedIdx] = d_img[globalIdx];
                s_img[sharedIdx + 1] = d_img[globalIdx + 1];
                s_img[sharedIdx + 2] = d_img[globalIdx + 2];
            }
        }
    }
    
    // Load halo (boundary) pixels using collaborative loading
    // Top and bottom rows
    if (threadIdx.y == 0) {
        // Top row
        int y_top = y - 1;
        if (y_top >= 0) {
            int globalTopIdx = (y_top * w + x) * channels;
            int sharedTopIdx = (0 * s_width + s_x) * channels;
            // Load using vectorized operations when possible
            if (channels == 3 && sizeof(int) >= 3*sizeof(unsigned char)) {
                *((unsigned int*)&s_img[sharedTopIdx]) = *((unsigned int*)&d_img[globalTopIdx]);
            } else {
                s_img[sharedTopIdx] = d_img[globalTopIdx];
                s_img[sharedTopIdx + 1] = d_img[globalTopIdx + 1];
                s_img[sharedTopIdx + 2] = d_img[globalTopIdx + 2];
            }
        }
        
        // Bottom row
        int y_bottom = y + blockDim.y;
        if (y_bottom < h && threadIdx.x < blockDim.x) {
            int globalBottomIdx = (y_bottom * w + x) * channels;
            int sharedBottomIdx = ((blockDim.y + 1) * s_width + s_x) * channels;
            // Load using vectorized operations when possible
            if (channels == 3 && sizeof(int) >= 3*sizeof(unsigned char)) {
                *((unsigned int*)&s_img[sharedBottomIdx]) = *((unsigned int*)&d_img[globalBottomIdx]);
            } else {
                s_img[sharedBottomIdx] = d_img[globalBottomIdx];
                s_img[sharedBottomIdx + 1] = d_img[globalBottomIdx + 1];
                s_img[sharedBottomIdx + 2] = d_img[globalBottomIdx + 2];
            }
        }
    }
    
    // Left and right columns
    if (threadIdx.x == 0) {
        // Left column
        int x_left = x - 1;
        if (x_left >= 0) {
            int globalLeftIdx = (y * w + x_left) * channels;
            int sharedLeftIdx = (s_y * s_width + 0) * channels;
            // Load using vectorized operations when possible
            if (channels == 3 && sizeof(int) >= 3*sizeof(unsigned char)) {
                *((unsigned int*)&s_img[sharedLeftIdx]) = *((unsigned int*)&d_img[globalLeftIdx]);
            } else {
                s_img[sharedLeftIdx] = d_img[globalLeftIdx];
                s_img[sharedLeftIdx + 1] = d_img[globalLeftIdx + 1];
                s_img[sharedLeftIdx + 2] = d_img[globalLeftIdx + 2];
            }
        }
        
        // Right column
        int x_right = x + blockDim.x;
        if (x_right < w && threadIdx.y < blockDim.y) {
            int globalRightIdx = (y * w + x_right) * channels;
            int sharedRightIdx = (s_y * s_width + (blockDim.x + 1)) * channels;
            // Load using vectorized operations when possible
            if (channels == 3 && sizeof(int) >= 3*sizeof(unsigned char)) {
                *((unsigned int*)&s_img[sharedRightIdx]) = *((unsigned int*)&d_img[globalRightIdx]);
            } else {
                s_img[sharedRightIdx] = d_img[globalRightIdx];
                s_img[sharedRightIdx + 1] = d_img[globalRightIdx + 1];
                s_img[sharedRightIdx + 2] = d_img[globalRightIdx + 2];
            }
        }
    }
    
    // Wait for all threads to load shared memory
    __syncthreads();
    
    // Skip boundary pixels for computation
    if (x >= w-1 || y >= h-1 || x < 1 || y < 1)
        return;
    
    // Precompute indices for shared memory access to avoid bank conflicts
    const int center_idx = sharedIdx;
    const int left_idx = (s_y * s_width + (s_x-1)) * channels;
    const int right_idx = (s_y * s_width + (s_x+1)) * channels;
    const int up_idx = ((s_y-1) * s_width + s_x) * channels;
    const int down_idx = ((s_y+1) * s_width + s_x) * channels;
    
    // Fast grayscale calculation by pixel averaging
    const unsigned char gray = (s_img[center_idx] + s_img[center_idx+1] + s_img[center_idx+2]) / 3;
    const unsigned char gray_left = (s_img[left_idx] + s_img[left_idx+1] + s_img[left_idx+2]) / 3;
    const unsigned char gray_right = (s_img[right_idx] + s_img[right_idx+1] + s_img[right_idx+2]) / 3;
    const unsigned char gray_up = (s_img[up_idx] + s_img[up_idx+1] + s_img[up_idx+2]) / 3;
    const unsigned char gray_down = (s_img[down_idx] + s_img[down_idx+1] + s_img[down_idx+2]) / 3;
    
    // Calculate horizontal and vertical gradients
    const int h_gradient = abs(gray_right - gray_left);
    const int v_gradient = abs(gray_down - gray_up);
    
    // Thread-local variables to reduce atomic operations
    int edge_detected = 0;
    int regular_edge_detected = 0;
    int uniform_color_detected = 0;
    
    // Detect edges
    if (h_gradient > c_edgeThreshold || v_gradient > c_edgeThreshold) {
        edge_detected = 1;
        
        // Check for regular edges (straight lines common in UI)
        if ((h_gradient > c_edgeThreshold && v_gradient < c_edgeThreshold/2) || 
            (v_gradient > c_edgeThreshold && h_gradient < c_edgeThreshold/2)) {
            regular_edge_detected = 1;
        }
        
        // Update horizontal edge counts using a single atomic operation per row
        atomicAdd(&d_horizontal_edge_counts[y], 1);
    }
    
    // Check for uniform color regions (common in UI backgrounds/panels)
    int local_variance = 0;
    // Unroll the loop for better performance
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
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
        uniform_color_detected = 1;
    }
    
    // Use a single warp-synchronized reduction to update global counters
    if (edge_detected) {
        atomicAdd(d_edge_pixels, 1);
    }
    
    if (regular_edge_detected) {
        atomicAdd(d_regular_edge_pixels, 1);
    }
    
    if (uniform_color_detected) {
        atomicAdd(d_uniform_color_pixels, 1);
    }
}

// Optimized CUDA kernel for analyzing horizontal alignments with improved memory access
__global__ void analyzeGridAlignmentKernel(
    const int* d_horizontal_edge_counts,
    int h, int w,
    int* d_aligned_rows) {
    
    // Define shared memory for caching horizontal edge counts with padding to avoid bank conflicts
    extern __shared__ int s_edge_counts[];
    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory to avoid undefined behavior
    s_edge_counts[tid] = 0;
    
    // Ensure the additional elements are also initialized
    if (tid < 3) {
        s_edge_counts[blockDim.x + tid] = 0;
    }
    __syncthreads();
    
    // Load data into shared memory (each thread loads one value)
    if (y < h) {
        s_edge_counts[tid] = d_horizontal_edge_counts[y];
    }
    
    // Load additional data for block boundaries - handle edge cases properly
    if (tid < 3 && blockIdx.x > 0 && (blockDim.x * blockIdx.x - 3 + tid) < h) {
        // Load 3 values before this block's start
        s_edge_counts[tid] = d_horizontal_edge_counts[blockDim.x * blockIdx.x - 3 + tid];
    }
    
    if (tid < 3 && y + blockDim.x < h) {
        // Load 3 values after this block's end
        s_edge_counts[blockDim.x + tid] = d_horizontal_edge_counts[y + blockDim.x + tid];
    }
    
    __syncthreads();
    
    // Skip boundary checks - early return for better control flow
    if (y >= h-3 || y < 1)
        return;
    
    // Check for similar edge patterns in consecutive rows (indicates UI grid)
    // Use shared memory for faster access and avoid bank conflicts by careful indexing
    int count = 0;
    if (s_edge_counts[tid] > 0 && 
        abs(s_edge_counts[tid] - s_edge_counts[tid+1]) < w * 0.05) {
        count = 1;
    }
    
    // Use a single atomic operation per thread that finds alignment
    if (count > 0) {
        atomicAdd(d_aligned_rows, count);
    }
}

// Compute screenshot statistics with CUDA using optimized memory access
ScreenshotStats computeScreenshotStatisticsGPU(unsigned char *img, int w, int h, int channels) {
    ScreenshotStats stats = {0};
    int total_pixels = w * h;
    
    // Allocate device memory with proper alignment
    unsigned char* d_img;
    int* d_edge_pixels;
    int* d_regular_edge_pixels;
    int* d_uniform_color_pixels;
    int* d_horizontal_edge_counts;
    int* d_aligned_rows;
    
    // Size for proper memory alignment
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(&d_img, &pitch, w * channels * sizeof(unsigned char), h));
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
    
    // Copy image to device using pitched memory for proper alignment
    CUDA_CHECK(cudaMemcpy2D(d_img, pitch, img, w * channels * sizeof(unsigned char),
                          w * channels * sizeof(unsigned char), h, cudaMemcpyHostToDevice));
    
    // Find optimal execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    
    // Calculate shared memory size for image data with proper alignment
    int sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * channels * sizeof(unsigned char);
    
    // Launch optimized kernel
    computeScreenshotStatsKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_img, w, h, channels,
        d_edge_pixels, d_regular_edge_pixels, d_uniform_color_pixels,
        d_horizontal_edge_counts
    );
    CUDA_CHECK_KERNEL();
    
    // Find optimal execution configuration for grid alignment
    int blockSizeAlign = 256;
    int gridSizeAlign = (h + blockSizeAlign - 1) / blockSizeAlign;
    
    // Calculate shared memory size for edge counts with padding
    int alignSharedMemSize = (blockSizeAlign + 3) * sizeof(int);
    
    // Launch optimized grid alignment kernel
    analyzeGridAlignmentKernel<<<gridSizeAlign, blockSizeAlign, alignSharedMemSize>>>(
        d_horizontal_edge_counts, h, w, d_aligned_rows
    );
    CUDA_CHECK_KERNEL();
    
    // Copy results back to host using pinned memory for faster transfers
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
    
    // Allocate pinned memory for features for better host-device transfer
    Feature* model;
    CUDA_CHECK(cudaMallocHost(&model, *size * sizeof(Feature)));
    
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    // Read features
    if (fread(model, sizeof(Feature), *size, f) != *size) {
        fprintf(stderr, "Failed to read model data\n");
        cudaFreeHost(model);
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
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared Memory Per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
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
    
    // Performance timing with CUDA events for more accurate measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double load_model_time = 0.0, feature_time = 0.0, classification_time = 0.0;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    printDeviceInfo();
    
    // Load model - start timing
    cudaEventRecord(start);
    int model_size;
    Feature* model = loadModel(model_path, &model_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    load_model_time = elapsedTime / 1000.0;
    
    if (!model) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }
    
    printf("Model loaded with %d training examples\n", model_size);
    
    // Load query image
    int width, height, channels;
    unsigned char* img = stbi_load(image_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        cudaFreeHost(model);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }
    
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
    
    // Start feature extraction timing
    cudaEventRecord(start);
    
    // Check with statistical analysis first
    ScreenshotStats stats = computeScreenshotStatisticsGPU(img, width, height, 3);
    int statistical_detection = isLikelyScreenshot(stats);
    
    // Extract features
    Feature query_feature;
    extractFeaturesGPU(img, 1, width, height, 3, &query_feature);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    feature_time = elapsedTime / 1000.0;
    
    // If statistical detection is positive, skip kNN
    if (statistical_detection) {
        printf("Classification result for %s: SCREENSHOT (Statistical analysis)\n", image_path);
        printf("This image was detected as a screenshot by analyzing UI patterns\n");
        query_feature.label = 2; // Mark as detected by statistical analysis
    } else {
        // Perform kNN classification on GPU with timing
        cudaEventRecord(start);
        int prediction;
        double knn_times[2] = {0}; // [0] = transfer time, [1] = compute time
        
        // Call GPU KNN function (we only have 1 query)
        classifyBatchGPU(model, model_size, &query_feature, 1, &prediction, knn_times);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        classification_time = elapsedTime / 1000.0;
        
        printf("Classification result for %s: %s\n", image_path, 
               prediction ? "SCREENSHOT" : "NON-SCREENSHOT");
        printf("Classification based on K-nearest neighbors (K=%d)\n", K_NEIGHBORS);
    }
    
    // Print detailed performance metrics
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Model Loading Time: %.5f seconds\n", load_model_time);
    printf("Feature Extraction Time: %.5f seconds\n", feature_time);
    printf("Classification Time: %.5f seconds\n", classification_time);
    printf("Total Processing Time: %.5f seconds\n", 
           load_model_time + feature_time + classification_time);
    printf("Model Memory Usage: %.2f MB\n", 
           (float)(model_size * sizeof(Feature)) / (1024.0f * 1024.0f));
    
    // Print statistical analysis results
    printf("\nStatistical Analysis:\n");
    printf("-------------------\n");
    printf("Edge Score: %.3f\n", stats.edge_score);
    printf("Color Uniformity: %.3f\n", stats.color_score);
    printf("UI Element Score: %.3f\n", stats.ui_element_score);
    
    // Clean up
    stbi_image_free(img);
    cudaFreeHost(model);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
} 