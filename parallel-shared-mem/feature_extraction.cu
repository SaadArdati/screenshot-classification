#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"

// Kernel for RGB to grayscale conversion with shared memory optimization
__global__ void rgbToGrayscaleKernel(const unsigned char* rgb, unsigned char* gray, 
                                    int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_pixels = width * height;
    
    for (int i = idx; i < total_pixels; i += stride) {
        int rgb_idx = i * channels;
        gray[i] = (rgb[rgb_idx] + rgb[rgb_idx + 1] + rgb[rgb_idx + 2]) / 3;
    }
}

// Kernel for histogram computation with shared memory optimization
__global__ void computeHistogramKernel(const unsigned char* gray, float* histogram, 
                                      int size) {
    __shared__ unsigned int temp_hist[NUM_BINS];
    
    // Initialize shared memory histogram bins to 0
    if (threadIdx.x < NUM_BINS) {
        temp_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple pixels
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        int bin = gray[i] * NUM_BINS / 256;
        atomicAdd(&temp_hist[bin], 1);
    }
    
    // Wait for all threads to finish
    __syncthreads();
    
    // Only threads with IDs < NUM_BINS write results back to global memory
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], (float)temp_hist[threadIdx.x]);
    }
}

// Function to extract features using GPU with shared memory optimizations
extern "C" void extractFeaturesGPU(const unsigned char* h_images, int batch_size,
                                  int width, int height, int channels, 
                                  Feature* h_features) {
    // Calculate total image size
    int img_size = width * height * channels;
    int gray_size = width * height;
    
    // Allocate device memory
    unsigned char* d_images;
    unsigned char* d_gray;
    float* d_histogram;
    
    CUDA_CHECK(cudaMalloc(&d_images, batch_size * img_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_gray, batch_size * gray_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(float)));
    
    // Copy images to device
    CUDA_CHECK(cudaMemcpy(d_images, h_images, batch_size * img_size * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    
    // Process each image in the batch
    for (int b = 0; b < batch_size; b++) {
        unsigned char* current_image = d_images + (b * img_size);
        unsigned char* current_gray = d_gray + (b * gray_size);
        
        // Clear histogram
        CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(float)));
        
        // Convert to grayscale
        int blockSize = 256;
        int gridSize = (gray_size + blockSize - 1) / blockSize;
        rgbToGrayscaleKernel<<<gridSize, blockSize>>>(current_image, current_gray, 
                                                     width, height, channels);
        CUDA_CHECK_KERNEL();
        
        // Compute histogram
        computeHistogramKernel<<<gridSize, blockSize>>>(current_gray, d_histogram, gray_size);
        CUDA_CHECK_KERNEL();
        
        // Copy histogram to host and normalize
        float histogram[NUM_BINS];
        CUDA_CHECK(cudaMemcpy(histogram, d_histogram, NUM_BINS * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Normalize and store in feature
        float total = width * height;
        for (int i = 0; i < NUM_BINS; i++) {
            h_features[b].bins[i] = histogram[i] / total;
        }
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_histogram));
} 