#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"

// Constants in constant memory for faster access
__constant__ int c_numBins = NUM_BINS;

// Kernel for RGB to grayscale conversion with coalesced memory access
__global__ void rgbToGrayscaleKernel(const unsigned char* rgb, unsigned char* gray, 
                                    int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_pixels = width * height;
    
    // Each thread processes multiple pixels in a strided fashion for better memory coalescing
    for (int i = idx; i < total_pixels; i += stride) {
        // RGB is stored as RGBRGBRGB... in sequential memory, access with offset for coalescing
        int rgb_idx = i * channels;
        gray[i] = (rgb[rgb_idx] + rgb[rgb_idx + 1] + rgb[rgb_idx + 2]) / 3;
    }
}

// Kernel for histogram computation with improved shared memory usage
__global__ void computeHistogramKernel(const unsigned char* gray, float* histogram, 
                                      int size) {
    // Using shared memory for per-block histogram
    __shared__ unsigned int temp_hist[NUM_BINS];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory histogram to zeros - efficient parallel init
    if (tid < NUM_BINS) {
        temp_hist[tid] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple pixels in a coalesced pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple pixels per thread for better occupancy
    for (int i = idx; i < size; i += stride) {
        // Compute bin index - use constant memory for faster access
        int bin = gray[i] * c_numBins / 256;
        // Update shared memory histogram - unavoidable atomic op but in shared memory
        atomicAdd(&temp_hist[bin], 1);
    }
    
    // Ensure all threads in block finished updating shared histogram
    __syncthreads();
    
    // Only threads with IDs < NUM_BINS write results back to global memory
    // This reduces global atomic operations significantly
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], (float)temp_hist[tid]);
    }
}

// Function to extract features using GPU with optimized memory access
extern "C" void extractFeaturesGPU(const unsigned char* h_images, int batch_size,
                                  int width, int height, int channels, 
                                  Feature* h_features) {
    // Calculate total image size, ensuring proper alignment
    int img_size = width * height * channels;
    int gray_size = width * height;
    
    // Allocate device memory with proper alignment
    unsigned char* d_images;
    unsigned char* d_gray;
    float* d_histogram;
    
    // Use cudaMalloc for device memory
    CUDA_CHECK(cudaMalloc(&d_images, batch_size * img_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_gray, batch_size * gray_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(float)));
    
    // Copy images to device - ensure memory alignment
    CUDA_CHECK(cudaMemcpy(d_images, h_images, batch_size * img_size * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    
    // Find optimal execution configuration
    int blockSize = 256;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               rgbToGrayscaleKernel, 0, 0));
    
    // Process each image in the batch
    for (int b = 0; b < batch_size; b++) {
        unsigned char* current_image = d_images + (b * img_size);
        unsigned char* current_gray = d_gray + (b * gray_size);
        
        // Clear histogram memory
        CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(float)));
        
        // Calculate grid size based on optimal occupancy
        int gridSize = min((gray_size + blockSize - 1) / blockSize, 65535);
        
        // Convert to grayscale with optimized kernel
        rgbToGrayscaleKernel<<<gridSize, blockSize>>>(current_image, current_gray, 
                                                     width, height, channels);
        CUDA_CHECK_KERNEL();
        
        // Compute histogram with optimized kernel
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