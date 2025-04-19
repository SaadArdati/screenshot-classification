#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"

// Constants in constant memory for faster access
__constant__ int c_numBins = NUM_BINS;
__constant__ int c_kNeighbors = K_NEIGHBORS;

// Kernel for distance calculation with improved memory coalescing and shared memory
__global__ void calculateDistancesKernel(const Feature* trainSet, 
                                        const Feature* queryFeature,
                                        float* distances, 
                                        int trainSize,
                                        int queryIdx) {
    // Use shared memory to cache the query feature
    __shared__ float query_bins[NUM_BINS];
    
    // Collaborative loading - each thread in the block helps load part of the query feature
    int tid = threadIdx.x;
    if (tid < NUM_BINS) {
        query_bins[tid] = queryFeature[queryIdx].bins[tid];
    }
    __syncthreads();
    
    // Each thread calculates distance for multiple training samples
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < trainSize; i += stride) {
        float dist = 0.0f;
        
        // Unroll loop for better performance with small fixed size
        #pragma unroll
        for (int k = 0; k < NUM_BINS; k++) {
            float diff = query_bins[k] - trainSet[i].bins[k];
            dist += diff * diff;
        }
        
        // Write result with coalesced memory access
        distances[queryIdx * trainSize + i] = dist;
    }
}

// Function to find k-nearest neighbors and return majority vote
// Optimized to use sorted insertion for better performance on small k values
int knnVote(float* distances, Feature* trainSet, int trainSize, int k) {
    // Allocate arrays for k smallest distances and their indices
    int* indices = (int*)malloc(k * sizeof(int));
    float* kDistances = (float*)malloc(k * sizeof(float));
    
    // Initialize with large values
    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        kDistances[i] = INFINITY;
    }
    
    // Find k smallest distances using a single pass
    for (int i = 0; i < trainSize; i++) {
        if (distances[i] < kDistances[k-1]) {
            // Use binary search to find insertion point for better performance
            int left = 0;
            int right = k - 1;
            
            while (left < right) {
                int mid = (left + right) / 2;
                if (kDistances[mid] < distances[i]) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            
            int insertPos = left;
            
            // Only shift what's needed
            for (int j = k - 1; j > insertPos; j--) {
                kDistances[j] = kDistances[j-1];
                indices[j] = indices[j-1];
            }
            
            kDistances[insertPos] = distances[i];
            indices[insertPos] = i;
        }
    }
    
    // Count votes for screenshot class
    int screenshot_votes = 0;
    for (int i = 0; i < k; i++) {
        if (indices[i] != -1 && trainSet[indices[i]].label == 1) {
            screenshot_votes++;
        }
    }
    
    // Free memory
    free(indices);
    free(kDistances);
    
    // Return majority vote
    return (screenshot_votes > k/2) ? 1 : 0;
}

// Function to classify images using KNN with GPU acceleration and optimized memory access
extern "C" void classifyBatchGPU(const Feature* train_features, int train_size,
                                const Feature* query_features, int query_size,
                                int* predictions, double* computation_times) {
    // Start timing for memory transfers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate device memory using page-locked memory where appropriate
    Feature* d_train_features;
    Feature* d_query_features;
    float* d_distances;
    
    // Use cudaMalloc for device memory
    CUDA_CHECK(cudaMalloc(&d_train_features, train_size * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_query_features, query_size * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_distances, query_size * train_size * sizeof(float)));
    
    // Copy data to device with proper alignment
    CUDA_CHECK(cudaMemcpy(d_train_features, train_features, train_size * sizeof(Feature),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query_features, query_features, query_size * sizeof(Feature),
                         cudaMemcpyHostToDevice));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transferTime;
    cudaEventElapsedTime(&transferTime, start, stop);
    computation_times[0] = transferTime / 1000.0; // Convert to seconds
    
    // Start timing for computation
    cudaEventRecord(start);
    
    // Find optimal execution configuration
    int blockSize = 256;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                                calculateDistancesKernel, 0, 0));
    int gridSize = min((train_size + blockSize - 1) / blockSize, 65535);
    
    // Process each query
    for (int q = 0; q < query_size; q++) {
        // Calculate distances using the optimized kernel
        calculateDistancesKernel<<<gridSize, blockSize>>>(d_train_features, d_query_features,
                                                        d_distances, train_size, q);
        CUDA_CHECK_KERNEL();
    }
    
    // Allocate page-locked host memory for faster transfers
    float* h_distances;
    CUDA_CHECK(cudaMallocHost(&h_distances, query_size * train_size * sizeof(float)));
    
    // Copy results back to host using page-locked memory
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, query_size * train_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);
    computation_times[1] = kernelTime / 1000.0; // Convert to seconds
    
    // Perform kNN voting for each query
    for (int q = 0; q < query_size; q++) {
        predictions[q] = knnVote(h_distances + (q * train_size), 
                               (Feature*)train_features, train_size, K_NEIGHBORS);
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_train_features));
    CUDA_CHECK(cudaFree(d_query_features));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFreeHost(h_distances));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
} 