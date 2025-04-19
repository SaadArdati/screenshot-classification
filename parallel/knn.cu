#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"

// Kernel for distance calculation with shared memory optimization
__global__ void calculateDistancesKernel(const Feature* trainSet, 
                                        const Feature* queryFeature,
                                        float* distances, 
                                        int trainSize,
                                        int queryIdx) {
    // Load query feature bins into shared memory
    __shared__ float query_bins[NUM_BINS];
    
    // Collaborative loading of query feature into shared memory
    if (threadIdx.x < NUM_BINS) {
        query_bins[threadIdx.x] = queryFeature[queryIdx].bins[threadIdx.x];
    }
    __syncthreads();
    
    // Each thread calculates distance for multiple training samples
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < trainSize; i += stride) {
        float dist = 0.0f;
        
        // Calculate Euclidean distance using the shared memory query feature
        for (int k = 0; k < NUM_BINS; k++) {
            float diff = query_bins[k] - trainSet[i].bins[k];
            dist += diff * diff;
        }
        
        // Store the distance
        distances[queryIdx * trainSize + i] = dist;
    }
}

// Function to find k-nearest neighbors and return majority vote
int knnVote(float* distances, Feature* trainSet, int trainSize, int k) {
    // Simple insertion sort to find k smallest distances
    int* indices = (int*)malloc(k * sizeof(int));
    float* kDistances = (float*)malloc(k * sizeof(float));
    
    // Initialize with large values
    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        kDistances[i] = INFINITY;
    }
    
    // Find k smallest distances
    for (int i = 0; i < trainSize; i++) {
        if (distances[i] < kDistances[k-1]) {
            // Insert into sorted list
            int j = k - 1;
            while (j > 0 && distances[i] < kDistances[j-1]) {
                kDistances[j] = kDistances[j-1];
                indices[j] = indices[j-1];
                j--;
            }
            kDistances[j] = distances[i];
            indices[j] = i;
        }
    }
    
    // Count votes
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

// Function to classify images using KNN with GPU acceleration and shared memory
extern "C" void classifyBatchGPU(const Feature* train_features, int train_size,
                                const Feature* query_features, int query_size,
                                int* predictions, double* computation_times) {
    // Start timing for memory transfers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate device memory
    Feature* d_train_features;
    Feature* d_query_features;
    float* d_distances;
    
    CUDA_CHECK(cudaMalloc(&d_train_features, train_size * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_query_features, query_size * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_distances, query_size * train_size * sizeof(float)));
    
    // Copy data to device
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
    
    // Find optimal block size for distance kernel
    int blockSize = 256;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                                calculateDistancesKernel, 0, 0));
    int gridSize = (train_size + blockSize - 1) / blockSize;
    
    // Process each query
    for (int q = 0; q < query_size; q++) {
        // Calculate distances using shared memory
        calculateDistancesKernel<<<gridSize, blockSize>>>(d_train_features, d_query_features,
                                                        d_distances, train_size, q);
        CUDA_CHECK_KERNEL();
    }
    
    // Allocate host memory for distances
    float* h_distances = (float*)malloc(query_size * train_size * sizeof(float));
    
    // Copy results back to host
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_distances);
} 