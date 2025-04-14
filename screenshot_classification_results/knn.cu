#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "common.h"

// Structure for storing distance and label pairs
typedef struct {
    float distance;
    int label;
} DistanceLabel;

// CUDA kernel for computing distances between one query and all training examples
__global__ void computeDistancesKernel(
    const Feature* train_features,
    const Feature* query_feature,
    DistanceLabel* distances,
    int train_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= train_size) return;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NUM_BINS; i++) {
        const float diff = train_features[idx].bins[i] - query_feature->bins[i];
        sum += diff * diff;
    }

    distances[idx].distance = sqrtf(sum);
    distances[idx].label = train_features[idx].label;
}

// CUDA kernel for parallel reduction to find K nearest neighbors
__global__ void findTopKKernel(
    DistanceLabel* distances,
    int n,
    DistanceLabel* top_k,
    int k
) {
    extern __shared__ DistanceLabel shared_distances[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory with maximum values
    if (tid < k) {
        shared_distances[tid].distance = INFINITY;
        shared_distances[tid].label = -1;
    }
    __syncthreads();

    // Each thread processes one element
    if (gid < n) {
        // Insert into local top-k if distance is smaller
        for (int i = 0; i < k; i++) {
            if (distances[gid].distance < shared_distances[i].distance) {
                // Shift elements to make room
                for (int j = k-1; j > i; j--) {
                    shared_distances[j] = shared_distances[j-1];
                }
                
                // Insert new element
                shared_distances[i] = distances[gid];
                break;
            }
        }
    }

    __syncthreads();

    // First thread writes results to global memory
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            top_k[blockIdx.x * k + i] = shared_distances[i];
        }
    }
}

// CUDA kernel for majority voting - FIXED VERSION
__global__ void majorityVoteKernel(
    DistanceLabel* top_k,
    int* predictions,
    int batch_size,
    int k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Count votes for each class (0 or 1)
    int votes_for_screenshot = 0;
    for (int i = 0; i < k; i++) {
        if (top_k[idx * k + i].label == 1) {
            votes_for_screenshot++;
        }
    }

    // Majority vote threshold should match CPU implementation
    // In CPU: result = (votes >= (K_NEIGHBORS / 2 + 1)) ? 1 : 0;
    predictions[idx] = (votes_for_screenshot >= (k / 2 + 1)) ? 1 : 0;
}

// Host function to classify a batch of query features
extern "C" void classifyBatchGPU(
    const Feature* train_features,
    int train_size,
    const Feature* query_features,
    int query_size,
    int* predictions,
    double* computation_times
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    Feature* d_train_features = allocateDeviceMemory<Feature>(train_size);
    Feature* d_query_features = allocateDeviceMemory<Feature>(query_size);
    DistanceLabel* d_distances = allocateDeviceMemory<DistanceLabel>(train_size * query_size);
    DistanceLabel* d_top_k = allocateDeviceMemory<DistanceLabel>(query_size * K_NEIGHBORS);
    int* d_predictions = allocateDeviceMemory<int>(query_size);
    
    // Copy training data to device (only once)
    cudaEventRecord(start);
    copyToDevice(d_train_features, train_features, train_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_time;
    cudaEventElapsedTime(&transfer_time, start, stop);
    computation_times[0] = transfer_time / 1000.0; // Convert to seconds
    
    // Copy query features to device
    cudaEventRecord(start);
    copyToDevice(d_query_features, query_features, query_size);
    
    // Calculate grid and block dimensions
    const int block_size = THREADS_PER_BLOCK;
    const int num_blocks_distance = (train_size + block_size - 1) / block_size;
    const int num_blocks_query = (query_size + block_size - 1) / block_size;
    
    // Process each query
    for (int i = 0; i < query_size; i++) {
        // Compute distances
        computeDistancesKernel<<<num_blocks_distance, block_size>>>(
            d_train_features,
            &d_query_features[i],
            &d_distances[i * train_size],
            train_size
        );
        CUDA_CHECK_KERNEL();
        
        // Find top-K nearest neighbors
        findTopKKernel<<<num_blocks_query, block_size, K_NEIGHBORS * sizeof(DistanceLabel)>>>(
            &d_distances[i * train_size],
            train_size,
            &d_top_k[i * K_NEIGHBORS],
            K_NEIGHBORS
        );
        CUDA_CHECK_KERNEL();
    }
    
    // Perform majority voting
    majorityVoteKernel<<<num_blocks_query, block_size>>>(
        d_top_k,
        d_predictions,
        query_size,
        K_NEIGHBORS
    );
    CUDA_CHECK_KERNEL();
    
    // Copy results back to host
    copyToHost(predictions, d_predictions, query_size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float compute_time;
    cudaEventElapsedTime(&compute_time, start, stop);
    computation_times[1] = compute_time / 1000.0; // Convert to seconds
    
    // Clean up
    freeDeviceMemory(d_train_features);
    freeDeviceMemory(d_query_features);
    freeDeviceMemory(d_distances);
    freeDeviceMemory(d_top_k);
    freeDeviceMemory(d_predictions);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
