#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"
#include "../include/feature_extraction.cuh"
#include "../include/stb_image.h"

// Optimize block sizes based on hardware capabilities
#define BLOCK_SIZE_GRAYSCALE 256  // Multiple of 32 for warp efficiency
#define BLOCK_SIZE_HISTOGRAM 256  // Adjust for shared memory constraints
#define BLOCK_SIZE_DISTANCE 128   // Balance between occupancy and shared memory
#define PIXELS_PER_THREAD 4       // Process multiple pixels per thread
#define FEATURE_TILE_SIZE 16      // Tiles for feature processing
#define TEST_BATCH_SIZE 64        // Batch size for test samples
#define TRAIN_CHUNK_SIZE 1024     // Chunk size for training samples

// Optimized kernel for RGB to grayscale with coalesced memory access
__global__ void rgbToGrayCoalesced(unsigned char* rgb, unsigned char* gray, int width, int height) {
    __shared__ unsigned char shared_rgb[BLOCK_SIZE_GRAYSCALE * 3 * PIXELS_PER_THREAD];
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD;
    int threadInBlock = threadIdx.x;
    
    // Coalesced load to shared memory
    if (idx < width * height) {
        for (int i = 0; i < PIXELS_PER_THREAD; i++) {
            int pixel_idx = idx + i;
            if (pixel_idx < width * height) {
                int rgb_idx = pixel_idx * 3;
                int shared_idx = (threadInBlock * PIXELS_PER_THREAD + i) * 3;
                
                shared_rgb[shared_idx] = rgb[rgb_idx];
                shared_rgb[shared_idx + 1] = rgb[rgb_idx + 1];
                shared_rgb[shared_idx + 2] = rgb[rgb_idx + 2];
            }
        }
    }
    __syncthreads();
    
    // Compute grayscale using shared memory
    if (idx < width * height) {
        for (int i = 0; i < PIXELS_PER_THREAD; i++) {
            int pixel_idx = idx + i;
            if (pixel_idx < width * height) {
                int shared_idx = (threadInBlock * PIXELS_PER_THREAD + i) * 3;
                float r = shared_rgb[shared_idx];
                float g = shared_rgb[shared_idx + 1];
                float b = shared_rgb[shared_idx + 2];
                
                // More accurate grayscale conversion
                gray[pixel_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            }
        }
    }
}

// Optimized histogram kernel with shared memory reduction
__global__ void computeHistogramShared(unsigned char* gray, float* histogram, int size) {
    __shared__ float shared_hist[NUM_BINS];
    
    // Initialize shared histogram
    if (threadIdx.x < NUM_BINS) {
        shared_hist[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Process multiple pixels per thread with coalesced access
    int idx = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
    for (int i = 0; i < PIXELS_PER_THREAD; i++) {
        int pixel_idx = idx + i * blockDim.x;
        if (pixel_idx < size) {
            int bin = gray[pixel_idx] * NUM_BINS / 256;
            bin = min(bin, NUM_BINS - 1);  // Prevent overflow
            atomicAdd(&shared_hist[bin], 1.0f);
        }
    }
    __syncthreads();
    
    // Merge shared histogram to global histogram
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], shared_hist[threadIdx.x]);
    }
}

// Optimized distance calculation for a chunk of training samples
__global__ void calculateDistancesChunked(Feature* trainSet, Feature* testBatch, float* distances, 
                                        int trainSize, int trainStart, int trainChunkSize, int testBatchSize) {
    __shared__ float shared_test[TEST_BATCH_SIZE][NUM_BINS];
    
    int trainIdx = trainStart + blockIdx.x * blockDim.x + threadIdx.x;
    int chunkIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load test features into shared memory (once per block)
    if (threadIdx.x < testBatchSize) {
        for (int i = 0; i < NUM_BINS; i++) {
            shared_test[threadIdx.x][i] = testBatch[threadIdx.x].bins[i];
        }
    }
    __syncthreads();
    
    // Process distances only for valid indices
    if (trainIdx < trainSize && chunkIdx < trainChunkSize) {
        // For each test sample in the batch
        for (int t = 0; t < testBatchSize; t++) {
            float dist = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < NUM_BINS; k++) {
                float diff = trainSet[trainIdx].bins[k] - shared_test[t][k];
                dist += diff * diff;
            }
            distances[t * trainChunkSize + chunkIdx] = dist;
        }
    }
}

// Structure for memory pool to reduce allocation overhead
struct GPUMemoryPool {
    unsigned char* rgb_buffer;
    unsigned char* gray_buffer;
    float* histogram_buffer;
    size_t max_size;
    bool initialized;
    
    GPUMemoryPool() : rgb_buffer(nullptr), gray_buffer(nullptr), 
                     histogram_buffer(nullptr), max_size(0), initialized(false) {}
    
    ~GPUMemoryPool() {
        if (initialized) {
            cudaFree(rgb_buffer);
            cudaFree(gray_buffer);
            cudaFree(histogram_buffer);
        }
    }
    
    bool allocate(size_t size) {
        if (initialized && size <= max_size) return true;
        
        if (initialized) {
            cudaFree(rgb_buffer);
            cudaFree(gray_buffer);
            cudaFree(histogram_buffer);
        }
        
        if (cudaMalloc(&rgb_buffer, size * 3) != cudaSuccess) return false;
        if (cudaMalloc(&gray_buffer, size) != cudaSuccess) return false;
        if (cudaMalloc(&histogram_buffer, NUM_BINS * sizeof(float)) != cudaSuccess) return false;
        
        max_size = size;
        initialized = true;
        return true;
    }
};

// Feature extraction using optimized GPU functions
void extractFeaturesGPU(const char* imagePath, Feature* feature, void* optional_pool) {
    static GPUMemoryPool pool;  // Static to reuse across calls
    
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", imagePath);
        return;
    }
    
    int size = width * height;
    
    // Ensure memory pool is large enough
    if (!pool.allocate(size)) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        stbi_image_free(img);
        return;
    }
    
    // Initialize feature
    memset(feature, 0, sizeof(Feature));
    
    // Copy image to GPU
    CUDA_CHECK(cudaMemcpy(pool.rgb_buffer, img, size * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Convert to grayscale with optimized kernel
    int gridSizeGray = (size + BLOCK_SIZE_GRAYSCALE * PIXELS_PER_THREAD - 1) / (BLOCK_SIZE_GRAYSCALE * PIXELS_PER_THREAD);
    rgbToGrayCoalesced<<<gridSizeGray, BLOCK_SIZE_GRAYSCALE>>>(pool.rgb_buffer, pool.gray_buffer, width, height);
    
    // Clear histogram
    CUDA_CHECK(cudaMemset(pool.histogram_buffer, 0, NUM_BINS * sizeof(float)));
    
    // Compute histogram with optimized kernel
    int gridSizeHist = (size + BLOCK_SIZE_HISTOGRAM * PIXELS_PER_THREAD - 1) / (BLOCK_SIZE_HISTOGRAM * PIXELS_PER_THREAD);
    computeHistogramShared<<<gridSizeHist, BLOCK_SIZE_HISTOGRAM>>>(pool.gray_buffer, pool.histogram_buffer, size);
    
    // Copy histogram to CPU
    float histogram[NUM_BINS];
    CUDA_CHECK(cudaMemcpy(histogram, pool.histogram_buffer, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Normalize histogram on CPU
    float total = width * height;
    for (int i = 0; i < NUM_BINS; i++) {
        feature->bins[i] = histogram[i] / total;
    }
    
    stbi_image_free(img);
}

// Optimized chunked batch classification
int classifyBatchGPU(Feature* trainSet, int trainSize, Feature* testSet, int testSize, int* predictions) {
    if (trainSize == 0 || testSize == 0) return -1;
    
    printf("Starting batch classification with %d training samples and %d test samples...\n", trainSize, testSize);
    
    // Allocate GPU memory for a chunk of training data
    Feature* d_trainChunk;
    Feature* d_testBatch;
    float* d_distances;
    float* h_distances = (float*)malloc(TEST_BATCH_SIZE * TRAIN_CHUNK_SIZE * sizeof(float));
    
    if (!h_distances) return -1;
    
    CUDA_CHECK(cudaMalloc(&d_trainChunk, TRAIN_CHUNK_SIZE * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_testBatch, TEST_BATCH_SIZE * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_distances, TEST_BATCH_SIZE * TRAIN_CHUNK_SIZE * sizeof(float)));
    
    // Process test samples in batches
    for (int testStart = 0; testStart < testSize; testStart += TEST_BATCH_SIZE) {
        int currentTestBatchSize = min(TEST_BATCH_SIZE, testSize - testStart);
        
        printf("\rProcessing test samples %d-%d of %d...", testStart, testStart + currentTestBatchSize, testSize);
        fflush(stdout);
        
        // Copy current test batch to GPU
        CUDA_CHECK(cudaMemcpy(d_testBatch, &testSet[testStart], currentTestBatchSize * sizeof(Feature), cudaMemcpyHostToDevice));
        
        // Process training samples in chunks
        float* minDistances = (float*)malloc(currentTestBatchSize * sizeof(float));
        int* bestLabels = (int*)malloc(currentTestBatchSize * sizeof(int));
        
        for (int i = 0; i < currentTestBatchSize; i++) {
            minDistances[i] = INFINITY;
            bestLabels[i] = 0;
        }
        
        for (int trainStart = 0; trainStart < trainSize; trainStart += TRAIN_CHUNK_SIZE) {
            int currentTrainChunkSize = min(TRAIN_CHUNK_SIZE, trainSize - trainStart);
            
            // Copy current training chunk to GPU
            CUDA_CHECK(cudaMemcpy(d_trainChunk, &trainSet[trainStart], currentTrainChunkSize * sizeof(Feature), cudaMemcpyHostToDevice));
            
            // Calculate distances for this chunk
            int gridSize = (currentTrainChunkSize + BLOCK_SIZE_DISTANCE - 1) / BLOCK_SIZE_DISTANCE;
            calculateDistancesChunked<<<gridSize, BLOCK_SIZE_DISTANCE>>>(d_trainChunk, d_testBatch, d_distances, 
                                                                        currentTrainChunkSize, 0, currentTrainChunkSize, 
                                                                        currentTestBatchSize);
            
            // Synchronize and copy results back
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_distances, d_distances, currentTestBatchSize * currentTrainChunkSize * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Update minimum distances and labels
            for (int t = 0; t < currentTestBatchSize; t++) {
                for (int j = 0; j < currentTrainChunkSize; j++) {
                    float dist = h_distances[t * currentTrainChunkSize + j];
                    if (dist < minDistances[t]) {
                        minDistances[t] = dist;
                        bestLabels[t] = trainSet[trainStart + j].label;
                    }
                }
            }
        }
        
        // Set predictions for this batch
        for (int t = 0; t < currentTestBatchSize; t++) {
            predictions[testStart + t] = bestLabels[t];
        }
        
        free(minDistances);
        free(bestLabels);
    }
    
    printf("\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_trainChunk));
    CUDA_CHECK(cudaFree(d_testBatch));
    CUDA_CHECK(cudaFree(d_distances));
    free(h_distances);
    
    return 0;
}
