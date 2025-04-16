#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"

// Atomic add operation for float values (for histogram computation)
__device__ void atomicAdd_float(float* address, float val) {
    atomicAdd(address, val);
}

// CUDA kernel for calculating edge statistics
__global__ void calculateEdgeStatisticsKernel(
    const unsigned char* d_input,
    int width,
    int height,
    int channels,
    ScreenshotStats* d_stats
) {
    extern __shared__ int shared_data[];
    int* horizontal_edges = shared_data;
    int* edge_pixels_count = shared_data + height;
    int* regular_edge_count = shared_data + height + 1;
    int* uniform_color_count = shared_data + height + 2;
    
    // Initialize shared memory
    const int tid = threadIdx.x;
    if (tid < height) {
        horizontal_edges[tid] = 0;
    }
    if (tid == 0) {
        edge_pixels_count[0] = 0;
        regular_edge_count[0] = 0;
        uniform_color_count[0] = 0;
    }
    __syncthreads();
    
    // Calculate global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_pixels = width * height;
    
    // Process pixels in stride
    for (int i = idx; i < total_pixels; i += stride) {
        if (i < total_pixels) {
            // Calculate pixel position
            const int y = i / width;
            const int x = i % width;
            const int base_idx = i * channels;
            
            // Skip border pixels
            if (x > 0 && x < width-1 && y > 0 && y < height-1) {
                // Get current pixel grayscale
                const unsigned char gray = (
                    d_input[base_idx] +
                    d_input[base_idx + 1] +
                    d_input[base_idx + 2]
                ) / 3;
                
                // Get neighboring pixels
                const int left_idx = ((y * width) + (x-1)) * channels;
                const int right_idx = ((y * width) + (x+1)) * channels;
                const int up_idx = (((y-1) * width) + x) * channels;
                const int down_idx = (((y+1) * width) + x) * channels;
                
                const unsigned char gray_left = (
                    d_input[left_idx] +
                    d_input[left_idx + 1] +
                    d_input[left_idx + 2]
                ) / 3;
                
                const unsigned char gray_right = (
                    d_input[right_idx] +
                    d_input[right_idx + 1] +
                    d_input[right_idx + 2]
                ) / 3;
                
                const unsigned char gray_up = (
                    d_input[up_idx] +
                    d_input[up_idx + 1] +
                    d_input[up_idx + 2]
                ) / 3;
                
                const unsigned char gray_down = (
                    d_input[down_idx] +
                    d_input[down_idx + 1] +
                    d_input[down_idx + 2]
                ) / 3;
                
                // Calculate gradients
                const int h_gradient = abs(gray_right - gray_left);
                const int v_gradient = abs(gray_down - gray_up);
                
                // Detect edges
                if (h_gradient > EDGE_THRESHOLD || v_gradient > EDGE_THRESHOLD) {
                    // Count edges
                    atomicAdd(&edge_pixels_count[0], 1);
                    
                    // Add to horizontal edge count for this row
                    atomicAdd(&horizontal_edges[y], 1);
                    
                    // Check for regular edges (straight lines)
                    if ((h_gradient > EDGE_THRESHOLD && v_gradient < EDGE_THRESHOLD/2) ||
                        (v_gradient > EDGE_THRESHOLD && h_gradient < EDGE_THRESHOLD/2)) {
                        atomicAdd(&regular_edge_count[0], 1);
                    }
                }
                
                // Check for uniform color regions
                int local_variance = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        const int local_idx = ((y+dy) * width + (x+dx)) * channels;
                        const unsigned char local_gray = (
                            d_input[local_idx] +
                            d_input[local_idx + 1] +
                            d_input[local_idx + 2]
                        ) / 3;
                        
                        local_variance += abs(gray - local_gray);
                    }
                }
                
                // Low local variance indicates uniform color region
                if (local_variance < 20) {
                    atomicAdd(&uniform_color_count[0], 1);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Combine results
    if (tid == 0) {
        int total_edges = edge_pixels_count[0];
        int aligned_rows = 0;
        
        // Check for grid alignment
        for (int y = 1; y < height-3; y++) {
            if (horizontal_edges[y] > 0 && 
                abs(horizontal_edges[y] - horizontal_edges[y+1]) < width * 0.05) {
                aligned_rows++;
            }
        }
        
        // Calculate statistics
        float edge_density = (float)total_edges / (float)total_pixels;
        float edge_regularity = total_edges > 0 ? (float)regular_edge_count[0] / (float)total_edges : 0;
        float grid_alignment = (float)aligned_rows / (float)height;
        float color_uniformity = (float)uniform_color_count[0] / (float)total_pixels;
        
        // Set output statistics
        d_stats->edge_score = (edge_regularity * 0.6f) + (edge_density * 0.2f) + (grid_alignment * 0.2f);
        d_stats->color_score = color_uniformity;
        d_stats->ui_element_score = edge_density * 0.5f + grid_alignment * 0.5f;
    }
}

// CUDA kernel for grayscale conversion and histogram calculation
__global__ void computeHistogramsKernel(
    const unsigned char* image,
    int width,
    int height,
    int channels,
    Feature* features
) {
    extern __shared__ int shared_histograms[];
    
    // Initialize shared memory histograms
    const int tid = threadIdx.x;
    const int total_bins = NUM_BINS * 4; // 4 histograms
    
    for (int i = tid; i < total_bins; i += blockDim.x) {
        shared_histograms[i] = 0;
    }
    __syncthreads();
    
    // Divide histograms in shared memory
    int* hist_bins = shared_histograms;
    int* edge_bins = shared_histograms + NUM_BINS;
    int* top_bins = shared_histograms + NUM_BINS * 2;
    int* bottom_bins = shared_histograms + NUM_BINS * 3;
    
    // Process pixels in parallel
    const int num_pixels = width * height;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = blockIdx.x * blockDim.x + tid; i < num_pixels; i += stride) {
        const int y = i / width;
        const int x = i % width;
        
        // Calculate grayscale value
        const int idx = (y * width + x) * channels;
        const unsigned char gray = (image[idx] + image[idx + 1] + image[idx + 2]) / 3;
        const int bin = gray * NUM_BINS / 256;
        
        // Add to main histogram (using atomic operations)
        atomicAdd(&hist_bins[bin], 1);
        
        // Top region processing
        if (y < height * 0.05) {
            atomicAdd(&top_bins[bin], 1);
        }
        
        // Bottom region processing
        if (y > height * 0.9) {
            atomicAdd(&bottom_bins[bin], 1);
        }
        
        // Edge detection
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            // Horizontal gradient
            const int left_idx = (y * width + (x-1)) * channels;
            const unsigned char left_gray = (image[left_idx] + image[left_idx + 1] + image[left_idx + 2]) / 3;
            const int h_diff = abs(gray - left_gray);
            
            // Vertical gradient
            const int up_idx = ((y-1) * width + x) * channels;
            const unsigned char up_gray = (image[up_idx] + image[up_idx + 1] + image[up_idx + 2]) / 3;
            const int v_diff = abs(gray - up_gray);
            
            // Detect edges
            const int diff = max(h_diff, v_diff);
            if (diff > EDGE_THRESHOLD) {
                atomicAdd(&edge_bins[bin], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Final reduction - first thread normalizes and writes to global memory
    if (tid == 0) {
        Feature f = {0};
        const float total_pixels = width * height;
        const float top_region_size = total_pixels * 0.05f;
        const float bottom_region_size = total_pixels * 0.1f;
        
        for (int i = 0; i < NUM_BINS; i++) {
            f.bins[i] = hist_bins[i] / total_pixels;
            f.edge_bins[i] = edge_bins[i] / total_pixels;
            f.top_region_bins[i] = top_bins[i] / top_region_size;
            f.bottom_region_bins[i] = bottom_bins[i] / bottom_region_size;
        }
        
        // Copy to output
        features[blockIdx.x] = f;
    }
}

// Host function to determine if an image is likely a screenshot
bool isLikelyScreenshot(ScreenshotStats stats) {
    // Calculate weighted score
    float score = 0;
    score += stats.edge_score * 0.4f;
    score += stats.color_score * 0.3f;
    score += stats.ui_element_score * 0.3f;
    
    return score > SCREENSHOT_SCORE_THRESHOLD;
}

// Host function to extract features from a batch of images
extern "C" void extractFeaturesGPU(
    const unsigned char* h_images,
    int batch_size,
    int width,
    int height,
    int channels,
    Feature* h_features
) {
    // Allocate device memory
    unsigned char* d_images;
    Feature* d_features;
    
    const size_t image_size = width * height * channels * sizeof(unsigned char);
    const size_t total_image_size = batch_size * image_size;
    
    CUDA_CHECK(cudaMalloc(&d_images, total_image_size));
    CUDA_CHECK(cudaMalloc(&d_features, batch_size * sizeof(Feature)));
    
    // Copy images to device
    CUDA_CHECK(cudaMemcpy(d_images, h_images, total_image_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    const int block_size = THREADS_PER_BLOCK;
    const int grid_size = batch_size;
    const int shared_mem_size = NUM_BINS * 4 * sizeof(int); // 4 histograms
    
    computeHistogramsKernel<<<grid_size, block_size, shared_mem_size>>>(
        d_images,
        width,
        height,
        channels,
        d_features
    );
    CUDA_CHECK_KERNEL();
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_features, d_features, batch_size * sizeof(Feature), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_features));
} 