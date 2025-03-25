#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "common.h"

// Atomic add operation for float values (for histogram computation)
__device__ void atomicAdd_float(float* address, float val) {
    atomicAdd(address, val);
}

// CUDA kernel for converting RGB to grayscale and computing histogram
__global__ void computeHistogramKernel(
    const unsigned char* d_input,  // Input image data
    int width,
    int height,
    int channels,
    float* d_histograms,          // Output histograms (one per image in batch)
    int batch_offset              // Offset for current image in batch
) {
    extern __shared__ int shared_hist[];  // Shared memory for histogram
    
    // Initialize shared memory
    const int tid = threadIdx.x;
    if (tid < NUM_BINS) {
        shared_hist[tid] = 0;
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
            const int base_idx = i * channels;
            
            // Convert RGB to grayscale
            const unsigned char gray = (
                d_input[base_idx] +      // R
                d_input[base_idx + 1] +  // G
                d_input[base_idx + 2]    // B
            ) / 3;

            // Calculate histogram bin
            const int bin = gray * NUM_BINS / 256;
            
            // Update histogram in shared memory
            atomicAdd(&shared_hist[bin], 1);
        }
    }
    
    __syncthreads();

    // Write results to global memory
    if (tid < NUM_BINS) {
        const float normalized_value = (float)shared_hist[tid] / (float)total_pixels;
        d_histograms[batch_offset * NUM_BINS + tid] = normalized_value;
    }
}

// Host function to extract features from a batch of images
extern "C" void extractFeaturesGPU(
    const unsigned char* h_images,    // Host array of image data
    int batch_size,                   // Number of images in batch
    int width,                        // Image width
    int height,                       // Image height
    int channels,                     // Number of channels (3 for RGB)
    Feature* h_features              // Host array for output features
) {
    const int image_size = width * height * channels;
    const int total_size = image_size * batch_size;

    // Allocate device memory
    unsigned char* d_images = allocateDeviceMemory<unsigned char>(total_size);
    float* d_histograms = allocateDeviceMemory<float>(batch_size * NUM_BINS);

    // Copy image data to device
    copyToDevice(d_images, h_images, total_size);

    // Calculate grid and block dimensions
    const int block_size = THREADS_PER_BLOCK;
    const int num_blocks = (width * height + block_size - 1) / block_size;

    // Process each image in the batch
    for (int i = 0; i < batch_size; i++) {
        // Compute histogram for current image
        computeHistogramKernel<<<num_blocks, block_size, NUM_BINS * sizeof(int)>>>(
            d_images + (i * image_size),
            width,
            height,
            channels,
            d_histograms,
            i
        );
        CUDA_CHECK_KERNEL();
    }

    // Copy results back to host
    for (int i = 0; i < batch_size; i++) {
        copyToHost(h_features[i].bins, d_histograms + (i * NUM_BINS), NUM_BINS);
    }

    // Free device memory
    freeDeviceMemory(d_images);
    freeDeviceMemory(d_histograms);
} 