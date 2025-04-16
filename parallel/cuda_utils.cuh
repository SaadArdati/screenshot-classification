#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel launch error checking macro
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper function to get optimal block size
inline int getOptimalBlockSize() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.maxThreadsPerBlock;
}

// Helper function to calculate grid size based on work size and block size
inline dim3 getOptimalGridSize(int work_size, int block_size) {
    return dim3((work_size + block_size - 1) / block_size);
}

// Memory management helpers
template<typename T>
T* allocateDeviceMemory(size_t count) {
    T* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
    return d_ptr;
}

template<typename T>
void freeDeviceMemory(T* d_ptr) {
    if (d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

template<typename T>
void copyToDevice(T* d_dst, const T* h_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copyToHost(T* h_dst, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

#endif // CUDA_UTILS_CUH 