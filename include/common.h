#ifndef COMMON_H
#define COMMON_H

#define NUM_BINS 16
#define K_NEIGHBORS 3
#define MAX_BATCH_SIZE 32

// Feature structure (same as CPU version)
typedef struct {
    float bins[NUM_BINS];
    int label; // 1 = screenshot, 0 = non-screenshot
} Feature;

// Constants for CUDA implementation
#define THREADS_PER_BLOCK 256  // Will be tuned later
#define SHARED_MEMORY_SIZE (NUM_BINS * sizeof(int))

#endif // COMMON_H 