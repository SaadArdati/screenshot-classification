#ifndef COMMON_H
#define COMMON_H

#define NUM_BINS 16
#define K_NEIGHBORS 3
#define MAX_BATCH_SIZE 32
#define MAX_IMAGES 50000  // Maximum number of images to process
#define EDGE_THRESHOLD 30  // Threshold for edge detection

// Screenshot score threshold
#define SCREENSHOT_SCORE_THRESHOLD 0.5

// Feature structure (enhanced version)
typedef struct {
    float bins[NUM_BINS];           // Grayscale histogram
    float edge_bins[NUM_BINS];      // Edge histogram  
    float top_region_bins[NUM_BINS];  // Top region features
    float bottom_region_bins[NUM_BINS]; // Bottom region features
    int label; // 1 = screenshot, 0 = non-screenshot, 2 = statistically detected
} Feature;

// Statistical metrics for screenshot detection
typedef struct {
    float edge_score;      // Combined edge metrics (regularity, grid alignment)
    float color_score;     // Color uniformity measure
    float ui_element_score; // UI element probability (corners, text)
} ScreenshotStats;

// Constants for CUDA implementation
#ifdef __CUDACC__
#define THREADS_PER_BLOCK 256  // Will be tuned later
#define SHARED_MEMORY_SIZE (NUM_BINS * sizeof(int) * 4)  // Increased for multiple histograms
#endif

#endif // COMMON_H
