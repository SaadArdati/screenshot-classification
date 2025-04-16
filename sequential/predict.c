#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>  // For timing measurements
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/common.h"

// Simplified computation of statistical metrics for screenshot detection
ScreenshotStats compute_screenshot_statistics(unsigned char *img, int w, int h, int channels) {
    ScreenshotStats stats = {0};
    
    // Single-pass analysis to extract all metrics
    int total_pixels = w * h;
    int edge_pixels = 0;
    int regular_edge_pixels = 0;
    int uniform_color_pixels = 0;
    int grid_aligned_pixels = 0;
    
    // Allocate memory for horizontal line analysis
    int *horizontal_edge_counts = (int *)calloc(h, sizeof(int));
    if (!horizontal_edge_counts) {
        fprintf(stderr, "Memory allocation failed\n");
        return stats;
    }
    
    // Single-pass image analysis
    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            const int idx = (y * w + x) * channels;
            
            // Get grayscale of current and neighboring pixels
            const unsigned char gray = (img[idx] + img[idx+1] + img[idx+2]) / 3;
            const unsigned char gray_left = (img[(y * w + (x-1)) * channels] + 
                                           img[(y * w + (x-1)) * channels + 1] + 
                                           img[(y * w + (x-1)) * channels + 2]) / 3;
            const unsigned char gray_right = (img[(y * w + (x+1)) * channels] + 
                                            img[(y * w + (x+1)) * channels + 1] + 
                                            img[(y * w + (x+1)) * channels + 2]) / 3;
            const unsigned char gray_up = (img[((y-1) * w + x) * channels] + 
                                         img[((y-1) * w + x) * channels + 1] + 
                                         img[((y-1) * w + x) * channels + 2]) / 3;
            const unsigned char gray_down = (img[((y+1) * w + x) * channels] + 
                                           img[((y+1) * w + x) * channels + 1] + 
                                           img[((y+1) * w + x) * channels + 2]) / 3;
            
            // Calculate horizontal and vertical gradients
            const int h_gradient = abs(gray_right - gray_left);
            const int v_gradient = abs(gray_down - gray_up);
            
            // Detect edges
            if (h_gradient > EDGE_THRESHOLD || v_gradient > EDGE_THRESHOLD) {
                edge_pixels++;
                horizontal_edge_counts[y]++;
                
                // Check for regular edges (straight lines common in UI)
                if ((h_gradient > EDGE_THRESHOLD && v_gradient < EDGE_THRESHOLD/2) || 
                    (v_gradient > EDGE_THRESHOLD && h_gradient < EDGE_THRESHOLD/2)) {
                    regular_edge_pixels++;
                }
            }
            
            // Check for uniform color regions (common in UI backgrounds/panels)
            int local_variance = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (y+dy >= 0 && y+dy < h && x+dx >= 0 && x+dx < w) {
                        const int local_idx = ((y+dy) * w + (x+dx)) * channels;
                        const unsigned char local_gray = (img[local_idx] + img[local_idx+1] + img[local_idx+2]) / 3;
                        local_variance += abs(gray - local_gray);
                    }
                }
            }
            
            // Low local variance indicates uniform color region
            if (local_variance < 20) {
                uniform_color_pixels++;
            }
        }
    }
    
    // Detect grid alignment (common in UI layouts) using horizontal edge analysis
    int aligned_rows = 0;
    for (int y = 1; y < h-3; y++) {
        // Check for similar edge patterns in consecutive rows (indicates UI grid)
        if (horizontal_edge_counts[y] > 0 && 
            abs(horizontal_edge_counts[y] - horizontal_edge_counts[y+1]) < w * 0.05) {
            aligned_rows++;
        }
    }
    
    // Calculate final statistics (normalized to [0,1] range)
    float edge_density = (float)edge_pixels / total_pixels;
    float edge_regularity = edge_pixels > 0 ? (float)regular_edge_pixels / edge_pixels : 0;
    float grid_alignment = (float)aligned_rows / h;
    float color_uniformity = (float)uniform_color_pixels / total_pixels;
    
    // Combine metrics into simplified scores
    stats.edge_score = (edge_regularity * 0.6) + (edge_density * 0.2) + (grid_alignment * 0.2);
    stats.color_score = color_uniformity;
    stats.ui_element_score = edge_density * 0.5 + grid_alignment * 0.5;
    
    free(horizontal_edge_counts);
    return stats;
}

// Simplified screenshot detection
int is_likely_screenshot(ScreenshotStats stats) {
    // Calculate weighted score
    float score = 0;
    score += stats.edge_score * 0.4;       // Edge characteristics
    score += stats.color_score * 0.3;      // Color uniformity
    score += stats.ui_element_score * 0.3; // UI element indicators
    
    // Output analysis
    printf("Screenshot Analysis:\n");
    printf("Edge Score: %.3f\n", stats.edge_score);
    printf("Color Score: %.3f\n", stats.color_score);
    printf("UI Element Score: %.3f\n", stats.ui_element_score);
    printf("Final Score: %.3f (Threshold: %.1f)\n", score, SCREENSHOT_SCORE_THRESHOLD);
    
    return score > SCREENSHOT_SCORE_THRESHOLD;
}

// Extract features from image
int extract_feature(const char *path, Feature *f) {
    int w, h, c;
    unsigned char *img = stbi_load(path, &w, &h, &c, 3);
    if (!img) return -1;
    
    // First perform statistical analysis
    ScreenshotStats stats = compute_screenshot_statistics(img, w, h, 3);
    int is_screenshot = is_likely_screenshot(stats);
    
    if (is_screenshot) {
        printf("Screenshot detected by statistical analysis\n");
    }

    // Extract histogram features for kNN (standard approach)
    const int total = w * h;
    int counts[NUM_BINS] = {0};
    int edge_counts[NUM_BINS] = {0};
    int top_counts[NUM_BINS] = {0};
    int bottom_counts[NUM_BINS] = {0};
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int idx = (y * w + x) * 3;
            // Calculate grayscale
            const unsigned char gray = (img[idx] + img[idx + 1] + img[idx + 2]) / 3;
            const int bin = gray * NUM_BINS / 256;
            
            // Add to overall histogram
            counts[bin]++;
            
            // Process top 5% region
            if (y < h * 0.05) {
                top_counts[bin]++;
            }
            
            // Process bottom 10% region
            if (y > h * 0.9) {
                bottom_counts[bin]++;
            }
            
            // Edge detection (simple horizontal gradient)
            if (x > 0) {
                const int prev_idx = idx - 3;
                const unsigned char prev_gray = (img[prev_idx] + img[prev_idx + 1] + img[prev_idx + 2]) / 3;
                int diff = abs(gray - prev_gray);
                
                // Vertical edge detection
                if (y > 0) {
                    const int above_idx = ((y-1) * w + x) * 3;
                    const unsigned char above_gray = (img[above_idx] + img[above_idx + 1] + img[above_idx + 2]) / 3;
                    int v_diff = abs(gray - above_gray);
                    diff = (diff > v_diff) ? diff : v_diff;  // Take maximum gradient
                }
                
                // If edge detected
                if (diff > EDGE_THRESHOLD) {
                    edge_counts[bin]++;
                }
            }
        }
    }
    
    // Normalize histograms
    for (int i = 0; i < NUM_BINS; i++) {
        f->bins[i] = (float) counts[i] / (float) total;
        f->edge_bins[i] = (float) edge_counts[i] / (float) total;
        f->top_region_bins[i] = (float) top_counts[i] / (float) (total * 0.05);
        f->bottom_region_bins[i] = (float) bottom_counts[i] / (float) (total * 0.1);
    }
    
    // Direct classification override if statistical analysis is confident
    if (is_screenshot) {
        stbi_image_free(img);
        return 2;  // Special code to indicate screenshot
    }
    
    stbi_image_free(img);
    return 0;
}

float euclidean(const Feature *a, const Feature *b) {
    float sum = 0;
    
    // Weight for each feature type
    const float hist_weight = 0.3;
    const float edge_weight = 0.3;
    const float top_weight = 0.2;
    const float bottom_weight = 0.2;
    
    // Regular histogram distance
    for (int i = 0; i < NUM_BINS; i++) {
        const float diff = a->bins[i] - b->bins[i];
        sum += hist_weight * diff * diff;
    }
    
    // Edge histogram distance
    for (int i = 0; i < NUM_BINS; i++) {
        const float diff = a->edge_bins[i] - b->edge_bins[i];
        sum += edge_weight * diff * diff;
    }
    
    // Top region histogram distance
    for (int i = 0; i < NUM_BINS; i++) {
        const float diff = a->top_region_bins[i] - b->top_region_bins[i];
        sum += top_weight * diff * diff;
    }
    
    // Bottom region histogram distance
    for (int i = 0; i < NUM_BINS; i++) {
        const float diff = a->bottom_region_bins[i] - b->bottom_region_bins[i];
        sum += bottom_weight * diff * diff;
    }
    
    return sqrtf(sum);
}

int classify_knn(const Feature *train, const int n, const Feature *q) {
    typedef struct {
        float dist;
        int label;
    } Neighbor;

    Neighbor *neighbors = malloc(n * sizeof(Neighbor));
    if (!neighbors) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < n; i++) {
        neighbors[i].dist = euclidean(&train[i], q);
        neighbors[i].label = train[i].label;
    }

    // Partial sort: find top K_NEIGHBORS
    for (int i = 0; i < K_NEIGHBORS; i++) {
        for (int j = i + 1; j < n; j++) {
            if (neighbors[j].dist < neighbors[i].dist) {
                const Neighbor tmp = neighbors[i];
                neighbors[i] = neighbors[j];
                neighbors[j] = tmp;
            }
        }
    }

    int votes = 0;
    for (int i = 0; i < K_NEIGHBORS; i++) votes += neighbors[i].label;
    const int result = votes >= K_NEIGHBORS / 2 + 1 ? 1 : 0;

    free(neighbors);
    return result;
}

// Load model from a file
Feature *load_model(const char *filename, int *size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model file: %s\n", filename);
        return NULL;
    }

    // Read dataset size
    if (fread(size, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read model size\n");
        fclose(f);
        return NULL;
    }

    // Allocate memory for features
    Feature *model = malloc(*size * sizeof(Feature));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(f);
        return NULL;
    }

    // Read features
    if (fread(model, sizeof(Feature), *size, f) != *size) {
        fprintf(stderr, "Failed to read model data\n");
        free(model);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return model;
}

int main(const int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model_file> <image_path>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    // Performance measurement variables
    clock_t start_time, end_time;
    double load_model_time = 0.0, feature_extraction_time = 0.0, classification_time = 0.0;

    // Load model
    int model_size;
    start_time = clock();
    Feature *model = load_model(model_path, &model_size);
    end_time = clock();
    load_model_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (!model) {
        return 1;
    }

    printf("Model loaded with %d training examples\n", model_size);

    // Extract features from query image
    Feature query;
    start_time = clock();
    int feature_result = extract_feature(image_path, &query);
    end_time = clock();
    feature_extraction_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    if (feature_result == 2) {
        // Screenshot detected by statistical analysis
        printf("Classification result for %s: SCREENSHOT (Statistical analysis)\n", image_path);
    } else if (feature_result != 0) {
        fprintf(stderr, "Failed to extract features from image: %s\n", image_path);
        free(model);
        return 1;
    } else {
        // Classify using KNN
        start_time = clock();
        const int result = classify_knn(model, model_size, &query);
        end_time = clock();
        classification_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Classification result for %s: %s\n", image_path, result ? "SCREENSHOT" : "NON-SCREENSHOT");
    }

    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Model Loading Time: %.5f seconds\n", load_model_time);
    printf("Feature Extraction Time: %.5f seconds\n", feature_extraction_time);
    printf("Classification Time: %.5f seconds\n", classification_time);
    printf("Total Processing Time: %.5f seconds\n", load_model_time + feature_extraction_time + classification_time);
    printf("Memory Usage (Model): %.2f MB\n", (float)(model_size * sizeof(Feature)) / (1024 * 1024));

    // Clean up
    free(model);

    return 0;
}
