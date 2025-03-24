#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <time.h>  // For timing measurements
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define NUM_BINS 16
#define K_NEIGHBORS 3

typedef struct {
    float bins[NUM_BINS];

     // 1 = screenshot, 0 = non-screenshot
     int label;
} Feature;

// Compute normalized grayscale histogram
int extract_feature(const char *path, Feature *f) {
    int w, h, c;
    unsigned char *img = stbi_load(path, &w, &h, &c, 3);
    if (!img) return -1;

    const int total = w * h;
    int counts[NUM_BINS] = {0};
    for (int i = 0; i < total; i++) {
        const int idx = i * 3;
        const unsigned char gray = (img[idx] + img[idx + 1] + img[idx + 2]) / 3;
        counts[gray * NUM_BINS / 256]++;
    }
    stbi_image_free(img);

    for (int i = 0; i < NUM_BINS; i++) {
        f->bins[i] = (float) counts[i] / (float) total;
    }
    return 0;
}

float euclidean(const Feature *a, const Feature *b) {
    float sum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        const float diff = a->bins[i] - b->bins[i];
        sum += diff * diff;
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
    if (extract_feature(image_path, &query) != 0) {
        fprintf(stderr, "Failed to extract features from image: %s\n", image_path);
        free(model);
        return 1;
    }
    end_time = clock();
    feature_extraction_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Classify using KNN
    start_time = clock();
    const int result = classify_knn(model, model_size, &query);
    end_time = clock();
    classification_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Classification result for %s: %s\n", image_path, result ? "SCREENSHOT" : "NON-SCREENSHOT");

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
