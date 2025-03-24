#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define NUM_BINS 16
#define K_NEIGHBORS 3

typedef struct {
    float bins[NUM_BINS];
    int label; // 1 = screenshot, 0 = non-screenshot
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
    int result = (votes >= (K_NEIGHBORS / 2 + 1)) ? 1 : 0;
    
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
    Feature *model = malloc((*size) * sizeof(Feature));
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

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model_file> <image_path>\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    const char *image_path = argv[2];
    
    // Load model
    int model_size;
    Feature *model = load_model(model_path, &model_size);
    if (!model) {
        return 1;
    }
    
    printf("Model loaded with %d training examples\n", model_size);
    
    // Extract features from query image
    Feature query;
    if (extract_feature(image_path, &query) != 0) {
        fprintf(stderr, "Failed to extract features from image: %s\n", image_path);
        free(model);
        return 1;
    }
    
    // Classify using KNN
    int result = classify_knn(model, model_size, &query);
    printf("Classification result for %s: %s\n", image_path, result ? "SCREENSHOT" : "NON-SCREENSHOT");
    
    // Clean up
    free(model);
    
    return 0;
} 