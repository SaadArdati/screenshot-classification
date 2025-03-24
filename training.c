#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <string.h>
#include <time.h>  // For timing measurements
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

// Load all images from `dir`, set label, append to array
int load_folder(const char *dirpath, const int label, Feature **arr, int *size) {
    DIR *d = opendir(dirpath);
    struct dirent *entry;
    if (!d) {
        fprintf(stderr, "Error opening directory: %s\n", dirpath);
        return -1;
    }

    while ((entry = readdir(d)) != NULL) {
        if (entry->d_type != DT_REG) continue;
        
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
        
        *arr = realloc(*arr, sizeof(Feature) * (*size + 1));
        if (!*arr) {
            fprintf(stderr, "Memory allocation failed\n");
            closedir(d);
            return -1;
        }
        
        Feature *f = &(*arr)[*size];
        f->label = label;
        
        if (extract_feature(fullpath, f) == 0) {
            (*size)++;
            if (*size % 100 == 0) {
                printf("Processed %d images\n", *size);
            }
        } else {
            fprintf(stderr, "Failed to extract features from: %s\n", fullpath);
        }
    }
    closedir(d);
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

// Calculate accuracy on test set
float evaluate_model(Feature *train_set, int train_size, Feature *test_set, int test_size) {
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        int prediction = classify_knn(train_set, train_size, &test_set[i]);
        if (prediction == test_set[i].label) {
            correct++;
        }
    }
    return (float)correct / test_size;
}

// Save the model (features and labels) to a file
int save_model(const char *filename, Feature *dataset, int size) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return -1;
    }
    
    // Write dataset size
    fwrite(&size, sizeof(int), 1, f);
    
    // Write features and labels
    for (int i = 0; i < size; i++) {
        fwrite(&dataset[i], sizeof(Feature), 1, f);
    }
    
    fclose(f);
    return 0;
}

// Function to measure memory usage - approximate, system-dependent
size_t get_memory_usage(Feature *train_data, int train_count, Feature *test_data, int test_count) {
    size_t memory = 0;
    
    // Memory for training data
    memory += train_count * sizeof(Feature);
    
    // Memory for test data
    memory += test_count * sizeof(Feature);
    
    // Additional memory for arrays, etc.
    memory += 1024 * 1024; // ~1MB for other variables
    
    return memory;
}

int main(int argc, char **argv) {
    // Path to the split_data directory
    char screenshots_train_dir[512] = "split_data/screenshots_256x256/train";
    char non_screenshots_train_dir[512] = "split_data/non_screenshot_256x256/train";
    char screenshots_test_dir[512] = "split_data/screenshots_256x256/test";
    char non_screenshots_test_dir[512] = "split_data/non_screenshot_256x256/test";
    char model_path[512] = "trained_model.bin";
    
    // Performance measurement variables
    clock_t start_time, end_time;
    double load_time = 0.0, training_time = 0.0, evaluation_time = 0.0, save_time = 0.0;
    
    // Parse command line arguments (if provided)
    if (argc > 1) {
        strcpy(model_path, argv[1]);
    }
    
    // Load training data
    Feature *train_data = NULL;
    int train_count = 0;
    
    printf("Loading screenshot training images...\n");
    start_time = clock();
    if (load_folder(screenshots_train_dir, 1, &train_data, &train_count) != 0) {
        fprintf(stderr, "Failed to load screenshot training images\n");
        free(train_data);
        return 1;
    }
    
    printf("Loading non-screenshot training images...\n");
    if (load_folder(non_screenshots_train_dir, 0, &train_data, &train_count) != 0) {
        fprintf(stderr, "Failed to load non-screenshot training images\n");
        free(train_data);
        return 1;
    }
    end_time = clock();
    load_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Training data loaded: %d images\n", train_count);
    
    // Load test data
    Feature *test_data = NULL;
    int test_count = 0;
    
    printf("Loading screenshot test images...\n");
    start_time = clock();
    if (load_folder(screenshots_test_dir, 1, &test_data, &test_count) != 0) {
        fprintf(stderr, "Failed to load screenshot test images\n");
        free(train_data);
        free(test_data);
        return 1;
    }
    
    printf("Loading non-screenshot test images...\n");
    if (load_folder(non_screenshots_test_dir, 0, &test_data, &test_count) != 0) {
        fprintf(stderr, "Failed to load non-screenshot test images\n");
        free(train_data);
        free(test_data);
        return 1;
    }
    end_time = clock();
    load_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Test data loaded: %d images\n", test_count);
    
    // Evaluate model
    printf("Evaluating model...\n");
    start_time = clock();
    float accuracy = evaluate_model(train_data, train_count, test_data, test_count);
    end_time = clock();
    evaluation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Model accuracy on test set: %.2f%%\n", accuracy * 100);
    
    // Save the trained model
    printf("Saving model to %s...\n", model_path);
    start_time = clock();
    if (save_model(model_path, train_data, train_count) != 0) {
        fprintf(stderr, "Failed to save model\n");
    } else {
        printf("Model saved successfully\n");
    }
    end_time = clock();
    save_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Calculate memory usage
    size_t memory_usage = get_memory_usage(train_data, train_count, test_data, test_count);
    
    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Data Loading Time: %.2f seconds\n", load_time);
    printf("Model Evaluation Time: %.2f seconds\n", evaluation_time);
    printf("Model Saving Time: %.2f seconds\n", save_time);
    printf("Total Processing Time: %.2f seconds\n", load_time + evaluation_time + save_time);
    printf("Approximate Memory Usage: %.2f MB\n", (float)memory_usage / (1024 * 1024));
    printf("Number of Training Examples: %d\n", train_count);
    printf("Number of Test Examples: %d\n", test_count);
    printf("Classification Accuracy: %.2f%%\n", accuracy * 100);
    
    // Clean up
    free(train_data);
    free(test_data);
    
    return 0;
}
