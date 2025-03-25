#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "common.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Function declarations
extern "C" void extractFeaturesGPU(const unsigned char* h_images, int batch_size,
                                 int width, int height, int channels,
                                 Feature* h_features);
extern "C" void classifyBatchGPU(const Feature* train_features, int train_size,
                               const Feature* query_features, int query_size,
                               int* predictions, double* computation_times);

// Performance monitoring structure
typedef struct {
    double data_loading_time;
    double feature_extraction_time;
    double knn_transfer_time;
    double knn_compute_time;
    double total_time;
    size_t peak_memory_usage;
    int total_images;
    float accuracy;
} PerformanceMetrics;

// Print device information with power limits
void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("\nCUDA Device Information:\n");
    printf("------------------------\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Grid Dimensions: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total Global Memory: %.2f GB\n", 
           (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Max Threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate * 1e-6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("\n");
}

// Get current GPU memory usage
size_t getCurrentGPUMemory() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem - free_mem;
}

// Print performance metrics
void printPerformanceMetrics(const PerformanceMetrics* metrics) {
    printf("\nDetailed Performance Metrics:\n");
    printf("---------------------------\n");
    printf("Data Loading Time: %.2f seconds\n", metrics->data_loading_time);
    printf("Feature Extraction Time: %.2f seconds\n", metrics->feature_extraction_time);
    printf("KNN Data Transfer Time: %.2f seconds\n", metrics->knn_transfer_time);
    printf("KNN Computation Time: %.2f seconds\n", metrics->knn_compute_time);
    printf("Total Processing Time: %.2f seconds\n", metrics->total_time);
    printf("Peak GPU Memory Usage: %.2f MB\n", metrics->peak_memory_usage / (1024.0f * 1024.0f));
    printf("Total Images Processed: %d\n", metrics->total_images);
    printf("Processing Speed: %.2f images/second\n", 
           metrics->total_images / metrics->total_time);
    printf("Classification Accuracy: %.2f%%\n", metrics->accuracy * 100);
    printf("\nPer-Phase Performance:\n");
    printf("Data Loading: %.1f%%\n", 
           (metrics->data_loading_time / metrics->total_time) * 100);
    printf("Feature Extraction: %.1f%%\n",
           (metrics->feature_extraction_time / metrics->total_time) * 100);
    printf("KNN Classification: %.1f%%\n",
           ((metrics->knn_transfer_time + metrics->knn_compute_time) / metrics->total_time) * 100);
}

// Load images in batches
int loadImageBatch(const char* dirpath, int label, unsigned char* batch_buffer,
                  Feature* features, int batch_size, int* current_index,
                  DIR* dir) {
    struct dirent* entry;
    int loaded = 0;
    int width, height, channels;

    while (loaded < batch_size && (entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) continue;

        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);

        // Load image
        unsigned char* img = stbi_load(fullpath, &width, &height, &channels, 3);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", fullpath);
            continue;
        }

        // Copy to batch buffer
        const int image_size = width * height * 3;
        memcpy(batch_buffer + (loaded * image_size), img, image_size);
        features[*current_index + loaded].label = label;

        stbi_image_free(img);
        loaded++;
    }

    return loaded;
}

// Save model to file
int saveModel(const char* filename, Feature* features, int total_features) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return -1;
    }

    // Write number of features
    fwrite(&total_features, sizeof(int), 1, f);
    
    // Write features
    fwrite(features, sizeof(Feature), total_features, f);
    
    fclose(f);
    return 0;
}

// Update the main function to include KNN classification and performance monitoring
int main(int argc, char** argv) {
    PerformanceMetrics metrics = {0};
    size_t peak_memory = 0;
    clock_t total_start = clock();

    // Initialize CUDA and print device info
    CUDA_CHECK(cudaSetDevice(0));
    printDeviceInfo();

    // Path to the split_data directory
    const char* screenshots_train_dir = "split_data/screenshots_256x256/train";
    const char* non_screenshots_train_dir = "split_data/non_screenshot_256x256/train";
    const char* model_path = (argc > 1) ? argv[1] : "trained_model.bin";

    // Performance measurement variables
    clock_t start_time, end_time;
    double total_time = 0.0, feature_extraction_time = 0.0;

    start_time = clock();

    // Allocate host memory for batch processing
    const int image_size = 256 * 256 * 3;  // Assuming 256x256 RGB images
    unsigned char* batch_buffer = (unsigned char*)malloc(MAX_BATCH_SIZE * image_size);
    Feature* all_features = (Feature*)malloc(100000 * sizeof(Feature)); // Adjust size as needed
    
    if (!batch_buffer || !all_features) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    int total_processed = 0;

    // Process screenshots
    printf("Processing screenshots...\n");
    DIR* dir = opendir(screenshots_train_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", screenshots_train_dir);
        return 1;
    }

    while (1) {
        int loaded = loadImageBatch(screenshots_train_dir, 1, batch_buffer,
                                  all_features, MAX_BATCH_SIZE, &total_processed, dir);
        if (loaded == 0) break;

        // Process batch on GPU
        clock_t batch_start = clock();
        extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, all_features + total_processed);
        feature_extraction_time += (double)(clock() - batch_start) / CLOCKS_PER_SEC;
        
        total_processed += loaded;
        printf("\rProcessed %d screenshots", total_processed);
        fflush(stdout);
    }
    printf("\n");
    closedir(dir);

    // Process non-screenshots
    printf("Processing non-screenshots...\n");
    dir = opendir(non_screenshots_train_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", non_screenshots_train_dir);
        return 1;
    }

    while (1) {
        int loaded = loadImageBatch(non_screenshots_train_dir, 0, batch_buffer,
                                  all_features, MAX_BATCH_SIZE, &total_processed, dir);
        if (loaded == 0) break;

        // Process batch on GPU
        clock_t batch_start = clock();
        extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, all_features + total_processed);
        feature_extraction_time += (double)(clock() - batch_start) / CLOCKS_PER_SEC;

        total_processed += loaded;
        printf("\rProcessed %d non-screenshots", total_processed);
        fflush(stdout);
    }
    printf("\n");
    closedir(dir);

    // Save model
    printf("Saving model to %s...\n", model_path);
    if (saveModel(model_path, all_features, total_processed) != 0) {
        fprintf(stderr, "Failed to save model\n");
    }

    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Total Processing Time: %.2f seconds\n", total_time);
    printf("Feature Extraction Time: %.2f seconds\n", feature_extraction_time);
    printf("Average Time per Image: %.4f seconds\n", feature_extraction_time / total_processed);
    printf("Total Images Processed: %d\n", total_processed);
    printf("Throughput: %.2f images/second\n", total_processed / total_time);

    // Record peak memory after feature extraction
    peak_memory = max(peak_memory, getCurrentGPUMemory());

    // Perform KNN classification on test set
    printf("\nPerforming KNN classification...\n");
    double knn_times[2];  // [0] = transfer time, [1] = compute time
    int* predictions = (int*)malloc(test_size * sizeof(int));
    
    clock_t knn_start = clock();
    classifyBatchGPU(train_features, train_size, test_features, test_size,
                     predictions, knn_times);
    clock_t knn_end = clock();

    // Record peak memory after KNN
    peak_memory = max(peak_memory, getCurrentGPUMemory());

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        if (predictions[i] == test_features[i].label) {
            correct++;
        }
    }
    metrics.accuracy = (float)correct / test_size;

    // Update performance metrics
    metrics.data_loading_time = data_loading_time;
    metrics.feature_extraction_time = feature_extraction_time;
    metrics.knn_transfer_time = knn_times[0];
    metrics.knn_compute_time = knn_times[1];
    metrics.total_time = (double)(clock() - total_start) / CLOCKS_PER_SEC;
    metrics.peak_memory_usage = peak_memory;
    metrics.total_images = total_processed;

    // Print detailed performance metrics
    printPerformanceMetrics(&metrics);

    // Clean up
    free(batch_buffer);
    free(predictions);
    free(all_features);

    return 0;
} 