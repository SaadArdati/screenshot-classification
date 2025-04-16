#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>
#include "../include/cuda_utils.cuh"
#include "../include/common.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

// Platform-specific directory handling
#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(dir) _mkdir(dir)
#define PATH_SEPARATOR "\\"
#else
#include <sys/stat.h>
#define CREATE_DIR(dir) mkdir(dir, 0777)
#define PATH_SEPARATOR "/"
#endif

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
    int statistical_detections;
} PerformanceMetrics;

// Print device information
void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("\nCUDA Device Information:\n");
    printf("------------------------\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Total Global Memory: %.2f GB\n", 
           (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("\n");
}

// Get current GPU memory usage
size_t getCurrentGPUMemory() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem - free_mem;
}

// Save model to file
int saveModel(const char* path, const Feature* features, int count) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    
    fwrite(&count, sizeof(int), 1, f);
    fwrite(features, sizeof(Feature), count, f);
    fclose(f);
    return 0;
}

// Print performance metrics
void printPerformanceMetrics(const PerformanceMetrics* metrics) {
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Data Loading Time: %.2f seconds\n", metrics->data_loading_time);
    printf("Total Processing Time: %.2f seconds\n", metrics->total_time);
    printf("Peak GPU Memory Usage: %.2f MB\n", metrics->peak_memory_usage / (1024.0f * 1024.0f));
    printf("Total Images Processed: %d\n", metrics->total_images);
    printf("Classification Accuracy: %.2f%%\n", metrics->accuracy * 100);
}

// Simple feature extraction (sequential fallback)
void extractFeaturesCPU(const char* imagePath, Feature* feature) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", imagePath);
        return;
    }
    
    // Initialize feature
    memset(feature, 0, sizeof(Feature));
    
    // Simple grayscale histogram
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            unsigned char gray = (img[idx] + img[idx+1] + img[idx+2]) / 3;
            int bin = gray * NUM_BINS / 256;
            feature->bins[bin] += 1.0f / (width * height);
        }
    }
    
    stbi_image_free(img);
}

// Load images from directory
int loadImagesFromDir(const char* dirPath, int label, Feature* features, int* count, int maxImages) {
    DIR* dir = opendir(dirPath);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", dirPath);
        return -1;
    }
    
    struct dirent* entry;
    int loaded = 0;
    
    while ((entry = readdir(dir)) != NULL && *count < maxImages) {
        if (entry->d_type != DT_REG) continue;
        
        char fullPath[512];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", dirPath, entry->d_name);
        
        // Extract features
        extractFeaturesCPU(fullPath, &features[*count]);
        features[*count].label = label;
        (*count)++;
        loaded++;
    }
    
    closedir(dir);
    printf("Loaded %d images from %s\n", loaded, dirPath);
    return loaded;
}

// Simple KNN classification (sequential fallback)
float evaluateModel(Feature* trainSet, int trainSize, Feature* testSet, int testSize) {
    if (trainSize == 0 || testSize == 0) return 0.0f;
    
    int correct = 0;
    
    for (int i = 0; i < testSize; i++) {
        // Find nearest neighbor
        float minDist = INFINITY;
        int prediction = 0;
        
        for (int j = 0; j < trainSize; j++) {
            float dist = 0.0f;
            for (int k = 0; k < NUM_BINS; k++) {
                float diff = testSet[i].bins[k] - trainSet[j].bins[k];
                dist += diff * diff;
            }
            
            if (dist < minDist) {
                minDist = dist;
                prediction = trainSet[j].label;
            }
        }
        
        if (prediction == testSet[i].label) {
            correct++;
        }
    }
    
    return (float)correct / testSize;
}

int main(int argc, char** argv) {
    // Check CUDA device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA devices found\n");
        return 1;
    }
    
    // Select device and print info
    CUDA_CHECK(cudaSetDevice(0));
    printDeviceInfo();
    
    // Check command line args
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_output_file>\n", argv[0]);
        return 1;
    }
    
    const char* modelPath = argv[1];
    
    // Performance metrics
    PerformanceMetrics metrics = {0};
    clock_t start_time = clock();
    
    // Get current directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        fprintf(stderr, "Failed to get current working directory\n");
        return 1;
    }
    
    // Setup directory paths
    char screenshots_train_dir[1024], non_screenshots_train_dir[1024];
    char screenshots_test_dir[1024], non_screenshots_test_dir[1024];
    
    snprintf(screenshots_train_dir, sizeof(screenshots_train_dir), 
             "%s/split_data/screenshots_256x256/train", cwd);
    snprintf(non_screenshots_train_dir, sizeof(non_screenshots_train_dir), 
             "%s/split_data/non_screenshot_256x256/train", cwd);
    snprintf(screenshots_test_dir, sizeof(screenshots_test_dir), 
             "%s/split_data/screenshots_256x256/test", cwd);
    snprintf(non_screenshots_test_dir, sizeof(non_screenshots_test_dir), 
             "%s/split_data/non_screenshot_256x256/test", cwd);
    
    printf("Screenshots train: %s\n", screenshots_train_dir);
    printf("Non-screenshots train: %s\n", non_screenshots_train_dir);
    printf("Screenshots test: %s\n", screenshots_test_dir);
    printf("Non-screenshots test: %s\n", non_screenshots_test_dir);
    printf("Model path: %s\n\n", modelPath);
    
    // Allocate memory for features
    const int MAX_IMG_COUNT = 50000;
    Feature* trainFeatures = (Feature*)malloc(MAX_IMG_COUNT * sizeof(Feature));
    Feature* testFeatures = (Feature*)malloc(MAX_IMG_COUNT * sizeof(Feature));
    
    if (!trainFeatures || !testFeatures) {
        fprintf(stderr, "Failed to allocate memory for features\n");
        free(trainFeatures);
        free(testFeatures);
        return 1;
    }
    
    // Load training data
    int trainCount = 0;
    clock_t data_start = clock();
    
    printf("Loading screenshot training images...\n");
    loadImagesFromDir(screenshots_train_dir, 1, trainFeatures, &trainCount, MAX_IMG_COUNT);
    
    printf("Loading non-screenshot training images...\n");
    loadImagesFromDir(non_screenshots_train_dir, 0, trainFeatures, &trainCount, MAX_IMG_COUNT);
    
    printf("Training data loaded: %d images\n", trainCount);
    
    // Load test data
    int testCount = 0;
    
    printf("Loading screenshot test images...\n");
    loadImagesFromDir(screenshots_test_dir, 1, testFeatures, &testCount, MAX_IMG_COUNT);
    
    printf("Loading non-screenshot test images...\n");
    loadImagesFromDir(non_screenshots_test_dir, 0, testFeatures, &testCount, MAX_IMG_COUNT);
    
    printf("Test data loaded: %d images\n", testCount);
    
    metrics.data_loading_time = (double)(clock() - data_start) / CLOCKS_PER_SEC;
    
    // Evaluate model
    printf("Evaluating model...\n");
    metrics.accuracy = evaluateModel(trainFeatures, trainCount, testFeatures, testCount);
    printf("Model accuracy on test set: %.2f%%\n", metrics.accuracy * 100);
    
    // Save model
    printf("Saving model to %s...\n", modelPath);
    if (saveModel(modelPath, trainFeatures, trainCount) == 0) {
        printf("Model saved successfully\n");
    } else {
        fprintf(stderr, "Failed to save model\n");
    }
    
    // Update metrics
    metrics.total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    metrics.total_images = trainCount + testCount;
    metrics.peak_memory_usage = trainCount * sizeof(Feature) + testCount * sizeof(Feature);
    
    // Print performance metrics
    printf("\n");
    printPerformanceMetrics(&metrics);
    
    // Cleanup
    free(trainFeatures);
    free(testFeatures);
    
    return 0;
}
