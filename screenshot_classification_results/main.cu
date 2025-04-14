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
#include "cuda_utils.cuh"
#include "common.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
} PerformanceMetrics;

// Asynchronous loading structure
typedef struct {
    char dirpath[512];
    int label;
    unsigned char* batch_buffer;
    Feature* features;
    int batch_size;
    int* current_index;
    DIR* dir;
    int loaded;
    int done;
    pthread_mutex_t mutex;
} LoadBatchArgs;

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
    printf("Shared Memory per Block: %d KB\n", (int)(prop.sharedMemPerBlock / 1024));
    printf("L2 Cache Size: %d KB\n", (int)(prop.l2CacheSize / 1024));
    printf("Max Threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / (1000.0f * 1000.0f));
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
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

// Asynchronous batch loading function
void* loadBatchAsync(void* args) {
    LoadBatchArgs* load_args = (LoadBatchArgs*)args;
    struct dirent* entry;
    int width, height, channels;
    int loaded = 0;
    
    pthread_mutex_lock(&load_args->mutex);
    DIR* dir = load_args->dir;
    pthread_mutex_unlock(&load_args->mutex);
    
    while (loaded < load_args->batch_size && (entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) continue;
        
        char fullpath[512];
        #ifdef _WIN32
        snprintf(fullpath, sizeof(fullpath), "%s%s%s", load_args->dirpath, PATH_SEPARATOR, entry->d_name);
        #else
        snprintf(fullpath, sizeof(fullpath), "%s/%s", load_args->dirpath, entry->d_name);
        #endif
        
        // Load image
        unsigned char* img = stbi_load(fullpath, &width, &height, &channels, 3);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", fullpath);
            continue;
        }
        
        // Copy to batch buffer
        const int image_size = width * height * 3;
        pthread_mutex_lock(&load_args->mutex);
        memcpy(load_args->batch_buffer + (loaded * image_size), img, image_size);
        load_args->features[*load_args->current_index + loaded].label = load_args->label;
        pthread_mutex_unlock(&load_args->mutex);
        
        stbi_image_free(img);
        loaded++;
    }
    
    pthread_mutex_lock(&load_args->mutex);
    load_args->loaded = loaded;
    load_args->done = 1;
    pthread_mutex_unlock(&load_args->mutex);
    
    return NULL;
}

// Load a batch of images asynchronously
int startLoadBatch(LoadBatchArgs* args, pthread_t* thread) {
    args->loaded = 0;
    args->done = 0;
    pthread_mutex_init(&args->mutex, NULL);
    
    if (pthread_create(thread, NULL, loadBatchAsync, args) != 0) {
        fprintf(stderr, "Failed to create loading thread\n");
        return 0;
    }
    
    return 1;
}

// Wait for batch loading to complete
int waitForBatchLoading(LoadBatchArgs* args, pthread_t thread) {
    pthread_join(thread, NULL);
    pthread_mutex_destroy(&args->mutex);
    return args->loaded;
}

// Update the main function to include KNN classification and performance monitoring
int main(int argc, char** argv) {
    PerformanceMetrics metrics = {0};
    size_t peak_memory = 0;
    clock_t total_start = clock();
    double data_loading_time = 0.0;
    double feature_extraction_time = 0.0;

    // Initialize CUDA and print device info
    CUDA_CHECK(cudaSetDevice(0));
    printDeviceInfo();

    // Get current working directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        fprintf(stderr, "Failed to get current working directory\n");
        return 1;
    }
    
    // Path to the split_data directory with platform-specific separators
    char screenshots_train_dir[1024], non_screenshots_train_dir[1024];
    char screenshots_test_dir[1024], non_screenshots_test_dir[1024];
    char model_path[1024];
    
    #ifdef _WIN32
    snprintf(screenshots_train_dir, sizeof(screenshots_train_dir), 
             "%s%s%s%s%s%s%s", cwd, PATH_SEPARATOR, "..", PATH_SEPARATOR, "split_data", PATH_SEPARATOR, "screenshots_256x256", PATH_SEPARATOR, "train");
    snprintf(non_screenshots_train_dir, sizeof(non_screenshots_train_dir), 
             "%s%s%s%s%s%s%s", cwd, PATH_SEPARATOR, "..", PATH_SEPARATOR, "split_data", PATH_SEPARATOR, "non_screenshot_256x256", PATH_SEPARATOR, "train");
    snprintf(screenshots_test_dir, sizeof(screenshots_test_dir), 
             "%s%s%s%s%s%s%s", cwd, PATH_SEPARATOR, "..", PATH_SEPARATOR, "split_data", PATH_SEPARATOR, "screenshots_256x256", PATH_SEPARATOR, "test");
    snprintf(non_screenshots_test_dir, sizeof(non_screenshots_test_dir), 
             "%s%s%s%s%s%s%s", cwd, PATH_SEPARATOR, "..", PATH_SEPARATOR, "split_data", PATH_SEPARATOR, "non_screenshot_256x256", PATH_SEPARATOR, "test");
    snprintf(model_path, sizeof(model_path), "%s%s%s", cwd, PATH_SEPARATOR, "trained_model.bin");
    #else
    snprintf(screenshots_train_dir, sizeof(screenshots_train_dir), 
             "%s/../split_data/screenshots_256x256/train", cwd);
    snprintf(non_screenshots_train_dir, sizeof(non_screenshots_train_dir), 
             "%s/../split_data/non_screenshot_256x256/train", cwd);
    snprintf(screenshots_test_dir, sizeof(screenshots_test_dir), 
             "%s/../split_data/screenshots_256x256/test", cwd);
    snprintf(non_screenshots_test_dir, sizeof(non_screenshots_test_dir), 
             "%s/../split_data/non_screenshot_256x256/test", cwd);
    snprintf(model_path, sizeof(model_path), "%s/../trained_model.bin", cwd);
    #endif
    
    printf("Using directories:\n");
    printf("Screenshots train: %s\n", screenshots_train_dir);
    printf("Non-screenshots train: %s\n", non_screenshots_train_dir);
    printf("Screenshots test: %s\n", screenshots_test_dir);
    printf("Non-screenshots test: %s\n", non_screenshots_test_dir);
    printf("Model path: %s\n\n", model_path);
    
    // Parse command line arguments
    if (argc > 1) {
        strncpy(model_path, argv[1], sizeof(model_path) - 1);
    }
    
    // Allocate memory for features and batch processing
    Feature* all_features = (Feature*)malloc(MAX_IMAGES * sizeof(Feature));
    Feature* test_features = (Feature*)malloc(MAX_IMAGES * sizeof(Feature));
    unsigned char* batch_buffer = (unsigned char*)malloc(MAX_BATCH_SIZE * 256 * 256 * 3);
    unsigned char* next_batch_buffer = (unsigned char*)malloc(MAX_BATCH_SIZE * 256 * 256 * 3);
    
    if (!all_features || !test_features || !batch_buffer || !next_batch_buffer) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    // Initialize counters
    int train_size = 0;
    int test_size = 0;
    
    // Start loading data
    clock_t load_start = clock();
    
    // Process training screenshots
    printf("Processing training screenshots...\n");
    DIR* dir = opendir(screenshots_train_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", screenshots_train_dir);
        return 1;
    }
    
    // Set up asynchronous loading
    LoadBatchArgs load_args = {0};
    strncpy(load_args.dirpath, screenshots_train_dir, sizeof(load_args.dirpath) - 1);
    load_args.label = 1;
    load_args.batch_buffer = batch_buffer;
    load_args.features = all_features;
    load_args.batch_size = MAX_BATCH_SIZE;
    load_args.current_index = &train_size;
    load_args.dir = dir;
    
    pthread_t loading_thread;
    if (!startLoadBatch(&load_args, &loading_thread)) {
        closedir(dir);
        return 1;
    }
    
    // Process batches with overlapped loading and computation
    while (1) {
        // Wait for current batch to load
        int loaded = waitForBatchLoading(&load_args, loading_thread);
        if (loaded == 0) break;
        
        // Start loading next batch
        LoadBatchArgs next_load_args = load_args;
        next_load_args.batch_buffer = next_batch_buffer;
        pthread_t next_loading_thread;
        int next_batch_loading = startLoadBatch(&next_load_args, &next_loading_thread);
        
        // Process current batch on GPU
        clock_t batch_start = clock();
        extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, all_features + train_size);
        feature_extraction_time += (double)(clock() - batch_start) / CLOCKS_PER_SEC;
        
        train_size += loaded;
        printf("\rProcessed %d training screenshots", train_size);
        fflush(stdout);
        
        // Swap buffers for next iteration
        unsigned char* temp = batch_buffer;
        batch_buffer = next_batch_buffer;
        next_batch_buffer = temp;
        
        // Update load args for next iteration
        load_args = next_load_args;
        loading_thread = next_loading_thread;
        
        if (!next_batch_loading) break;
    }
    printf("\n");
    closedir(dir);
    
    // Process training non-screenshots
    printf("Processing training non-screenshots...\n");
    dir = opendir(non_screenshots_train_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", non_screenshots_train_dir);
        return 1;
    }
    
    int total_processed = train_size;
    
    // Set up asynchronous loading for non-screenshots
    strncpy(load_args.dirpath, non_screenshots_train_dir, sizeof(load_args.dirpath) - 1);
    load_args.label = 0;
    load_args.batch_buffer = batch_buffer;
    load_args.features = all_features;
    load_args.batch_size = MAX_BATCH_SIZE;
    load_args.current_index = &train_size;
    load_args.dir = dir;
    
    if (!startLoadBatch(&load_args, &loading_thread)) {
        closedir(dir);
        return 1;
    }
    
    // Process batches with overlapped loading and computation
    while (1) {
        // Wait for current batch to load
        int loaded = waitForBatchLoading(&load_args, loading_thread);
        if (loaded == 0) break;
        
        // Start loading next batch
        LoadBatchArgs next_load_args = load_args;
        next_load_args.batch_buffer = next_batch_buffer;
        pthread_t next_loading_thread;
        int next_batch_loading = startLoadBatch(&next_load_args, &next_loading_thread);
        
        // Process current batch on GPU
        clock_t batch_start = clock();
        extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, all_features + train_size);
        feature_extraction_time += (double)(clock() - batch_start) / CLOCKS_PER_SEC;
        
        train_size += loaded;
        printf("\rProcessed %d training non-screenshots", train_size - total_processed);
        fflush(stdout);
        
        // Swap buffers for next iteration
        unsigned char* temp = batch_buffer;
        batch_buffer = next_batch_buffer;
        next_batch_buffer = temp;
        
        // Update load args for next iteration
        load_args = next_load_args;
        loading_thread = next_loading_thread;
        
        if (!next_batch_loading) break;
    }
    printf("\n");
    closedir(dir);
    
    // Process test data
    printf("Processing test data...\n");
    
    // Test screenshots
    dir = opendir(screenshots_test_dir);
    if (dir) {
        // Set up asynchronous loading for test screenshots
        strncpy(load_args.dirpath, screenshots_test_dir, sizeof(load_args.dirpath) - 1);
        load_args.label = 1;
        load_args.batch_buffer = batch_buffer;
        load_args.features = test_features;
        load_args.batch_size = MAX_BATCH_SIZE;
        load_args.current_index = &test_size;
        load_args.dir = dir;
        
        if (!startLoadBatch(&load_args, &loading_thread)) {
            closedir(dir);
            return 1;
        }
        
        // Process batches with overlapped loading and computation
        while (1) {
            // Wait for current batch to load
            int loaded = waitForBatchLoading(&load_args, loading_thread);
            if (loaded == 0) break;
            
            // Start loading next batch
            LoadBatchArgs next_load_args = load_args;
            next_load_args.batch_buffer = next_batch_buffer;
            pthread_t next_loading_thread;
            int next_batch_loading = startLoadBatch(&next_load_args, &next_loading_thread);
            
            // Process current batch on GPU
            clock_t batch_start = clock();
            extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, test_features + test_size);
            feature_extraction_time += (double)(clock() - batch_start) / CLOCKS_PER_SEC;
            
            test_size += loaded;
            printf("\rProcessed %d test screenshots", test_size);
            fflush(stdout);
            
            // Swap buffers for next iteration
            unsigned char* temp = batch_buffer;
            batch_buffer = next_batch_buffer;
            next_batch_buffer = temp;
            
            // Update load args for next iteration
            load_args = next_load_args;
            loading_thread = next_loading_thread;
            
            if (!next_batch_loading) break;
        }
        printf("\n");
        closedir(dir);
    }
    
    // Test non-screenshots
    dir = opendir(non_screenshots_test_dir);
    if (dir) {
        // Set up asynchronous loading for test non-screenshots
        strncpy(load_args.dirpath, non_screenshots_test_dir, sizeof(load_args.dirpath) - 1);
        load_args.label = 0;
        load_args.batch_buffer = batch_buffer;
        load_args.features = test_features;
        load_args.batch_size = MAX_BATCH_SIZE;
        load_args.current_index = &test_size;
        load_args.dir = dir;
        
        if (!startLoadBatch(&load_args, &loading_thread)) {
            closedir(dir);
            return 1;
        }
        
        // Process batches with overlapped loading and computation
        while (1) {
            // Wait for current batch to load
            int loaded = waitForBatchLoading(&load_args, loading_thread);
            if (loaded == 0) break;
            
            // Start loading next batch
            LoadBatchArgs next_load_args = load_args;
            next_load_args.batch_buffer = next_batch_buffer;
            pthread_t next_loading_thread;
            int next_batch_loading = startLoadBatch(&next_load_args, &next_loading_thread);
            
            // Process current batch on GPU
            clock_t batch_start = clock();
            extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, test_features + test_size);
            feature_extraction_time += (double)(clock() - batch_start) / CLOCKS_PER_SEC;
            
            test_size += loaded;
            printf("\rProcessed %d test non-screenshots", test_size);
            fflush(stdout);
            
            // Swap buffers for next iteration
            unsigned char* temp = batch_buffer;
            batch_buffer = next_batch_buffer;
            next_batch_buffer = temp;
            
            // Update load args for next iteration
            load_args = next_load_args;
            loading_thread = next_loading_thread;
            
            if (!next_batch_loading) break;
        }
        printf("\n");
        closedir(dir);
    }
    data_loading_time = (double)(clock() - load_start) / CLOCKS_PER_SEC;
    
    // Save model
    printf("Saving model to %s...\n", model_path);
    if (saveModel(model_path, all_features, train_size) != 0) {
        fprintf(stderr, "Failed to save model\n");
    }
    
    // Record peak memory after feature extraction
    peak_memory = max(peak_memory, getCurrentGPUMemory());
    
    // Perform KNN classification on test set
    printf("\nPerforming KNN classification...\n");
    double knn_times[2] = {0};  // [0] = transfer time, [1] = compute time
    int* predictions = (int*)malloc(test_size * sizeof(int));
    
    if (!predictions) {
        fprintf(stderr, "Failed to allocate memory for predictions\n");
        return 1;
    }
    
    classifyBatchGPU(all_features, train_size, test_features, test_size, predictions, knn_times);
    
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
    metrics.total_images = train_size + test_size;
    
    // Print detailed performance metrics
    printPerformanceMetrics(&metrics);
    
    // Clean up
    free(batch_buffer);
    free(next_batch_buffer);
    free(predictions);
    free(all_features);
    free(test_features);
    
    return 0;
}
