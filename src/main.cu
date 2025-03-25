#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "common.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Function declarations
extern "C" void extractFeaturesGPU(const unsigned char* h_images, int batch_size,
                                 int width, int height, int channels,
                                 Feature* h_features);

// Print device information
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
    printf("\n");
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

int main(int argc, char** argv) {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    printDeviceInfo();

    // Path to the split_data directory
    const char* screenshots_train_dir = "split_data/screenshots_256x256/train";
    const char* non_screenshots_train_dir = "split_data/non_screenshot_256x256/train";
    const char* model_path = (argc > 1) ? argv[1] : "trained_model.bin";

    // Allocate host memory for batch processing
    const int image_size = 256 * 256 * 3;  // Assuming 256x256 RGB images
    unsigned char* batch_buffer = (unsigned char*)malloc(MAX_BATCH_SIZE * image_size);
    Feature* features = (Feature*)malloc(MAX_BATCH_SIZE * sizeof(Feature));

    if (!batch_buffer || !features) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Process screenshots
    DIR* dir = opendir(screenshots_train_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", screenshots_train_dir);
        return 1;
    }

    int total_processed = 0;
    while (1) {
        int loaded = loadImageBatch(screenshots_train_dir, 1, batch_buffer,
                                  features, MAX_BATCH_SIZE, &total_processed, dir);
        if (loaded == 0) break;

        // Process batch on GPU
        extractFeaturesGPU(batch_buffer, loaded, 256, 256, 3, features + total_processed);
        total_processed += loaded;
    }
    closedir(dir);

    // Process non-screenshots (similar to screenshots)
    // ... (similar code for non-screenshots)

    // Clean up
    free(batch_buffer);
    free(features);

    return 0;
} 