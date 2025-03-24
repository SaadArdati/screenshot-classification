#include <stdio.h>
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

// Load all images from `dir`, set label, append to array
int load_folder(const char *dirpath, const int label, Feature **arr, int *size) {
    DIR *d = opendir(dirpath);
    struct dirent *entry;
    if (!d) return -1;

    while ((entry = readdir(d)) != NULL) {
        if (entry->d_type != DT_REG) continue;
        *arr = realloc(*arr, sizeof(Feature) * (*size + 1));
        Feature *f = &(*arr)[*size];
        f->label = label;
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
        if (extract_feature(fullpath, f) == 0) (*size)++;
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
    Neighbor neighbors[n];
    for (int i = 0; i < n; i++) {
        neighbors[i] = (Neighbor){euclidean(&train[i], q), train[i].label};
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
    return (votes >= (K_NEIGHBORS / 2 + 1)) ? 1 : 0;
}

int main(const int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <screenshots_dir> <nonscreenshots_dir> <query_image>\n", argv[0]);
        return 1;
    }
    Feature *dataset = NULL;
    int count = 0;

    load_folder(argv[1], 1, &dataset, &count);
    load_folder(argv[2], 0, &dataset, &count);

    Feature query;
    if (extract_feature(argv[3], &query) != 0) {
        fprintf(stderr, "Failed to load query image.\n");
        return 1;
    }

    const int result = classify_knn(dataset, count, &query);
    printf("Prediction: %s\n", result ? "SCREENSHOT" : "NON-SCREENSHOT");
    free(dataset);
    return 0;
}
