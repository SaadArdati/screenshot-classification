#ifndef FEATURE_EXTRACTION_CUH
#define FEATURE_EXTRACTION_CUH

#include "common.h"

// Function declarations
void extractFeaturesGPU(const char* imagePath, Feature* feature, void* optional_pool = nullptr);
int classifyBatchGPU(Feature* trainSet, int trainSize, Feature* testSet, int testSize, int* predictions);

#endif // FEATURE_EXTRACTION_CUH
