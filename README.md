# Screenshot Classification with CUDA

A screenshot classification system with both sequential (C) and CUDA-accelerated implementations using K-nearest neighbors (KNN) and grayscale histogram features.

## Project Overview

This project implements a machine learning solution for classifying images as screenshots or non-screenshots using:
- Grayscale histogram features with edge detection
- K-nearest neighbors classification
- Both sequential (CPU) and parallel (CUDA) implementations
- Performance monitoring tools

## Hardware Requirements

- Any system with a C compiler for the sequential implementation
- NVIDIA GPU with CUDA support (only for the parallel implementation)
- Minimum 8GB RAM recommended
- Storage space for dataset

## Software Requirements

1. C compiler (GCC, Clang, etc.)
2. CMake 3.10 or later (for the full build)
3. CUDA Toolkit 11.0 or later (only for the parallel implementation)

## Project Structure

```
screenshot-classification/
├── CMakeLists.txt           # Main build configuration
├── build.sh                 # Convenience build script
├── include/                 # Shared header files
│   ├── common.h            # Common definitions
│   ├── cuda_utils.cuh      # CUDA utilities
│   └── stb_image.h         # Image loading library
├── sequential/             # Sequential CPU implementation
│   ├── CMakeLists.txt      # Sequential build config
│   ├── Makefile           # Standalone Makefile for sequential
│   ├── training.c          # Training on CPU
│   └── predict.c           # Prediction on CPU
├── parallel/               # Parallel CUDA implementation
│   ├── CMakeLists.txt      # CUDA build config
│   ├── training.cu         # Training on GPU
│   ├── predict.cu          # Prediction on GPU
│   ├── feature_extraction.cu # Feature extraction kernels
│   └── knn.cu              # KNN classification kernels
└── data/                   # Dataset directory
    ├── screenshots/
    │   ├── train/
    │   └── test/
    └── non_screenshots/
        ├── train/
        └── test/
```

## Build Instructions

### Option 1: Using the Convenience Build Script

The easiest way to build the project is using the provided build script:

```bash
# Build in sequential mode (C only, no CUDA required)
./build.sh --mode=sequential

# Build in parallel mode (CUDA)
./build.sh --mode=parallel
```

### Option 2: Build Only Sequential Implementation (No CUDA Required)

This option is perfect for systems without CUDA, like macOS or systems without NVIDIA GPUs.

```bash
# Method 1: Using the standalone Makefile in the sequential directory
cd sequential
make

# Method 2: Using CMake with sequential mode flag
mkdir build && cd build
cmake -DBUILD_MODE=sequential ..
make
```

### Option 3: Build Only CUDA Implementation

This requires a working CUDA installation:

```bash
mkdir build && cd build
cmake -DBUILD_MODE=parallel ..
make
```

By default, if no mode is specified, the system will build in sequential mode.

## Dataset Preparation

Place your images in the following directories:
   - Screenshots: `data/screenshots/{train,test}`
   - Non-screenshots: `data/non_screenshots/{train,test}`

## Running the Code

### Sequential Implementation:

```bash
# Using the standalone build
cd sequential
./train_seq [output_model_path]
./predict_seq [model_path] [image_path]

# Using the CMake build
./bin/train_seq [output_model_path]
./bin/predict_seq [model_path] [image_path]
```

### CUDA Implementation (if built):

```bash
./bin/train_cuda [output_model_path]
./bin/predict_cuda [model_path] [image_path]
```

## Performance Comparison

| Metric                  | CPU Sequential  | CUDA Parallel  | Improvement |
|-------------------------|-----------------|----------------|-------------|
| Total Processing Time   | ~126.69 seconds | ~3.26 seconds  | ~39x faster |
| Feature Extraction Time | ~3.05 seconds   | ~0.08 seconds  | ~38x faster |
| KNN Computation Time    | ~0.21 seconds   | ~0.006 seconds | ~35x faster |
| Processing Speed        | ~255.84 img/s   | ~9950 img/s    | ~39x faster |
| Classification Accuracy | ~62.81%         | ~62.81%        | Same        |

## Project Features

1. **Enhanced Feature Extraction**:
   - Grayscale histograms
   - Edge detection histograms
   - Region-specific features (status bar, navigation bar)
   - Statistical UI pattern analysis

2. **CUDA Optimizations**:
   - Shared memory for histogram computation
   - Batch processing for efficient GPU utilization
   - Parallel KNN implementation with reduction
   - Optimized memory transfers

3. **Performance Monitoring**:
   - Detailed timing for each phase
   - GPU memory usage tracking
   - Accuracy measurements

## Troubleshooting

1. **Missing CUDA error**: 
   - If you don't have CUDA installed, use the sequential-only build
   - Use `./build.sh --mode=sequential` or run `make` directly in the sequential directory

2. **Memory Issues**:
   - Reduce `MAX_BATCH_SIZE` in `include/common.h`
   - For CUDA builds, monitor GPU memory usage with `nvidia-smi`

3. **Performance Issues**:
   - For CUDA builds, ensure your GPU drivers are up to date
   - Adjust `THREADS_PER_BLOCK` in `include/common.h`

4. **Build Issues**:
   - If your CMake version is too old, update it or use the standalone Makefile
   - For CUDA builds, verify your CUDA installation:
     ```bash
     nvcc --version
     ``` 