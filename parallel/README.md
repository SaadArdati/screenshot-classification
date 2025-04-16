# CUDA Screenshot Classification

A high-performance, GPU-accelerated system for detecting screenshots in images using machine learning techniques.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (minimum version 10.0 recommended)
- C/C++ development environment

## Building

Build both the training and prediction utilities using:

```bash
make
```

## Usage

### Training

To train a new model from a dataset:

```bash
./training <path_to_training_images> <output_model_file>
```

The training data should be organized with screenshots in a "screenshots" directory and non-screenshots in a "non-screenshots" directory.

### Prediction

To classify a new image:

```bash
./predict <model_file> <image_path>
```

## Features

- CUDA-accelerated feature extraction and KNN classification
- Statistical analysis to detect common UI patterns
- High-speed batch processing for training
- Detailed performance metrics

## Implementation Details

The system uses a two-stage classification approach:
1. Statistical pattern analysis to detect common UI elements
2. K-nearest neighbors algorithm for appearance-based classification

The CUDA implementation offers significant performance improvements over the CPU version while maintaining the same classification accuracy. 