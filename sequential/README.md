# Screenshot Classification with CUDA

A CUDA-accelerated implementation of screenshot classification using K-nearest neighbors (KNN) and grayscale histogram features.

## Project Overview

This project implements a machine learning solution for classifying images as screenshots or non-screenshots using:
- CUDA-accelerated feature extraction
- Parallel KNN classification
- Real-time performance monitoring
- Visualization tools for performance analysis

## Hardware Requirements

- NVIDIA Jetson Orin Nano Developer Kit
- Minimum 8GB RAM
- Storage space for dataset

## Software Requirements

1. JetPack SDK 6.0 or later
2. CUDA Toolkit (included in JetPack)
3. CMake 3.18 or later
4. Python 3.x with matplotlib (for visualizations)
5. GCC/G++ compiler

## Project Structure

```
screenshot-classification/
├── CMakeLists.txt          # Build configuration
├── include/
│   ├── stb_image.h        # Image loading library
│   ├── cuda_utils.cuh     # CUDA utilities
│   └── common.h          # Shared definitions
├── src/
│   ├── main.cu           # Main program
│   ├── feature_extraction.cu  # CUDA feature extraction
│   ├── knn.cu            # CUDA KNN implementation
│   └── cuda_visualize_performance.py  # Performance visualization
└── split_data/           # Dataset directory
    ├── screenshots_256x256/
    │   ├── train/
    │   └── test/
    └── non_screenshot_256x256/
        ├── train/
        └── test/
```

## Setup Instructions

1. **Clone and Prepare Environment**:
   ```bash
   # Clone repository
   git clone [repository-url]
   cd screenshot-classification

   # Create necessary directories
   mkdir -p include src lib build
   mkdir -p split_data/{screenshots_256x256,non_screenshot_256x256}/{train,test}

   # Download stb_image.h
   wget -O include/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
   ```

2. **Prepare Dataset**:
   - Place screenshot images in `split_data/screenshots_256x256/{train,test}`
   - Place non-screenshot images in `split_data/non_screenshot_256x256/{train,test}`
   - All images should be 256x256 pixels in RGB format

3. **Build the Project**:
   ```bash
   cd build
   cmake ..
   make -j4
   ```

## Running the Code

1. **Training and Classification**:
   ```bash
   ./train_cuda [optional: output_model_path]
   ```
   Default model path is `trained_model.bin`

2. **Monitor GPU Performance**:
   ```bash
   # In another terminal
   watch -n 0.5 nvidia-smi
   ```

3. **Visualize Performance**:
   ```bash
   # After running the training
   python3 src/cuda_visualize_performance.py
   ```
   This will generate three plots:
   - Time distribution pie chart
   - Processing time bar chart
   - Additional metrics visualization

## Performance Monitoring

The code provides detailed performance metrics:
- Data loading time
- Feature extraction time
- KNN classification time
- GPU memory usage
- Processing speed (images/second)
- Classification accuracy

## Recent Changes and Improvements

1. **CUDA Optimizations**:
   - Batch processing for efficient GPU utilization
   - Shared memory usage for histogram computation
   - Parallel KNN implementation
   - Optimized memory transfers

2. **Performance Monitoring**:
   - Detailed timing for each phase
   - GPU memory usage tracking
   - Real-time progress indicators
   - Accuracy measurements

3. **Visualization Tools**:
   - Added Python script for performance visualization
   - Multiple chart types for different metrics
   - Easy-to-understand performance breakdown

4. **Code Structure**:
   - Modular CUDA implementation
   - Separated feature extraction and KNN logic
   - Improved error handling
   - Better memory management

## Troubleshooting

1. **Memory Issues**:
   - Reduce `MAX_BATCH_SIZE` in `include/common.h`
   - Monitor GPU memory usage with `nvidia-smi`
   - Close other GPU-intensive applications

2. **Performance Issues**:
   - Ensure proper cooling for the Jetson
   - Monitor thermal throttling:
     ```bash
     watch -n 0.5 cat /sys/devices/virtual/thermal/thermal_zone*/temp
     ```

3. **Build Issues**:
   - Verify CUDA installation:
     ```bash
     nvcc --version
     ```
   - Check CMake version:
     ```bash
     cmake --version
     ```

## Hardware Specifications

Tested on NVIDIA Jetson Orin Nano:
- Device Name: Orin
- Compute Capability: 8.7
- Total Global Memory: 7.44 GB
- Shared Memory per Block: 48 KB
- L2 Cache Size: 2048 KB
- Memory Clock Rate: 1.02 GHz
- Memory Bus Width: 128 bits
- Max Threads per Block: 1024
- Max Threads per MultiProcessor: 1536

## Dataset Statistics

- Training Set:
  - Screenshots: 9,600 images
  - Non-Screenshots: 25,932 images
  - Total: 35,532 images

- Test Set:
  - Screenshots: 2,399 images
  - Non-Screenshots: 6,481 images
  - Total: 8,880 images

## Performance Results

| Metric                    | Value          |
|--------------------------|----------------|
| Total Processing Time    | 126.69 seconds |
| Data Loading Time        | 125.16 seconds |
| Feature Extraction Time  | 3.05 seconds   |
| KNN Computation Time     | 0.21 seconds   |
| Processing Speed         | 255.84 img/s   |
| Peak GPU Memory Usage    | 4,502.64 MB    |
| Classification Accuracy  | 62.81%         |

### Phase Distribution
- Data Loading: 98.8%
- Feature Extraction: 2.4%
- KNN Classification: 0.2%

### Performance Analysis

1. **Processing Efficiency**:
   - The system processes over 255 images per second
   - Total dataset of 32,413 images processed in ~127 seconds
   - Extremely efficient KNN computation (0.21 seconds)

2. **Memory Usage**:
   - Peak memory usage is ~4.5GB
   - Well within the Jetson's 8GB capacity
   - Efficient memory management for large datasets

3. **Bottlenecks**:
   - Data loading dominates (98.8% of total time)
   - GPU computation is very efficient (only 2.6% combined)
   - Potential for I/O optimization

4. **Classification Performance**:
   - Accuracy of 62.81% on test set
   - Room for improvement in classification accuracy
   - Fast inference time (0.21s for KNN)

### Optimization Opportunities

1. **I/O Performance**:
   - Consider data preprocessing
   - Implement async data loading
   - Use memory mapping for large datasets

2. **GPU Utilization**:
   - Current GPU computation is very efficient
   - Could potentially process larger batches
   - Room for parallel data loading

3. **Memory Management**:
   - Current usage (4.5GB) allows for larger batches
   - Could implement memory pooling
   - Potential for streaming larger datasets

## Contributing

Feel free to submit issues and enhancement requests. To contribute:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 