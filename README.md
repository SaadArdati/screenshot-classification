# Screenshot Classification with CUDA

A CUDA-accelerated implementation of screenshot classification using K-nearest neighbors (KNN) and grayscale histogram features.

## Hardware Requirements

- NVIDIA Jetson Orin Nano Developer Kit
- Minimum 8GB RAM
- Storage space for dataset

## Software Requirements

1. JetPack SDK 6.0 or later
2. CUDA Toolkit (included in JetPack)
3. CMake 3.18 or later
4. GCC/G++ compiler

## Setup Instructions for Jetson Orin Nano

1. **Install Required Packages**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake build-essential
   ```

2. **Clone the Repository**:
   ```bash
   git clone [repository-url]
   cd screenshot-classification
   ```

3. **Prepare Dataset**:
   Create the following directory structure:
   ```
   split_data/
   ├── screenshots_256x256/
   │   ├── train/
   │   │   └── [screenshot images]
   │   └── test/
   │       └── [screenshot images]
   └── non_screenshot_256x256/
       ├── train/
       │   └── [non-screenshot images]
       └── test/
           └── [non-screenshot images]
   ```
   
   Note: All images should be 256x256 pixels in RGB format.

4. **Build the Project**:
   ```bash
   mkdir build
   cd build
   cmake ..
   make -j4
   ```

## Running the Code

1. **Training**:
   ```bash
   ./train_cuda [optional: output_model_path]
   ```
   Default model path is `trained_model.bin`

2. **Monitor Performance**:
   - The program will display real-time processing statistics
   - GPU utilization can be monitored using:
     ```bash
     watch -n 0.5 nvidia-smi
     ```

## Performance Optimization

The code is optimized for the Jetson Orin Nano with:
- Batch processing (32 images per batch)
- Shared memory usage for histogram computation
- Efficient memory transfers
- Parallel histogram computation

Typical performance metrics on Jetson Orin Nano:
- Processing Speed: ~XXX images/second
- Memory Usage: ~XXX MB
- GPU Utilization: ~XX%

## Troubleshooting

1. **Out of Memory Errors**:
   - Reduce `MAX_BATCH_SIZE` in `include/common.h`
   - Close other GPU-intensive applications

2. **Performance Issues**:
   - Ensure proper cooling for the Jetson
   - Monitor thermal throttling using:
     ```bash
     watch -n 0.5 cat /sys/devices/virtual/thermal/thermal_zone*/temp
     ```

3. **Build Errors**:
   - Ensure CUDA toolkit is properly installed:
     ```bash
     nvcc --version
     ```
   - Check CMake version:
     ```bash
     cmake --version
     ```

## Project Structure

```
screenshot-classification/
├── CMakeLists.txt          # Build configuration
├── include/
│   ├── cuda_utils.cuh      # CUDA utility functions
│   └── common.h           # Shared definitions
├── src/
│   ├── main.cu            # Main program
│   ├── feature_extraction.cu  # CUDA feature extraction
│   └── knn.cu             # CUDA KNN implementation
└── README.md
```

## Performance Comparison

| Metric              | CPU Version | GPU Version |
|---------------------|-------------|-------------|
| Processing Speed    | X img/s     | Y img/s     |
| Memory Usage        | X MB        | Y MB        |
| Training Time       | X seconds   | Y seconds   |
| Power Consumption   | X W         | Y W         |

## Contributing

Feel free to submit issues and enhancement requests! 