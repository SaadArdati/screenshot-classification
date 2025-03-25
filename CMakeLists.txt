cmake_minimum_required(VERSION 3.18)
project(screenshot-classification CUDA CXX)

# Set CUDA architecture for Jetson Orin Nano
set(CMAKE_CUDA_ARCHITECTURES 87)  # Ampere architecture
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find required packages
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable(train_cuda
    src/main.cu
    src/feature_extraction.cu
    src/knn.cu
)

# Link CUDA libraries
target_link_libraries(train_cuda ${CUDA_LIBRARIES})

# Set CUDA specific flags
set_target_properties(train_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable debug symbols in CUDA code
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(train_cuda PROPERTIES 
        CUDA_FLAGS "-G"
    )
endif()

