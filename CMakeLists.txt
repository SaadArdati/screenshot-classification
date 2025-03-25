cmake_minimum_required(VERSION 3.18)
project(screenshot-classification CUDA CXX)

# Detect platform
if(WIN32)
    # Windows-specific settings
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust based on your Windows GPU
    # Find CUDA installation on Windows
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
    endif()
else()
    # Jetson Orin Nano settings
    set(CMAKE_CUDA_ARCHITECTURES 87)  # Ampere architecture
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find required packages
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Add source files
set(SOURCES
    src/main.cu
    src/feature_extraction.cu
    src/knn.cu
)

# Add executable
add_executable(train_cuda ${SOURCES})

# Link CUDA libraries
target_link_libraries(train_cuda ${CUDA_LIBRARIES})

# Set CUDA specific flags
set_target_properties(train_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create include directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/include)

# Enable debug symbols in CUDA code
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(train_cuda PROPERTIES 
        CUDA_FLAGS "-G"
    )
endif()

# Print platform-specific information
message(STATUS "Building for platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# Add platform-specific compile definitions
if(WIN32)
    target_compile_definitions(train_cuda PRIVATE WINDOWS_BUILD)
else()
    target_compile_definitions(train_cuda PRIVATE JETSON_BUILD)
endif()

