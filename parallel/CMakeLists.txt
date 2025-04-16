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
    # Tesla T4 settings
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Tesla T4 architecture
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find required packages
find_package(CUDA REQUIRED)

# Create necessary directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Download stb_image.h if it doesn't exist
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/include/stb_image.h")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${CMAKE_SOURCE_DIR}/include/stb_image.h"
        SHOW_PROGRESS
    )
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Add source files
set(SOURCES
    src/main.cu
    src/feature_extraction.cu
    src/knn.cu
)

# Parallel CUDA implementation

# Verify CUDA is available (should already be checked by parent)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create CUDA library with shared components
cuda_add_library(cuda_components STATIC 
    feature_extraction.cu
    knn.cu
)

# Add executables
cuda_add_executable(train_cuda training.cu)
target_link_libraries(train_cuda cuda_components ${CUDA_LIBRARIES})

cuda_add_executable(predict_cuda predict.cu)
target_link_libraries(predict_cuda cuda_components ${CUDA_LIBRARIES})

# Set CUDA properties
set_target_properties(train_cuda predict_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install targets
install(TARGETS train_cuda predict_cuda cuda_components
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Set CUDA specific flags
set_target_properties(train_cuda PROPERTIES
    CUDA_FLAGS "-G"
)

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

# Print project structure information
message(STATUS "Project structure setup:")
message(STATUS "  Include directory: ${CMAKE_SOURCE_DIR}/include")
message(STATUS "  Source directory: ${CMAKE_SOURCE_DIR}/src")
message(STATUS "  Library directory: ${CMAKE_SOURCE_DIR}/lib")

