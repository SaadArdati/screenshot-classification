cmake_minimum_required(VERSION 3.10)

# Set mode option (sequential or parallel)
set(BUILD_MODE "sequential" CACHE STRING "Build mode: sequential or parallel")
# Check valid mode
if(NOT BUILD_MODE STREQUAL "sequential" AND NOT BUILD_MODE STREQUAL "parallel")
  message(FATAL_ERROR "Invalid BUILD_MODE. Use -DBUILD_MODE=sequential or -DBUILD_MODE=parallel")
endif()

# Configure project based on mode
if(BUILD_MODE STREQUAL "parallel")
  message(STATUS "Building in parallel (CUDA) mode")
  project(screenshot-classification LANGUAGES C CXX CUDA)
  
  # Set CUDA architecture based on your GPU
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  
  # Find CUDA
  find_package(CUDA REQUIRED)
else()
  message(STATUS "Building in sequential (C-only) mode")
  project(screenshot-classification LANGUAGES C CXX)
endif()

# Common compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Include directories
include_directories(include)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add appropriate subdirectory based on mode
if(BUILD_MODE STREQUAL "sequential")
  add_subdirectory(sequential)
else()
  add_subdirectory(parallel)
endif()

# Create a data directory for storing datasets and models
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data) 