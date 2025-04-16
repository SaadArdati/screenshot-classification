#!/bin/bash

# Script to build the screenshot classification project in different modes

# Default mode is sequential
MODE="sequential"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode=*) MODE="${1#*=}" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate mode
if [ "$MODE" != "sequential" ] && [ "$MODE" != "parallel" ]; then
    echo "Error: Invalid mode. Use --mode=sequential or --mode=parallel"
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build based on mode
if [ "$MODE" == "sequential" ]; then
    echo "Building in sequential mode (C only)"
    cmake -DBUILD_MODE=sequential ..
    make
elif [ "$MODE" == "parallel" ]; then
    echo "Building in parallel mode (CUDA)"
    
    # Check if CUDA is available
    if ! command -v nvcc &> /dev/null; then
        echo "Error: CUDA compiler (nvcc) not found. Cannot build in parallel mode."
        echo "Please install CUDA or use --mode=sequential instead."
        exit 1
    fi
    
    cmake -DBUILD_MODE=parallel ..
    make
fi

echo ""
echo "Build completed in $MODE mode."
echo "Executables can be found in the build/bin directory." 