import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import glob
import argparse
from pathlib import Path
import requests
from google.colab import files

def run_cmd(cmd, verbose=True):
    """Run shell commands and capture output"""
    if verbose:
        print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    if verbose:
        print(stdout)
        if stderr and process.returncode != 0:
            print(f"Error: {stderr}")
    return stdout, stderr, process.returncode

def check_cuda():
    """Check if CUDA is available"""
    print("Checking CUDA availability...")
    try:
        stdout, stderr, code = run_cmd("nvcc --version")
        if code != 0:
            print("CUDA compiler (nvcc) not found. Only sequential implementation will be built.")
            return False
            
        stdout, stderr, code = run_cmd("nvidia-smi")
        if code != 0:
            print("NVIDIA GPU not accessible. Only sequential implementation will be built.")
            return False
            
        return True
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        print("Only sequential implementation will be built.")
        return False

def setup_kaggle():
    """Setup Kaggle API and credentials"""
    print("Setting up Kaggle API...")
    run_cmd("pip install kaggle")
    run_cmd("mkdir -p ~/.kaggle")

    # Check if kaggle.json already exists
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("\nKaggle API credentials required.")
        print("Please upload your kaggle.json file below.")
        
        try:
            uploaded = files.upload()
            if 'kaggle.json' in uploaded:
                run_cmd("cp kaggle.json ~/.kaggle/")
                run_cmd("chmod 600 ~/.kaggle/kaggle.json")
                print("Kaggle credentials uploaded and configured.")
                return True
            else:
                print("No kaggle.json file was uploaded.")
                return False
        except Exception as e:
            print(f"Error during file upload: {e}")
            return False
    else:
        print("Kaggle credentials found.")
        run_cmd("chmod 600 ~/.kaggle/kaggle.json")
        return True

def test_kaggle_auth():
    """Test Kaggle API authentication"""
    print("Testing Kaggle API authentication...")
    stdout, stderr, code = run_cmd("kaggle datasets list", verbose=False)
    if code != 0 or "401 - Unauthorized" in stderr:
        print("Kaggle API authentication failed. Will use direct downloads instead.")
        return False
    return True

def download_from_direct_url(url, output_file):
    """Download file from direct URL"""
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total_size)
                    sys.stdout.write("\r[%s%s] %d%%" % ('=' * done, ' ' * (50-done), 
                                                       int(downloaded/total_size*100)))
                    sys.stdout.flush()
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def clone_repositories():
    """Clone the necessary repositories"""
    print("Cloning screenshot classification repository (refactor branch)...")
    run_cmd("git clone -b refactor https://github.com/SaadArdati/screenshot-classification.git")

    print("Cloning data preprocessing repository...")
    run_cmd("git clone https://github.com/SaadArdati/parallel-prog-data-sanitization.git data_prep")

    # Change to project directory
    os.chdir("screenshot-classification")

def download_datasets():
    """Download and extract datasets from Kaggle or direct URLs"""
    print("Creating directory structure for datasets...")
    run_cmd("mkdir -p before_processing/screenshots")
    run_cmd("mkdir -p before_processing/unsplash_400x")
    run_cmd("mkdir -p before_processing/google_universal_256x256")
    run_cmd("mkdir -p after_processing/screenshots_256x256")
    run_cmd("mkdir -p after_processing/non_screenshot_256x256")
    run_cmd("mkdir -p split_data/screenshots_256x256/train")
    run_cmd("mkdir -p split_data/screenshots_256x256/test")
    run_cmd("mkdir -p split_data/non_screenshot_256x256/train")
    run_cmd("mkdir -p split_data/non_screenshot_256x256/test")
    
    # First try Kaggle API
    kaggle_auth_works = test_kaggle_auth()
    
    if kaggle_auth_works:
        print("Using Kaggle API to download datasets...")
        # Download datasets from Kaggle
        print("Downloading Android screenshots dataset...")
        run_cmd("kaggle datasets download -d uzairkhan45/categorized-android-apps-screenshots")
        
        print("Downloading Unsplash random images dataset...")
        run_cmd("kaggle datasets download -d lprdosmil/unsplash-random-images-collection")
        
        print("Downloading Google universal images dataset...")
        run_cmd("kaggle datasets download -d saberghaderi/google-universal-image-256x256-guie-jpgcsv")
    else:
        print("Using direct downloads for datasets...")
        # Use direct download links (these are example URLs, replace with actual mirrors if available)
        download_from_direct_url(
            "https://storage.googleapis.com/kaggle-datasets-mirrors/uzairkhan45/categorized-android-apps-screenshots/archive.zip",
            "categorized-android-apps-screenshots.zip"
        )
        
        download_from_direct_url(
            "https://storage.googleapis.com/kaggle-datasets-mirrors/lprdosmil/unsplash-random-images-collection/archive.zip",
            "unsplash-random-images-collection.zip"
        )
        
        download_from_direct_url(
            "https://storage.googleapis.com/kaggle-datasets-mirrors/saberghaderi/google-universal-image-256x256-guie-jpgcsv/archive.zip",
            "google-universal-image-256x256-guie-jpgcsv.zip"
        )
    
    # Check if downloads exist before trying to extract
    datasets_exist = True
    
    if not os.path.exists("categorized-android-apps-screenshots.zip"):
        print("Warning: Android screenshots dataset not downloaded.")
        datasets_exist = False
    
    if not os.path.exists("unsplash-random-images-collection.zip"):
        print("Warning: Unsplash dataset not downloaded.")
        datasets_exist = False
    
    if not os.path.exists("google-universal-image-256x256-guie-jpgcsv.zip"):
        print("Warning: Google universal dataset not downloaded.")
        datasets_exist = False
    
    if not datasets_exist:
        print("\nDataset downloads failed. Downloading sample images for testing...")
        
        # Create sample data directories
        run_cmd("mkdir -p data/screenshots/train data/screenshots/test")
        run_cmd("mkdir -p data/non_screenshots/train data/non_screenshots/test")
        
        # Download sample images from GitHub
        print("Downloading sample screenshot images...")
        run_cmd("curl -L https://github.com/SaadArdati/screenshot-classification/raw/refactor/test.jpeg -o data/screenshots/train/sample1.jpeg")
        run_cmd("cp data/screenshots/train/sample1.jpeg data/screenshots/test/sample1.jpeg")
        
        print("Downloading sample non-screenshot images...")
        run_cmd("curl -L https://source.unsplash.com/random/256x256 -o data/non_screenshots/train/sample1.jpeg")
        run_cmd("curl -L https://source.unsplash.com/random/256x256 -o data/non_screenshots/test/sample1.jpeg")
        
        print("Sample images downloaded for testing purposes.")
        return
    
    print("Extracting datasets...")
    
    if os.path.exists("categorized-android-apps-screenshots.zip"):
        run_cmd("unzip -q categorized-android-apps-screenshots.zip -d before_processing/screenshots/")
    
    if os.path.exists("unsplash-random-images-collection.zip"):
        run_cmd("unzip -q unsplash-random-images-collection.zip -d before_processing/unsplash_400x/")
    
    if os.path.exists("google-universal-image-256x256-guie-jpgcsv.zip"):
        run_cmd("unzip -q google-universal-image-256x256-guie-jpgcsv.zip -d before_processing/google_universal_256x256/")

def prepare_sample_data():
    """Prepare sample data using the provided images"""
    print("Setting up sample test data...")
    
    # Create sample data directories
    run_cmd("mkdir -p data/screenshots/train data/screenshots/test")
    run_cmd("mkdir -p data/non_screenshots/train data/non_screenshots/test")
    
    # Check if uploaded images exist
    screenshot_exists = os.path.exists("screenshot.jpeg")
    nonscreenshot_exists = os.path.exists("nonscreenshot.png")
    
    if not screenshot_exists or not nonscreenshot_exists:
        print("Please upload screenshot.jpeg and nonscreenshot.png files")
        try:
            uploaded = files.upload()
            if 'screenshot.jpeg' in uploaded:
                screenshot_exists = True
            if 'nonscreenshot.png' in uploaded:
                nonscreenshot_exists = True
        except Exception as e:
            print(f"Error during file upload: {e}")
    
    # Copy files to appropriate directories
    if screenshot_exists:
        run_cmd("cp screenshot.jpeg data/screenshots/train/screenshot.jpeg")
        run_cmd("cp screenshot.jpeg data/screenshots/test/screenshot.jpeg")
        print("Screenshot image added to dataset")
    else:
        # Fallback to test.jpeg from the repository
        run_cmd("cp test.jpeg data/screenshots/train/screenshot.jpeg")
        run_cmd("cp test.jpeg data/screenshots/test/screenshot.jpeg")
        print("Using test.jpeg from repository as screenshot")
    
    if nonscreenshot_exists:
        run_cmd("cp nonscreenshot.png data/non_screenshots/train/nonscreenshot.png")
        run_cmd("cp nonscreenshot.png data/non_screenshots/test/nonscreenshot.png")
        print("Non-screenshot image added to dataset")
    else:
        # Create a simple non-screenshot image if needed
        print("Creating a sample non-screenshot image")
        run_cmd("python -c \"from PIL import Image; img = Image.new('RGB', (256, 256), (255, 200, 200)); img.save('data/non_screenshots/train/sample_nonscreenshot.png'); img.save('data/non_screenshots/test/sample_nonscreenshot.png')\"")
    
    print("Sample data setup complete")

def create_benchmark_script():
    """Create and write the benchmark script to a file"""
    print("Creating benchmark script...")
    
    benchmark_script = """#!/bin/bash

MODE=$1
ITERATIONS=5
MODEL_FILE="model_${MODE}.bin"

# Function to run training and capture time
run_training() {
    echo "Training $MODE model..."
    
    if [ "$MODE" == "sequential" ]; then
        /usr/bin/time -v ./build/bin/train_seq $MODEL_FILE > train_${MODE}_output.txt 2> train_${MODE}_time.txt
    else
        /usr/bin/time -v ./build/bin/train_cuda $MODEL_FILE > train_${MODE}_output.txt 2> train_${MODE}_time.txt
    fi
    
    # Extract and save metrics
    echo "Extracting training metrics..."
    
    # Extract times from output
    grep "Total Processing Time" train_${MODE}_output.txt | awk '{print $4}' > train_${MODE}_total_time.txt
    grep "Data Loading Time" train_${MODE}_output.txt | awk '{print $4}' > train_${MODE}_loading_time.txt
    grep "Memory Usage" train_${MODE}_output.txt | awk '{print $3}' > train_${MODE}_memory.txt
    
    # Extract additional metrics from output
    grep "Accuracy" train_${MODE}_output.txt | awk '{print $2}' > train_${MODE}_accuracy.txt
    
    # Extract additional metrics from time command
    grep "Maximum resident set size" train_${MODE}_time.txt | awk '{print $6}' > train_${MODE}_max_memory.txt
    grep "User time" train_${MODE}_time.txt | awk '{print $4}' > train_${MODE}_user_time.txt
    grep "System time" train_${MODE}_time.txt | awk '{print $4}' > train_${MODE}_system_time.txt
    
    # Calculate processing speed
    TOTAL_EXAMPLES=$(grep "Number of training examples" train_${MODE}_output.txt | awk '{print $5}')
    TOTAL_TIME=$(cat train_${MODE}_total_time.txt)
    echo "scale=2; $TOTAL_EXAMPLES / $TOTAL_TIME" | bc > train_${MODE}_images_per_sec.txt
    
    echo "Training metrics extracted"
}

# Function to run prediction and capture time
run_prediction() {
    echo "Running prediction benchmarks for $MODE implementation..."
    rm -f prediction_${MODE}_times.txt
    
    # Get sample images for testing
    SCREENSHOT_IMG=$(find data/screenshots/test -type f | head -n 1)
    NON_SCREENSHOT_IMG=$(find data/non_screenshots/test -type f | head -n 1)
    
    echo "Using screenshot test image: $SCREENSHOT_IMG"
    echo "Using non-screenshot test image: $NON_SCREENSHOT_IMG"
    
    # Test screenshot
    echo "Testing screenshot image..."
    for i in $(seq 1 $ITERATIONS); do
        if [ "$MODE" == "sequential" ]; then
            /usr/bin/time -v ./build/bin/predict_seq $MODEL_FILE "$SCREENSHOT_IMG" > predict_${MODE}_output_$i.txt 2> predict_${MODE}_time_$i.txt
        else
            /usr/bin/time -v ./build/bin/predict_cuda $MODEL_FILE "$SCREENSHOT_IMG" > predict_${MODE}_output_$i.txt 2> predict_${MODE}_time_$i.txt
        fi
        
        # Extract detailed metrics
        grep "Total Processing Time" predict_${MODE}_output_$i.txt | awk '{print $4}' >> predict_${MODE}_total_time.txt
        grep "Feature Extraction Time" predict_${MODE}_output_$i.txt | awk '{print $4}' >> predict_${MODE}_feature_time.txt
        grep "Classification Time" predict_${MODE}_output_$i.txt | awk '{print $3}' >> predict_${MODE}_knn_time.txt
        grep "Memory Usage" predict_${MODE}_output_$i.txt | awk '{print $3}' >> predict_${MODE}_memory.txt
    done
    
    # Calculate average prediction metrics
    echo "scale=6; $(cat predict_${MODE}_total_time.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_total_time.txt
    echo "scale=6; $(cat predict_${MODE}_feature_time.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_feature_time.txt
    echo "scale=6; $(cat predict_${MODE}_knn_time.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_knn_time.txt
    echo "scale=2; $(cat predict_${MODE}_memory.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_memory.txt
    
    echo "Average prediction time: $(cat avg_predict_${MODE}_total_time.txt) seconds"
    echo "Average feature extraction time: $(cat avg_predict_${MODE}_feature_time.txt) seconds"
    echo "Average KNN computation time: $(cat avg_predict_${MODE}_knn_time.txt) seconds"
    echo "Average memory usage: $(cat avg_predict_${MODE}_memory.txt) MB"
}

# Main benchmark
echo "Starting $MODE benchmarks..."
run_training
run_prediction
echo "$MODE benchmarking complete."
"""
    
    with open("benchmark.sh", "w") as f:
        f.write(benchmark_script)
    
    run_cmd("chmod +x benchmark.sh")

def create_plot_script():
    """Create the script to plot benchmark results and save it as a separate file"""
    print("Creating plotting script...")
    
    plot_script = """#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

def read_metric(filename, default=0):
    try:
        with open(filename, 'r') as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return default

# Read training metrics
seq_train_total_time = read_metric('train_sequential_total_time.txt')
seq_train_loading_time = read_metric('train_sequential_loading_time.txt')
seq_train_memory = read_metric('train_sequential_memory.txt')
seq_train_accuracy = read_metric('train_sequential_accuracy.txt')
seq_train_images_per_sec = read_metric('train_sequential_images_per_sec.txt')

cuda_train_total_time = read_metric('train_parallel_total_time.txt')
cuda_train_loading_time = read_metric('train_parallel_loading_time.txt')
cuda_train_memory = read_metric('train_parallel_memory.txt')
cuda_train_accuracy = read_metric('train_parallel_accuracy.txt')
cuda_train_images_per_sec = read_metric('train_parallel_images_per_sec.txt')

# Read prediction metrics
seq_predict_total_time = read_metric('avg_predict_sequential_total_time.txt')
seq_predict_feature_time = read_metric('avg_predict_sequential_feature_time.txt')
seq_predict_knn_time = read_metric('avg_predict_sequential_knn_time.txt')
seq_predict_memory = read_metric('avg_predict_sequential_memory.txt')

cuda_predict_total_time = read_metric('avg_predict_parallel_total_time.txt')
cuda_predict_feature_time = read_metric('avg_predict_parallel_feature_time.txt')
cuda_predict_knn_time = read_metric('avg_predict_parallel_knn_time.txt')
cuda_predict_memory = read_metric('avg_predict_parallel_memory.txt')

# Calculate speedups
if cuda_train_total_time > 0 and seq_train_total_time > 0:
    train_speedup = seq_train_total_time / cuda_train_total_time
else:
    train_speedup = 0

if cuda_predict_total_time > 0 and seq_predict_total_time > 0:
    predict_speedup = seq_predict_total_time / cuda_predict_total_time
else:
    predict_speedup = 0

if cuda_predict_feature_time > 0 and seq_predict_feature_time > 0:
    feature_speedup = seq_predict_feature_time / cuda_predict_feature_time
else:
    feature_speedup = 0

if cuda_predict_knn_time > 0 and seq_predict_knn_time > 0:
    knn_speedup = seq_predict_knn_time / cuda_predict_knn_time
else:
    knn_speedup = 0

# Create time comparison plot
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
train_times = [seq_train_total_time, cuda_train_total_time]
plt.bar(['Sequential', 'CUDA'], train_times, color=['blue', 'green'])
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
for i, v in enumerate(train_times):
    plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')

plt.subplot(1, 2, 2)
predict_times = [seq_predict_total_time, cuda_predict_total_time]
plt.bar(['Sequential', 'CUDA'], predict_times, color=['blue', 'green'])
plt.title('Prediction Time Comparison')
plt.ylabel('Time (seconds)')
for i, v in enumerate(predict_times):
    plt.text(i, v + 0.002, f'{v:.4f}s', ha='center')

plt.suptitle(f'Sequential vs CUDA Processing Time Comparison\\nTraining Speedup: {train_speedup:.2f}x, Prediction Speedup: {predict_speedup:.2f}x')
plt.tight_layout()
plt.savefig('time_comparison.png')
print("Time comparison saved to time_comparison.png")

# Create detailed time distribution plot
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
seq_times = [seq_train_loading_time, seq_predict_feature_time, seq_predict_knn_time]
labels = ['Data Loading', 'Feature Extraction', 'KNN Computation']
plt.bar(labels, seq_times, color=['skyblue', 'royalblue', 'navy'])
plt.title('Sequential Time Distribution')
plt.ylabel('Time (seconds)')
for i, v in enumerate(seq_times):
    plt.text(i, v + 0.01, f'{v:.3f}s', ha='center')

plt.subplot(1, 2, 2)
cuda_times = [cuda_train_loading_time, cuda_predict_feature_time, cuda_predict_knn_time]
plt.bar(labels, cuda_times, color=['lightgreen', 'forestgreen', 'darkgreen'])
plt.title('CUDA Time Distribution')
plt.ylabel('Time (seconds)')
for i, v in enumerate(cuda_times):
    plt.text(i, v + 0.01, f'{v:.3f}s', ha='center')

plt.suptitle('Time Distribution Comparison')
plt.tight_layout()
plt.savefig('time_distribution_comparison.png')
print("Time distribution comparison saved to time_distribution_comparison.png")

# Create memory comparison plot
plt.figure(figsize=(10, 6))
memory_usage = [seq_train_memory, cuda_train_memory]
plt.bar(['Sequential', 'CUDA'], memory_usage, color=['blue', 'green'])
plt.title('Memory Usage Comparison')
plt.ylabel('Memory (MB)')
for i, v in enumerate(memory_usage):
    plt.text(i, v + 0.5, f'{v:.2f} MB', ha='center')
plt.tight_layout()
plt.savefig('memory_comparison.png')
print("Memory comparison saved to memory_comparison.png")

# Create processing speed comparison
plt.figure(figsize=(10, 6))
processing_speed = [seq_train_images_per_sec, cuda_train_images_per_sec]
plt.bar(['Sequential', 'CUDA'], processing_speed, color=['blue', 'green'])
plt.title('Processing Speed Comparison')
plt.ylabel('Images per Second')
for i, v in enumerate(processing_speed):
    plt.text(i, v + 5, f'{v:.2f} img/s', ha='center')
plt.tight_layout()
plt.savefig('processing_time_comparison.png')
print("Processing speed comparison saved to processing_time_comparison.png")

# Create speedup comparison
plt.figure(figsize=(10, 6))
speedups = [train_speedup, predict_speedup, feature_speedup, knn_speedup]
labels = ['Training', 'Prediction', 'Feature Extraction', 'KNN Computation']
plt.bar(labels, speedups, color=['purple', 'magenta', 'crimson', 'darkorange'])
plt.title('CUDA Speedup Comparison')
plt.ylabel('Speedup Factor (x)')
for i, v in enumerate(speedups):
    plt.text(i, v + 0.1, f'{v:.2f}x', ha='center')
plt.tight_layout()
plt.savefig('speedup_comparison.png')
print("Speedup comparison saved to speedup_comparison.png")

# Create metrics comparison dashboard
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.bar(['Sequential', 'CUDA'], [seq_train_total_time, cuda_train_total_time], color=['blue', 'green'])
plt.title('Total Processing Time (s)')

plt.subplot(2, 2, 2)
plt.bar(['Sequential', 'CUDA'], [seq_train_memory, cuda_train_memory], color=['blue', 'green'])
plt.title('Memory Usage (MB)')

plt.subplot(2, 2, 3)
plt.bar(['Sequential', 'CUDA'], [seq_train_images_per_sec, cuda_train_images_per_sec], color=['blue', 'green'])
plt.title('Processing Speed (img/s)')

plt.subplot(2, 2, 4)
plt.bar(['Sequential', 'CUDA'], [seq_train_accuracy, cuda_train_accuracy], color=['blue', 'green'])
plt.title('Classification Accuracy (%)')

plt.suptitle('Performance Metrics Comparison')
plt.tight_layout()
plt.savefig('metrics_comparison.png')
print("Metrics comparison saved to metrics_comparison.png")

# Print summary report
print("\\nBenchmark Results Summary:")
print("==========================")
print(f"Training Time (Sequential): {seq_train_total_time:.4f} seconds")
if cuda_train_total_time > 0:
    print(f"Training Time (CUDA): {cuda_train_total_time:.4f} seconds")
    print(f"Training Speedup: {train_speedup:.2f}x")

print(f"\\nPrediction Time (Sequential): {seq_predict_total_time:.6f} seconds")
if cuda_predict_total_time > 0:
    print(f"Prediction Time (CUDA): {cuda_predict_total_time:.6f} seconds")
    print(f"Prediction Speedup: {predict_speedup:.2f}x")

print(f"\\nFeature Extraction Time (Sequential): {seq_predict_feature_time:.6f} seconds")
if cuda_predict_feature_time > 0:
    print(f"Feature Extraction Time (CUDA): {cuda_predict_feature_time:.6f} seconds")
    print(f"Feature Extraction Speedup: {feature_speedup:.2f}x")

print(f"\\nKNN Computation Time (Sequential): {seq_predict_knn_time:.6f} seconds")
if cuda_predict_knn_time > 0:
    print(f"KNN Computation Time (CUDA): {cuda_predict_knn_time:.6f} seconds")
    print(f"KNN Computation Speedup: {knn_speedup:.2f}x")

print(f"\\nMemory Usage (Sequential): {seq_train_memory:.2f} MB")
if cuda_train_memory > 0:
    print(f"Memory Usage (CUDA): {cuda_train_memory:.2f} MB")

print(f"\\nProcessing Speed (Sequential): {seq_train_images_per_sec:.2f} images/second")
if cuda_train_images_per_sec > 0:
    print(f"Processing Speed (CUDA): {cuda_train_images_per_sec:.2f} images/second")

print(f"\\nClassification Accuracy (Sequential): {seq_train_accuracy:.2f}%")
if cuda_train_accuracy > 0:
    print(f"Classification Accuracy (CUDA): {cuda_train_accuracy:.2f}%")
"""
    
    with open("plot_results.py", "w") as f:
        f.write(plot_script)
    
    # Make it executable
    run_cmd("chmod +x plot_results.py")

def build_and_benchmark(has_cuda):
    """Build and benchmark the project"""
    print("Building sequential implementation...")
    run_cmd("./build.sh --mode=sequential")
    run_cmd("./benchmark.sh sequential")
    
    if has_cuda:
        print("Building CUDA implementation...")
        run_cmd("./build.sh --mode=parallel")
        run_cmd("./benchmark.sh parallel")
    
    # Run the plotting script
    run_cmd("python plot_results.py")
    
    # Display results summary
    print("\nBenchmark complete! Results:")
    print("----------------------------")
    # This will be handled by plot_results.py

def main():
    """Main function to run the entire workflow"""
    print("Starting screenshot classification setup and benchmarking...")
    
    # Install necessary Linux packages
    run_cmd("apt-get update && apt-get install -y cmake build-essential bc time")
    
    # Setup Kaggle API
    setup_kaggle()
    
    # Check CUDA availability
    has_cuda = check_cuda()
    
    # Clone repositories
    clone_repositories()
    
    # Prepare sample data using provided images
    prepare_sample_data()
    
    # Create benchmark script as a separate file
    create_benchmark_script()
    
    # Create plot script as a separate file
    create_plot_script()
    
    # Build and benchmark
    build_and_benchmark(has_cuda)
    
    print("\nSetup and benchmarking complete!")
    print("You can find the benchmark results in the visualization PNG files:")
    print("- time_comparison.png")
    print("- time_distribution_comparison.png")
    print("- memory_comparison.png")
    print("- processing_time_comparison.png")
    print("- speedup_comparison.png")
    print("- metrics_comparison.png")

if __name__ == "__main__":
    main()