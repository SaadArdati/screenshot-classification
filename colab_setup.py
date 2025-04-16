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

def prepare_data():
    """Run data preparation scripts"""
    print("Running data preparation scripts...")
    
    # Check if we already have sample data
    if os.path.exists("data/screenshots/train/sample1.jpeg"):
        print("Using sample data - skipping data preparation...")
        return
    
    # Copy the scripts from data_prep repo
    run_cmd("cp ../data_prep/resize-dataset.py .")
    run_cmd("cp ../data_prep/split_dataset.py .")
    run_cmd("cp ../data_prep/unfold_images.py .")
    
    # Process Screenshot dataset
    print("Processing screenshot dataset...")
    run_cmd("python -c \"import os; from PIL import Image; os.makedirs('after_processing/screenshots_256x256', exist_ok=True); [Image.open(os.path.join('before_processing/screenshots', f)).resize((256, 256)).convert('RGB').save(os.path.join('after_processing/screenshots_256x256', f)) for f in os.listdir('before_processing/screenshots') if f.endswith(('.jpg', '.jpeg', '.png'))]\"")
    
    # Process Unsplash dataset
    print("Processing Unsplash dataset...")
    run_cmd("sed -i 's/INPUT_DIR = \"before_processing\\/unsplash_400x\"/INPUT_DIR = \"before_processing\\/unsplash_400x\"/g' resize-dataset.py")
    run_cmd("sed -i 's/OUTPUT_DIR = \"after_processing\\/non_screenshot_256x256\"/OUTPUT_DIR = \"after_processing\\/non_screenshot_256x256\"/g' resize-dataset.py")
    run_cmd("python resize-dataset.py")
    
    # Process Google Universal dataset
    print("Processing Google Universal dataset...")
    run_cmd("sed -i 's/SOURCE_DIR = \"before_processing\\/google_universal_256x256\"/SOURCE_DIR = \"before_processing\\/google_universal_256x256\"/g' unfold_images.py")
    run_cmd("sed -i 's/TARGET_DIR = \"after_processing\\/non_screenshot_256x256\"/TARGET_DIR = \"after_processing\\/non_screenshot_256x256\"/g' unfold_images.py")
    run_cmd("python unfold_images.py")
    
    # Split the datasets
    print("Splitting datasets...")
    # Split screenshots
    run_cmd("sed -i 's/SOURCE_DIR = \"after_processing\\/non_screenshot_256x256\"/SOURCE_DIR = \"after_processing\\/screenshots_256x256\"/g' split_dataset.py")
    run_cmd("sed -i 's/TRAIN_DIR = \"split_data\\/non_screenshot_256x256\\/train\"/TRAIN_DIR = \"split_data\\/screenshots_256x256\\/train\"/g' split_dataset.py")
    run_cmd("sed -i 's/TEST_DIR = \"split_data\\/non_screenshot_256x256\\/test\"/TEST_DIR = \"split_data\\/screenshots_256x256\\/test\"/g' split_dataset.py")
    run_cmd("python split_dataset.py")
    
    # Split non-screenshots
    run_cmd("sed -i 's/SOURCE_DIR = \"after_processing\\/screenshots_256x256\"/SOURCE_DIR = \"after_processing\\/non_screenshot_256x256\"/g' split_dataset.py")
    run_cmd("sed -i 's/TRAIN_DIR = \"split_data\\/screenshots_256x256\\/train\"/TRAIN_DIR = \"split_data\\/non_screenshot_256x256\\/train\"/g' split_dataset.py")
    run_cmd("sed -i 's/TEST_DIR = \"split_data\\/screenshots_256x256\\/test\"/TEST_DIR = \"split_data\\/non_screenshot_256x256\\/test\"/g' split_dataset.py")
    run_cmd("python split_dataset.py")
    
    # Check if any files were processed
    screenshot_files = glob.glob("split_data/screenshots_256x256/train/*")
    non_screenshot_files = glob.glob("split_data/non_screenshot_256x256/train/*")
    
    if len(screenshot_files) == 0 or len(non_screenshot_files) == 0:
        print("\nNo files were processed. Downloading sample images for testing...")
        
        # Create sample data directories
        run_cmd("mkdir -p data/screenshots/train data/screenshots/test")
        run_cmd("mkdir -p data/non_screenshots/train data/non_screenshots/test")
        
        # Download sample images
        print("Downloading sample screenshot images...")
        run_cmd("curl -L https://github.com/SaadArdati/screenshot-classification/raw/refactor/test.jpeg -o data/screenshots/train/sample1.jpeg")
        run_cmd("cp data/screenshots/train/sample1.jpeg data/screenshots/test/sample1.jpeg")
        
        print("Downloading sample non-screenshot images...")
        run_cmd("curl -L https://source.unsplash.com/random/256x256 -o data/non_screenshots/train/sample1.jpeg")
        run_cmd("curl -L https://source.unsplash.com/random/256x256 -o data/non_screenshots/test/sample1.jpeg")
        
        print("Sample images downloaded for testing purposes.")
        return
    
    # Move the split data to the expected location
    run_cmd("mkdir -p data/screenshots/train data/screenshots/test data/non_screenshots/train data/non_screenshots/test")
    run_cmd("cp -r split_data/screenshots_256x256/train/* data/screenshots/train/ 2>/dev/null || true")
    run_cmd("cp -r split_data/screenshots_256x256/test/* data/screenshots/test/ 2>/dev/null || true")
    run_cmd("cp -r split_data/non_screenshot_256x256/train/* data/non_screenshots/train/ 2>/dev/null || true")
    run_cmd("cp -r split_data/non_screenshot_256x256/test/* data/non_screenshots/test/ 2>/dev/null || true")

def create_benchmark_script():
    """Create the benchmark script"""
    print("Creating benchmark script...")
    benchmark_script = """#!/bin/bash

MODE=$1
ITERATIONS=5
TOTAL_TIME=0
MODEL_FILE="model_${MODE}.bin"

# Function to run training and capture time
run_training() {
    echo "Training $MODE model..."
    START=$(date +%s.%N)
    
    if [ "$MODE" == "sequential" ]; then
        ./build/bin/train_seq $MODEL_FILE > train_${MODE}_output.txt
    else
        ./build/bin/train_cuda $MODEL_FILE > train_${MODE}_output.txt
    fi
    
    END=$(date +%s.%N)
    TRAINING_TIME=$(echo "$END - $START" | bc)
    echo "Training time: $TRAINING_TIME seconds"
    echo "$TRAINING_TIME" > training_${MODE}_time.txt
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
        START=$(date +%s.%N)
        
        if [ "$MODE" == "sequential" ]; then
            ./build/bin/predict_seq $MODEL_FILE "$SCREENSHOT_IMG" > /dev/null
        else
            ./build/bin/predict_cuda $MODEL_FILE "$SCREENSHOT_IMG" > /dev/null
        fi
        
        END=$(date +%s.%N)
        DIFF=$(echo "$END - $START" | bc)
        echo "Iteration $i: $DIFF seconds"
        echo "$DIFF" >> prediction_${MODE}_times.txt
    done
    
    # Test non-screenshot
    echo "Testing non-screenshot image..."
    for i in $(seq 1 $ITERATIONS); do
        START=$(date +%s.%N)
        
        if [ "$MODE" == "sequential" ]; then
            ./build/bin/predict_seq $MODEL_FILE "$NON_SCREENSHOT_IMG" > /dev/null
        else
            ./build/bin/predict_cuda $MODEL_FILE "$NON_SCREENSHOT_IMG" > /dev/null
        fi
        
        END=$(date +%s.%N)
        DIFF=$(echo "$END - $START" | bc)
        echo "Iteration $i: $DIFF seconds"
        echo "$DIFF" >> prediction_${MODE}_times.txt
    done
    
    # Calculate average prediction time
    TOTAL=$(cat prediction_${MODE}_times.txt | paste -sd+ | bc)
    AVG=$(echo "$TOTAL / (2 * $ITERATIONS)" | bc -l)
    echo "Average prediction time: $AVG seconds"
    echo "$AVG" > avg_prediction_${MODE}_time.txt
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
    """Create the script to plot benchmark results"""
    print("Creating plotting script...")
    plot_script = """import matplotlib.pyplot as plt
import numpy as np
import os

# Read benchmark results
try:
    with open('training_sequential_time.txt', 'r') as f:
        seq_train_time = float(f.read().strip())
except (FileNotFoundError, ValueError):
    seq_train_time = 0
    
try:
    with open('training_parallel_time.txt', 'r') as f:
        cuda_train_time = float(f.read().strip())
except (FileNotFoundError, ValueError):
    # If CUDA benchmarks weren't run
    cuda_train_time = 0
    
try:
    with open('avg_prediction_sequential_time.txt', 'r') as f:
        seq_predict_time = float(f.read().strip())
except (FileNotFoundError, ValueError):
    seq_predict_time = 0
    
try:
    with open('avg_prediction_parallel_time.txt', 'r') as f:
        cuda_predict_time = float(f.read().strip())
except (FileNotFoundError, ValueError):
    # If CUDA benchmarks weren't run
    cuda_predict_time = 0

# Create comparison plot
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training time comparison
train_times = [seq_train_time, cuda_train_time]
ax[0].bar(['Sequential', 'CUDA'], train_times, color=['blue', 'green'])
ax[0].set_title('Training Time Comparison')
ax[0].set_ylabel('Time (seconds)')
for i, v in enumerate(train_times):
    ax[0].text(i, v + 0.1, f'{v:.2f}s', ha='center')

# Prediction time comparison
predict_times = [seq_predict_time, cuda_predict_time]
ax[1].bar(['Sequential', 'CUDA'], predict_times, color=['blue', 'green'])
ax[1].set_title('Prediction Time Comparison')
ax[1].set_ylabel('Time (seconds)')
for i, v in enumerate(predict_times):
    ax[1].text(i, v + 0.002, f'{v:.4f}s', ha='center')

# Speedup calculation
if cuda_train_time > 0 and seq_train_time > 0:
    train_speedup = seq_train_time / cuda_train_time
else:
    train_speedup = 0
    
if cuda_predict_time > 0 and seq_predict_time > 0:
    predict_speedup = seq_predict_time / cuda_predict_time
else:
    predict_speedup = 0

plt.suptitle(f'Sequential vs CUDA Performance Comparison\\nTraining Speedup: {train_speedup:.2f}x, Prediction Speedup: {predict_speedup:.2f}x')
plt.tight_layout()
plt.savefig('benchmark_results.png')
print("Benchmark results saved to benchmark_results.png")
plt.show()
"""
    
    with open("plot_results.py", "w") as f:
        f.write(plot_script)

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
    
    # Display results
    print("\n=== BENCHMARK RESULTS ===")
    try:
        with open("training_sequential_time.txt", "r") as f:
            seq_train_time = float(f.read().strip())
        print(f"Training time (Sequential): {seq_train_time} seconds")
    except:
        print("Could not read sequential training time")
    
    if has_cuda:
        try:
            with open("training_parallel_time.txt", "r") as f:
                cuda_train_time = float(f.read().strip())
            print(f"Training time (CUDA): {cuda_train_time} seconds")
        except:
            print("Could not read CUDA training time")
    
    try:
        with open("avg_prediction_sequential_time.txt", "r") as f:
            seq_predict_time = float(f.read().strip())
        print(f"Average prediction time (Sequential): {seq_predict_time} seconds")
    except:
        print("Could not read sequential prediction time")
    
    if has_cuda:
        try:
            with open("avg_prediction_parallel_time.txt", "r") as f:
                cuda_predict_time = float(f.read().strip())
            print(f"Average prediction time (CUDA): {cuda_predict_time} seconds")
            
            # Calculate speedups
            train_speedup = seq_train_time / cuda_train_time
            predict_speedup = seq_predict_time / cuda_predict_time
            print(f"\nTraining speedup with CUDA: {train_speedup:.2f}x")
            print(f"Prediction speedup with CUDA: {predict_speedup:.2f}x")
        except:
            print("Could not read CUDA prediction time")

def main():
    """Main function to run the entire workflow"""
    print("Starting screenshot classification setup and benchmarking...")
    
    # Install necessary Linux packages
    run_cmd("apt-get update && apt-get install -y cmake build-essential bc")
    
    # Setup Kaggle API
    setup_kaggle()
    
    # Check CUDA availability
    has_cuda = check_cuda()
    
    # Clone repositories
    clone_repositories()
    
    # Download datasets
    download_datasets()
    
    # Prepare data
    prepare_data()
    
    # Create benchmark script
    create_benchmark_script()
    
    # Create plot script
    create_plot_script()
    
    # Build and benchmark
    build_and_benchmark(has_cuda)
    
    print("\nSetup and benchmarking complete!")
    print("You can find the benchmark results in benchmark_results.png")

if __name__ == "__main__":
    main()