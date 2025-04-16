import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import glob
import argparse
import shutil
from pathlib import Path
import requests
from google.colab import files

def run_cmd(cmd, verbose=True, capture_output=False):
    """Run shell commands and capture output"""
    if verbose:
        print(f"Running: {cmd}")
    
    if capture_output:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')
        
        if verbose:
            print(stdout)
            if stderr and process.returncode != 0:
                print(f"Error: {stderr}")
        return stdout, stderr, process.returncode
    else:
        # Run command and pass output directly to console
        return_code = subprocess.call(cmd, shell=True)
        return None, None, return_code

def check_cuda():
    """Check if CUDA is available"""
    print("Checking CUDA availability...")
    try:
        stdout, stderr, code = run_cmd("nvcc --version", capture_output=True)
        if code != 0:
            print("CUDA compiler (nvcc) not found. Exiting.")
            sys.exit(1)
            
        stdout, stderr, code = run_cmd("nvidia-smi", capture_output=True)
        if code != 0:
            print("NVIDIA GPU not accessible. Exiting.")
            sys.exit(1)
            
        return True
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        print("Exiting.")
        sys.exit(1)

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
    stdout, stderr, code = run_cmd("kaggle datasets list", verbose=False, capture_output=True)
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
    
    # Make build script executable
    print("Making build script executable...")
    run_cmd("chmod +x build.sh")
    run_cmd("chmod +x benchmark.sh")

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
        print("Screenshot image not found, benchmarking may fail")
    
    if nonscreenshot_exists:
        run_cmd("cp nonscreenshot.png data/non_screenshots/train/nonscreenshot.png")
        run_cmd("cp nonscreenshot.png data/non_screenshots/test/nonscreenshot.png")
        print("Non-screenshot image added to dataset")
    else:
        print("Non-screenshot image not found, benchmarking may fail")
    
    print("Sample data setup complete")

def benchmark():
    """Run benchmarking by running executables and capturing output"""
    print("\nRunning benchmarking...")
    
    # Create results directory
    run_cmd("mkdir -p benchmark_results")
    
    # Define benchmark metrics
    metrics = {
        'sequential': {
            'train_time': 0.0,
            'loading_time': 0.0,
            'train_memory': 0.0,
            'accuracy': 0.0,
            'images_per_sec': 0.0,
            'predict_time': 0.0,
            'feature_time': 0.0,
            'knn_time': 0.0,
            'predict_memory': 0.0
        },
        'parallel': {
            'train_time': 0.0,
            'loading_time': 0.0,
            'train_memory': 0.0,
            'accuracy': 0.0,
            'images_per_sec': 0.0,
            'predict_time': 0.0,
            'feature_time': 0.0,
            'knn_time': 0.0,
            'predict_memory': 0.0
        }
    }
    
    # Check if executables exist
    seq_train_exists = os.path.exists("build/bin/train_seq")
    seq_predict_exists = os.path.exists("build/bin/predict_seq")
    cuda_train_exists = os.path.exists("build/bin/train_cuda")
    cuda_predict_exists = os.path.exists("build/bin/predict_cuda")
    
    print(f"Sequential executables exist: Train {seq_train_exists}, Predict {seq_predict_exists}")
    print(f"CUDA executables exist: Train {cuda_train_exists}, Predict {cuda_predict_exists}")
    
    # Exit if not all executables exist
    if not seq_train_exists or not seq_predict_exists or not cuda_train_exists or not cuda_predict_exists:
        print("Error: Both sequential AND CUDA executables must be available for benchmarking. Exiting.")
        sys.exit(1)
    
    # Run sequential training benchmark
    print("\nRunning sequential training...")
    stdout, stderr, code = run_cmd("./build/bin/train_seq model_sequential.bin", capture_output=True)
    
    if code == 0:
        print("Sequential training completed successfully")
        # Parse output for metrics
        try:
            for line in stdout.splitlines():
                if "Total Processing Time" in line:
                    metrics['sequential']['train_time'] = float(line.split()[-1])
                elif "Data Loading Time" in line:
                    metrics['sequential']['loading_time'] = float(line.split()[-1])
                elif "Memory Usage" in line:
                    metrics['sequential']['train_memory'] = float(line.split()[-2])
                elif "Accuracy" in line:
                    metrics['sequential']['accuracy'] = float(line.split()[-2])
                elif "Number of training examples" in line:
                    examples = int(line.split()[-1])
                    if metrics['sequential']['train_time'] > 0:
                        metrics['sequential']['images_per_sec'] = examples / metrics['sequential']['train_time']
        except Exception as e:
            print(f"Error parsing sequential training metrics: {e}")
    else:
        print(f"Sequential training failed with code {code}")
        sys.exit(1)
    
    # Run sequential prediction benchmark
    print("\nRunning sequential prediction...")
    # Find a test image
    screenshot_img = next(iter(glob.glob("data/screenshots/test/*.*")), None)
    
    if screenshot_img:
        stdout, stderr, code = run_cmd(f"./build/bin/predict_seq model_sequential.bin \"{screenshot_img}\"", capture_output=True)
        
        if code == 0:
            try:
                for line in stdout.splitlines():
                    if "Total Processing Time" in line:
                        metrics['sequential']['predict_time'] = float(line.split()[-1])
                    elif "Feature Extraction Time" in line:
                        metrics['sequential']['feature_time'] = float(line.split()[-1])
                    elif "Classification Time" in line:
                        metrics['sequential']['knn_time'] = float(line.split()[-1])
                    elif "Memory Usage" in line:
                        metrics['sequential']['predict_memory'] = float(line.split()[-2])
            except Exception as e:
                print(f"Error parsing sequential prediction metrics: {e}")
        else:
            print(f"Sequential prediction failed with code {code}")
            sys.exit(1)
    else:
        print("No test image found for prediction benchmark")
        sys.exit(1)
    
    # Run CUDA training benchmark
    print("\nRunning CUDA training...")
    stdout, stderr, code = run_cmd("./build/bin/train_cuda model_parallel.bin", capture_output=True)
    
    if code == 0:
        print("CUDA training completed successfully")
        # Parse output for metrics
        try:
            for line in stdout.splitlines():
                if "Total Processing Time" in line:
                    metrics['parallel']['train_time'] = float(line.split()[-1])
                elif "Data Loading Time" in line:
                    metrics['parallel']['loading_time'] = float(line.split()[-1])
                elif "Memory Usage" in line:
                    metrics['parallel']['train_memory'] = float(line.split()[-2])
                elif "Accuracy" in line:
                    metrics['parallel']['accuracy'] = float(line.split()[-2])
                elif "Number of training examples" in line:
                    examples = int(line.split()[-1])
                    if metrics['parallel']['train_time'] > 0:
                        metrics['parallel']['images_per_sec'] = examples / metrics['parallel']['train_time']
        except Exception as e:
            print(f"Error parsing CUDA training metrics: {e}")
    else:
        print(f"CUDA training failed with code {code}")
        sys.exit(1)
    
    # Run CUDA prediction benchmark
    print("\nRunning CUDA prediction...")
    if screenshot_img:
        stdout, stderr, code = run_cmd(f"./build/bin/predict_cuda model_parallel.bin \"{screenshot_img}\"", capture_output=True)
        
        if code == 0:
            try:
                for line in stdout.splitlines():
                    if "Total Processing Time" in line:
                        metrics['parallel']['predict_time'] = float(line.split()[-1])
                    elif "Feature Extraction Time" in line:
                        metrics['parallel']['feature_time'] = float(line.split()[-1])
                    elif "Classification Time" in line:
                        metrics['parallel']['knn_time'] = float(line.split()[-1])
                    elif "Memory Usage" in line:
                        metrics['parallel']['predict_memory'] = float(line.split()[-2])
            except Exception as e:
                print(f"Error parsing CUDA prediction metrics: {e}")
        else:
            print(f"CUDA prediction failed with code {code}")
            sys.exit(1)
    
    # Generate visualization based on the collected metrics
    generate_visualizations(metrics)

def generate_visualizations(metrics):
    """Generate visualization plots based on the gathered metrics"""
    print("\nGenerating visualization plots...")
    
    # Calculate speedups
    seq_train_total_time = metrics['sequential']['train_time']
    cuda_train_total_time = metrics['parallel']['train_time']
    seq_predict_total_time = metrics['sequential']['predict_time']
    cuda_predict_total_time = metrics['parallel']['predict_time']
    seq_predict_feature_time = metrics['sequential']['feature_time']
    cuda_predict_feature_time = metrics['parallel']['feature_time']
    seq_predict_knn_time = metrics['sequential']['knn_time']
    cuda_predict_knn_time = metrics['parallel']['knn_time']
    
    train_speedup = seq_train_total_time / cuda_train_total_time if cuda_train_total_time > 0 else 0
    predict_speedup = seq_predict_total_time / cuda_predict_total_time if cuda_predict_total_time > 0 else 0
    feature_speedup = seq_predict_feature_time / cuda_predict_feature_time if cuda_predict_feature_time > 0 else 0
    knn_speedup = seq_predict_knn_time / cuda_predict_knn_time if cuda_predict_knn_time > 0 else 0
    
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
    
    plt.suptitle(f'Sequential vs CUDA Processing Time Comparison\nTraining Speedup: {train_speedup:.2f}x, Prediction Speedup: {predict_speedup:.2f}x')
    plt.tight_layout()
    plt.savefig('time_comparison.png')
    print("Time comparison saved to time_comparison.png")
    
    # Create detailed time distribution plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    seq_times = [metrics['sequential']['loading_time'], metrics['sequential']['feature_time'], metrics['sequential']['knn_time']]
    labels = ['Data Loading', 'Feature Extraction', 'KNN Computation']
    plt.bar(labels, seq_times, color=['skyblue', 'royalblue', 'navy'])
    plt.title('Sequential Time Distribution')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(seq_times):
        plt.text(i, v + 0.01, f'{v:.3f}s', ha='center')
    
    plt.subplot(1, 2, 2)
    cuda_times = [metrics['parallel']['loading_time'], metrics['parallel']['feature_time'], metrics['parallel']['knn_time']]
    plt.bar(labels, cuda_times, color=['lightgreen', 'forestgreen', 'darkgreen'])
    plt.title('CUDA Time Distribution')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(cuda_times):
        plt.text(i, v + 0.01, f'{v:.3f}s', ha='center')
    
    plt.suptitle('Time Distribution Comparison')
    plt.tight_layout()
    plt.savefig('time_distribution_comparison.png')
    print("Time distribution comparison saved to time_distribution_comparison.png")
    
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
    
    # Print summary report
    print("\nBenchmark Results Summary:")
    print("==========================")
    print(f"Training Time (Sequential): {seq_train_total_time:.4f} seconds")
    print(f"Training Time (CUDA): {cuda_train_total_time:.4f} seconds")
    print(f"Training Speedup: {train_speedup:.2f}x")
    
    print(f"\nPrediction Time (Sequential): {seq_predict_total_time:.6f} seconds")
    print(f"Prediction Time (CUDA): {cuda_predict_total_time:.6f} seconds")
    print(f"Prediction Speedup: {predict_speedup:.2f}x")
    
    print(f"\nFeature Extraction Time (Sequential): {seq_predict_feature_time:.6f} seconds")
    print(f"Feature Extraction Time (CUDA): {cuda_predict_feature_time:.6f} seconds")
    print(f"Feature Extraction Speedup: {feature_speedup:.2f}x")
    
    print(f"\nKNN Computation Time (Sequential): {seq_predict_knn_time:.6f} seconds")
    print(f"KNN Computation Time (CUDA): {cuda_predict_knn_time:.6f} seconds")
    print(f"KNN Computation Speedup: {knn_speedup:.2f}x")

def build():
    """Build both sequential and CUDA implementations with detailed error reporting"""
    print("Building sequential implementation...")
    
    # First, check if we're in the correct directory
    stdout, stderr, code = run_cmd("pwd", capture_output=True)
    print(f"Current directory: {stdout.strip()}")
    
    # Check build.sh exists and is executable
    if not os.path.exists("build.sh"):
        print("Error: build.sh not found in current directory.")
        return False
    
    # Check CMake is installed
    stdout, stderr, code = run_cmd("cmake --version", capture_output=True)
    if code != 0:
        print("Error: CMake not found. Please install CMake.")
        return False
    
    # Show sequential build details
    print("Running sequential build with output:")
    stdout, stderr, code = run_cmd("./build.sh --mode=sequential", capture_output=True)
    print(f"Build stdout: {stdout}")
    if stderr:
        print(f"Build stderr: {stderr}")
    
    # Check if sequential build succeeded
    if not os.path.exists("build/bin/train_seq") or not os.path.exists("build/bin/predict_seq"):
        print("Error: Sequential build failed. Trying to diagnose the issue...")
        
        # Check CMake output
        if os.path.exists("build/CMakeCache.txt"):
            stdout, stderr, code = run_cmd("cat build/CMakeCache.txt | grep -E 'CMAKE_C_COMPILER|CMAKE_CXX_COMPILER'", capture_output=True)
            print(f"CMake compiler settings: {stdout}")
        else:
            print("CMake cache file not found.")
        
        # Check if build directory was created
        if not os.path.exists("build"):
            print("Error: Build directory not created.")
            return False
        
        # Check for common build files
        stdout, stderr, code = run_cmd("ls -la build/", capture_output=True)
        print(f"Build directory contents: {stdout}")
        
        # Check for compilation errors in CMake logs
        if os.path.exists("build/CMakeFiles/CMakeError.log"):
            stdout, stderr, code = run_cmd("tail -n 50 build/CMakeFiles/CMakeError.log", capture_output=True)
            print(f"CMake error log: {stdout}")
        
        # Try to manually build
        print("Attempting manual build with more verbose output...")
        run_cmd("mkdir -p build")
        os.chdir("build")
        stdout, stderr, code = run_cmd("cmake -DBUILD_MODE=sequential ..", capture_output=True)
        print(f"CMake output: {stdout}")
        if stderr:
            print(f"CMake errors: {stderr}")
        
        stdout, stderr, code = run_cmd("make VERBOSE=1", capture_output=True)
        print(f"Make output: {stdout}")
        if stderr:
            print(f"Make errors: {stderr}")
        
        os.chdir("..")
        
        # Check again after manual build
        if not os.path.exists("build/bin/train_seq") or not os.path.exists("build/bin/predict_seq"):
            print("Error: Sequential build failed even after manual attempt.")
            
            # Check if executables were built but in a different location
            stdout, stderr, code = run_cmd("find build -name '*_seq' -type f", capture_output=True)
            if stdout:
                print(f"Found executables in non-standard locations: {stdout}")
                print("Trying to copy them to the expected location...")
                run_cmd("mkdir -p build/bin")
                for line in stdout.splitlines():
                    exe_path = line.strip()
                    if exe_path:
                        run_cmd(f"cp {exe_path} build/bin/")
                
                # Check once more
                if os.path.exists("build/bin/train_seq") and os.path.exists("build/bin/predict_seq"):
                    print("Successfully copied executables to the correct location.")
                else:
                    print("Failed to find or copy required executables.")
                    return False
            else:
                print("No sequential executables found anywhere in the build directory.")
                return False
    
    print("Sequential build successful.")
    
    print("Building CUDA implementation...")
    stdout, stderr, code = run_cmd("./build.sh --mode=parallel", capture_output=True)
    print(f"CUDA build stdout: {stdout}")
    if stderr:
        print(f"CUDA build stderr: {stderr}")
    
    # Check if CUDA build succeeded
    if not os.path.exists("build/bin/train_cuda") or not os.path.exists("build/bin/predict_cuda"):
        print("Error: CUDA build failed. Trying to diagnose the issue...")
        
        # Check CUDA toolkit information
        stdout, stderr, code = run_cmd("nvcc --version", capture_output=True)
        print(f"NVCC version: {stdout}")
        
        # Check GPU information
        stdout, stderr, code = run_cmd("nvidia-smi", capture_output=True)
        print(f"GPU information: {stdout}")
        
        # Try to manually build
        print("Attempting manual CUDA build with more verbose output...")
        run_cmd("mkdir -p build")
        os.chdir("build")
        stdout, stderr, code = run_cmd("cmake -DBUILD_MODE=parallel ..", capture_output=True)
        print(f"CMake output: {stdout}")
        if stderr:
            print(f"CMake errors: {stderr}")
        
        stdout, stderr, code = run_cmd("make VERBOSE=1", capture_output=True)
        print(f"Make output: {stdout}")
        if stderr:
            print(f"Make errors: {stderr}")
        
        os.chdir("..")
        
        # Check again after manual build
        if not os.path.exists("build/bin/train_cuda") or not os.path.exists("build/bin/predict_cuda"):
            print("Error: CUDA build failed even after manual attempt.")
            
            # Check if executables were built but in a different location
            stdout, stderr, code = run_cmd("find build -name '*_cuda' -type f", capture_output=True)
            if stdout:
                print(f"Found executables in non-standard locations: {stdout}")
                print("Trying to copy them to the expected location...")
                run_cmd("mkdir -p build/bin")
                for line in stdout.splitlines():
                    exe_path = line.strip()
                    if exe_path:
                        run_cmd(f"cp {exe_path} build/bin/")
                
                # Check once more
                if os.path.exists("build/bin/train_cuda") and os.path.exists("build/bin/predict_cuda"):
                    print("Successfully copied executables to the correct location.")
                else:
                    print("Failed to find or copy required CUDA executables.")
                    return False
            else:
                print("No CUDA executables found anywhere in the build directory.")
                return False
    
    print("CUDA build successful.")
    
    # Both builds succeeded
    return True

def main():
    """Main function to run the entire workflow"""
    print("Starting screenshot classification setup and benchmarking...")
    
    # Install necessary Linux packages
    run_cmd("apt-get update && apt-get install -y cmake build-essential bc time")
    
    # Setup Kaggle API
    setup_kaggle()
    
    # Check CUDA availability
    check_cuda()
    
    # Clone repositories
    clone_repositories()
    
    # Prepare sample data
    prepare_sample_data()
    
    # Build both implementations
    if build():
        # Run benchmarks if builds succeeded
        benchmark()
        
        print("\nSetup and benchmarking complete!")
        print("You can find the benchmark results in the visualization PNG files:")
        print("- time_comparison.png")
        print("- time_distribution_comparison.png")
        print("- speedup_comparison.png")
    else:
        print("Failed to build both sequential and CUDA implementations. Cannot perform benchmarking.")
        sys.exit(1)

if __name__ == "__main__":
    main()