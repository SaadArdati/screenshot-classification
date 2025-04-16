#!/usr/bin/env python3
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

plt.suptitle(f'Sequential vs CUDA Processing Time Comparison\nTraining Speedup: {train_speedup:.2f}x, Prediction Speedup: {predict_speedup:.2f}x')
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
print("\nBenchmark Results Summary:")
print("==========================")
print(f"Training Time (Sequential): {seq_train_total_time:.4f} seconds")
if cuda_train_total_time > 0:
    print(f"Training Time (CUDA): {cuda_train_total_time:.4f} seconds")
    print(f"Training Speedup: {train_speedup:.2f}x")

print(f"\nPrediction Time (Sequential): {seq_predict_total_time:.6f} seconds")
if cuda_predict_total_time > 0:
    print(f"Prediction Time (CUDA): {cuda_predict_total_time:.6f} seconds")
    print(f"Prediction Speedup: {predict_speedup:.2f}x")

print(f"\nFeature Extraction Time (Sequential): {seq_predict_feature_time:.6f} seconds")
if cuda_predict_feature_time > 0:
    print(f"Feature Extraction Time (CUDA): {cuda_predict_feature_time:.6f} seconds")
    print(f"Feature Extraction Speedup: {feature_speedup:.2f}x")

print(f"\nKNN Computation Time (Sequential): {seq_predict_knn_time:.6f} seconds")
if cuda_predict_knn_time > 0:
    print(f"KNN Computation Time (CUDA): {cuda_predict_knn_time:.6f} seconds")
    print(f"KNN Computation Speedup: {knn_speedup:.2f}x")

print(f"\nMemory Usage (Sequential): {seq_train_memory:.2f} MB")
if cuda_train_memory > 0:
    print(f"Memory Usage (CUDA): {cuda_train_memory:.2f} MB")

print(f"\nProcessing Speed (Sequential): {seq_train_images_per_sec:.2f} images/second")
if cuda_train_images_per_sec > 0:
    print(f"Processing Speed (CUDA): {cuda_train_images_per_sec:.2f} images/second")

print(f"\nClassification Accuracy (Sequential): {seq_train_accuracy:.2f}%")
if cuda_train_accuracy > 0:
    print(f"Classification Accuracy (CUDA): {cuda_train_accuracy:.2f}%") 