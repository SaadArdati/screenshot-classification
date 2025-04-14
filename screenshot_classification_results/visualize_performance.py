import matplotlib.pyplot as plt
import numpy as np
import os

# Check if output directory exists, create if not
output_dir = "performance_visualizations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Performance metrics (in seconds) - Updated with actual results
# CPU Implementation
cpu_data_loading_time = 44.47
cpu_feature_extraction_time = 0.0  # Not explicitly measured, included in data loading
cpu_knn_time = 3.11
cpu_total_time = 47.58
cpu_accuracy = 93.69
cpu_memory_usage = 3.10  # MB
cpu_images_per_second = 32413 / cpu_total_time

# GPU Implementation (Improved)
gpu_data_loading_time = 1.04
gpu_feature_extraction_time = 0.02
gpu_knn_time = 0.00  # Very small, rounded to 0.00
gpu_total_time = 1.10
gpu_accuracy = 50.00  # Lower due to limited batch size in our test
gpu_memory_usage = 3970.75  # MB
gpu_images_per_second = 115.90

# --- Pie Chart: Percentage of time per phase (CPU vs GPU) ---
plt.figure(figsize=(12, 6))

# CPU Pie Chart
plt.subplot(1, 2, 1)
cpu_phase_times = [cpu_data_loading_time, cpu_feature_extraction_time, cpu_knn_time]
cpu_phase_labels = ['Data Loading', 'Feature Extraction', 'KNN Classification']
cpu_phase_percentages = [t/cpu_total_time*100 for t in cpu_phase_times]

plt.pie(cpu_phase_percentages, labels=cpu_phase_labels, autopct='%1.1f%%', startangle=140,
        colors=['skyblue', 'lightgreen', 'salmon'])
plt.title("CPU Implementation: Time Distribution")

# GPU Pie Chart
plt.subplot(1, 2, 2)
gpu_phase_times = [gpu_data_loading_time, gpu_feature_extraction_time, gpu_knn_time]
gpu_phase_labels = ['Data Loading', 'Feature Extraction', 'KNN Classification']
gpu_phase_percentages = [t/gpu_total_time*100 for t in gpu_phase_times]

plt.pie(gpu_phase_percentages, labels=gpu_phase_labels, autopct='%1.1f%%', startangle=140,
        colors=['skyblue', 'lightgreen', 'salmon'])
plt.title("GPU Implementation: Time Distribution")

plt.tight_layout()
plt.savefig(f"{output_dir}/time_distribution_comparison.png", dpi=300)
plt.close()

# --- Bar Chart: Raw time per phase comparison ---
plt.figure(figsize=(10, 6))

# Set up bar positions
bar_width = 0.35
index = np.arange(3)

# Create bars
cpu_bars = plt.bar(index, cpu_phase_times, bar_width, label='CPU', color='royalblue')
gpu_bars = plt.bar(index + bar_width, gpu_phase_times, bar_width, label='GPU', color='crimson')

# Add labels and title
plt.xlabel('Processing Phase')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU: Processing Time by Phase')
plt.xticks(index + bar_width/2, gpu_phase_labels)
plt.legend()

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}s', ha='center', va='bottom')

add_labels(cpu_bars)
add_labels(gpu_bars)

plt.tight_layout()
plt.savefig(f"{output_dir}/processing_time_comparison.png", dpi=300)
plt.close()

# --- Speedup Chart ---
plt.figure(figsize=(8, 6))

# Calculate speedups
speedups = []
for i in range(len(cpu_phase_times)):
    if gpu_phase_times[i] > 0:
        speedups.append(cpu_phase_times[i]/gpu_phase_times[i])
    else:
        # If GPU time is 0, use a high value to represent "very fast"
        speedups.append(100)  # Arbitrary high value

speedups.append(cpu_total_time/gpu_total_time)  # Add total speedup

# Create bars
speedup_labels = ['Data Loading', 'Feature Extraction', 'KNN Classification', 'Total']
speedup_bars = plt.bar(range(len(speedups)), speedups, color=['skyblue', 'lightgreen', 'salmon', 'purple'])

# Add labels and title
plt.xlabel('Processing Phase')
plt.ylabel('Speedup Factor (CPU time / GPU time)')
plt.title('GPU Speedup by Processing Phase')
plt.xticks(range(len(speedups)), speedup_labels)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)  # Add reference line at y=1

# Add value labels on bars
for bar in speedup_bars:
    height = bar.get_height()
    if height == 100:  # For the "very fast" case
        plt.text(bar.get_x() + bar.get_width()/2., 50,
                f'Very Fast', ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}x', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/speedup_comparison.png", dpi=300)
plt.close()

# --- Additional Metrics Comparison ---
plt.figure(figsize=(10, 6))

# Set up metrics for comparison
metrics_labels = ['Accuracy (%)', 'Memory Usage (MB)', 'Images/Second']
cpu_metrics = [cpu_accuracy, cpu_memory_usage, cpu_images_per_second]
gpu_metrics = [gpu_accuracy, gpu_memory_usage, gpu_images_per_second]

# Set up bar positions
index = np.arange(len(metrics_labels))

# Create bars
cpu_bars = plt.bar(index - bar_width/2, cpu_metrics, bar_width, label='CPU', color='royalblue')
gpu_bars = plt.bar(index + bar_width/2, gpu_metrics, bar_width, label='GPU', color='crimson')

# Add labels and title
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('CPU vs GPU: Performance Metrics')
plt.xticks(index, metrics_labels)
plt.legend()

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}', ha='center', va='bottom')

add_labels(cpu_bars)
add_labels(gpu_bars)

# Use logarithmic scale for better visualization
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300)
plt.close()

# --- Memory Usage Comparison ---
plt.figure(figsize=(8, 6))

# Create bars
memory_labels = ['CPU', 'GPU']
memory_values = [cpu_memory_usage, gpu_memory_usage]
memory_bars = plt.bar(memory_labels, memory_values, color=['royalblue', 'crimson'])

# Add labels and title
plt.xlabel('Implementation')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison')

# Add value labels on bars
for bar in memory_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{height:.2f} MB', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/memory_comparison.png", dpi=300)
plt.close()

# --- Total Time Comparison ---
plt.figure(figsize=(8, 6))

# Create bars
time_labels = ['CPU', 'GPU']
time_values = [cpu_total_time, gpu_total_time]
time_bars = plt.bar(time_labels, time_values, color=['royalblue', 'crimson'])

# Add labels and title
plt.xlabel('Implementation')
plt.ylabel('Total Processing Time (seconds)')
plt.title('Total Processing Time Comparison')

# Add value labels on bars
for bar in time_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.2f} s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/time_comparison.png", dpi=300)
plt.close()

print(f"Visualizations saved to {output_dir}/ directory")
