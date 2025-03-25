import matplotlib.pyplot as plt

# Performance metrics (in seconds)
data_loading_time = 125.16
feature_extraction_time = 3.05
knn_classification_time = 0.21
total_time = 126.69

# Calculated percentages (given in output, but we can also compute them)
phase_times = [data_loading_time, feature_extraction_time, knn_classification_time]
phase_labels = ['Data Loading', 'Feature Extraction', 'KNN Classification']
phase_percentages = [98.8, 2.4, 0.2]

# --- Pie Chart: Percentage of time per phase ---
plt.figure(figsize=(6,6))
plt.pie(phase_percentages, labels=phase_labels, autopct='%1.1f%%', startangle=140)
plt.title("Percentage of Processing Time per Phase")
plt.show()

# --- Bar Chart: Raw time per phase ---
plt.figure(figsize=(8,6))
bars = plt.bar(phase_labels, phase_times, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel("Time (seconds)")
plt.title("Processing Time for Each Phase")
for bar, time in zip(bars, phase_times):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{time:.2f}', ha='center', va='bottom')
plt.show()

# --- Additional Metrics ---
# Classification Accuracy and Processing Speed could be visualized too.
accuracy = 62.81  # in percent
processing_speed = 255.84  # images per second
peak_gpu_memory_usage = 4502.64  # in MB

metrics_labels = ['Classification Accuracy (%)', 'Processing Speed (img/s)', 'Peak GPU Memory (MB)']
metrics_values = [accuracy, processing_speed, peak_gpu_memory_usage]

plt.figure(figsize=(8,6))
bars = plt.bar(metrics_labels, metrics_values, color=['orchid', 'gold', 'lightcoral'])
for bar, value in zip(bars, metrics_values):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(metrics_values)*0.02, f'{value:.2f}', ha='center', va='bottom')
plt.title("Additional Performance Metrics")
plt.ylabel("Value")
plt.show()
