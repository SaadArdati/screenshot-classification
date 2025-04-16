#!/bin/bash

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