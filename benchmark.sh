#!/bin/bash

MODE=$1
ITERATIONS=5
MODEL_FILE="model_${MODE}.bin"

# Function to run training and capture time
run_training() {
    echo "Training $MODE model..."
    echo "Running command: ./build/bin/train_seq $MODEL_FILE"
    
    # Direct debugging output
    echo "DEBUGGING: Listing executable"
    ls -la ./build/bin/train_*
    
    # Run with verbose output
    if [ "$MODE" == "sequential" ]; then
        ./build/bin/train_seq $MODEL_FILE | tee train_${MODE}_output.txt
        RET=${PIPESTATUS[0]}
        if [ $RET -ne 0 ]; then
            echo "ERROR: Training executable failed with code $RET"
        fi
    else
        ./build/bin/train_cuda $MODEL_FILE | tee train_${MODE}_output.txt
        RET=${PIPESTATUS[0]}
        if [ $RET -ne 0 ]; then
            echo "ERROR: Training executable failed with code $RET"
        fi
    fi
    
    # Extract and save metrics
    echo "Extracting training metrics..."
    
    # Create default values in case extraction fails
    echo "0.0" > train_${MODE}_total_time.txt
    echo "0.0" > train_${MODE}_loading_time.txt
    echo "0.0" > train_${MODE}_memory.txt
    echo "0.0" > train_${MODE}_accuracy.txt
    echo "0.0" > train_${MODE}_images_per_sec.txt
    
    # Extract times from output if they exist
    if grep -q "Total Processing Time" train_${MODE}_output.txt; then
        grep "Total Processing Time" train_${MODE}_output.txt | awk '{print $4}' > train_${MODE}_total_time.txt
        echo "Found total time: $(cat train_${MODE}_total_time.txt)"
    else
        echo "WARNING: Could not find 'Total Processing Time' in output"
    fi
    
    if grep -q "Data Loading Time" train_${MODE}_output.txt; then
        grep "Data Loading Time" train_${MODE}_output.txt | awk '{print $4}' > train_${MODE}_loading_time.txt
        echo "Found loading time: $(cat train_${MODE}_loading_time.txt)"
    else
        echo "WARNING: Could not find 'Data Loading Time' in output"
    fi
    
    if grep -q "Memory Usage" train_${MODE}_output.txt; then
        grep "Memory Usage" train_${MODE}_output.txt | awk '{print $3}' > train_${MODE}_memory.txt
        echo "Found memory usage: $(cat train_${MODE}_memory.txt)"
    else
        echo "WARNING: Could not find 'Memory Usage' in output"
    fi
    
    if grep -q "Accuracy" train_${MODE}_output.txt; then
        grep "Accuracy" train_${MODE}_output.txt | awk '{print $2}' > train_${MODE}_accuracy.txt
        echo "Found accuracy: $(cat train_${MODE}_accuracy.txt)"
    else
        echo "WARNING: Could not find 'Accuracy' in output"
    fi
    
    # Calculate processing speed only if we have the necessary data
    if grep -q "Number of training examples" train_${MODE}_output.txt && [ -s train_${MODE}_total_time.txt ]; then
        TOTAL_EXAMPLES=$(grep "Number of training examples" train_${MODE}_output.txt | awk '{print $5}')
        TOTAL_TIME=$(cat train_${MODE}_total_time.txt)
        if [ ! -z "$TOTAL_EXAMPLES" ] && [ ! -z "$TOTAL_TIME" ] && [ "$TOTAL_TIME" != "0.0" ]; then
            echo "scale=2; $TOTAL_EXAMPLES / $TOTAL_TIME" | bc > train_${MODE}_images_per_sec.txt
            echo "Calculated images per second: $(cat train_${MODE}_images_per_sec.txt)"
        else
            echo "WARNING: Could not calculate images per second (examples: $TOTAL_EXAMPLES, time: $TOTAL_TIME)"
        fi
    else
        echo "WARNING: Missing data for calculating images per second"
    fi
    
    echo "Training metrics extraction completed"
}

# Function to run prediction and capture time
run_prediction() {
    echo "Running prediction benchmarks for $MODE implementation..."
    
    # Create empty files for metrics
    rm -f predict_${MODE}_total_time.txt
    rm -f predict_${MODE}_feature_time.txt
    rm -f predict_${MODE}_knn_time.txt
    rm -f predict_${MODE}_memory.txt
    
    # Get sample images for testing
    SCREENSHOT_IMG=$(find data/screenshots/test -type f | head -n 1)
    NON_SCREENSHOT_IMG=$(find data/non_screenshots/test -type f | head -n 1)
    
    echo "Using screenshot test image: $SCREENSHOT_IMG"
    echo "Using non-screenshot test image: $NON_SCREENSHOT_IMG"
    
    # Direct debugging output
    echo "DEBUGGING: Listing executable"
    ls -la ./build/bin/predict_*
    
    # Test screenshot
    echo "Testing screenshot image..."
    for i in $(seq 1 $ITERATIONS); do
        echo "Running iteration $i..."
        
        if [ "$MODE" == "sequential" ]; then
            echo "COMMAND: ./build/bin/predict_seq $MODEL_FILE \"$SCREENSHOT_IMG\""
            ./build/bin/predict_seq $MODEL_FILE "$SCREENSHOT_IMG" | tee predict_${MODE}_output_$i.txt
            RET=${PIPESTATUS[0]}
            if [ $RET -ne 0 ]; then
                echo "ERROR: Prediction executable failed with code $RET"
                continue
            fi
        else
            echo "COMMAND: ./build/bin/predict_cuda $MODEL_FILE \"$SCREENSHOT_IMG\""
            ./build/bin/predict_cuda $MODEL_FILE "$SCREENSHOT_IMG" | tee predict_${MODE}_output_$i.txt
            RET=${PIPESTATUS[0]}
            if [ $RET -ne 0 ]; then
                echo "ERROR: Prediction executable failed with code $RET"
                continue
            fi
        fi
        
        # Set default values in case metrics are not found
        echo "0.0" >> predict_${MODE}_total_time.txt.tmp
        echo "0.0" >> predict_${MODE}_feature_time.txt.tmp
        echo "0.0" >> predict_${MODE}_knn_time.txt.tmp
        echo "0.0" >> predict_${MODE}_memory.txt.tmp
        
        # Extract detailed metrics if they exist
        if grep -q "Total Processing Time" predict_${MODE}_output_$i.txt; then
            grep "Total Processing Time" predict_${MODE}_output_$i.txt | awk '{print $4}' > predict_${MODE}_total_time.txt.tmp
            echo "Found total time: $(cat predict_${MODE}_total_time.txt.tmp)"
            cat predict_${MODE}_total_time.txt.tmp >> predict_${MODE}_total_time.txt
        else
            echo "WARNING: Could not find 'Total Processing Time' in output"
            echo "0.0" >> predict_${MODE}_total_time.txt
        fi
        
        if grep -q "Feature Extraction Time" predict_${MODE}_output_$i.txt; then
            grep "Feature Extraction Time" predict_${MODE}_output_$i.txt | awk '{print $4}' > predict_${MODE}_feature_time.txt.tmp
            echo "Found feature time: $(cat predict_${MODE}_feature_time.txt.tmp)"
            cat predict_${MODE}_feature_time.txt.tmp >> predict_${MODE}_feature_time.txt
        else
            echo "WARNING: Could not find 'Feature Extraction Time' in output"
            echo "0.0" >> predict_${MODE}_feature_time.txt
        fi
        
        if grep -q "Classification Time" predict_${MODE}_output_$i.txt; then
            grep "Classification Time" predict_${MODE}_output_$i.txt | awk '{print $3}' > predict_${MODE}_knn_time.txt.tmp
            echo "Found KNN time: $(cat predict_${MODE}_knn_time.txt.tmp)"
            cat predict_${MODE}_knn_time.txt.tmp >> predict_${MODE}_knn_time.txt
        else
            echo "WARNING: Could not find 'Classification Time' in output"
            echo "0.0" >> predict_${MODE}_knn_time.txt
        fi
        
        if grep -q "Memory Usage" predict_${MODE}_output_$i.txt; then
            grep "Memory Usage" predict_${MODE}_output_$i.txt | awk '{print $3}' > predict_${MODE}_memory.txt.tmp
            echo "Found memory usage: $(cat predict_${MODE}_memory.txt.tmp)"
            cat predict_${MODE}_memory.txt.tmp >> predict_${MODE}_memory.txt
        else
            echo "WARNING: Could not find 'Memory Usage' in output"
            echo "0.0" >> predict_${MODE}_memory.txt
        fi
        
        # Clean up temporary files
        rm -f predict_${MODE}_total_time.txt.tmp
        rm -f predict_${MODE}_feature_time.txt.tmp
        rm -f predict_${MODE}_knn_time.txt.tmp
        rm -f predict_${MODE}_memory.txt.tmp
    done
    
    # Calculate average prediction metrics
    echo "Calculating average metrics..."
    if [ -s predict_${MODE}_total_time.txt ]; then
        echo "scale=6; $(cat predict_${MODE}_total_time.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_total_time.txt
        echo "Average prediction time: $(cat avg_predict_${MODE}_total_time.txt) seconds"
    else
        echo "0.0" > avg_predict_${MODE}_total_time.txt
        echo "No valid data for total prediction time"
    fi
    
    if [ -s predict_${MODE}_feature_time.txt ]; then
        echo "scale=6; $(cat predict_${MODE}_feature_time.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_feature_time.txt
        echo "Average feature extraction time: $(cat avg_predict_${MODE}_feature_time.txt) seconds"
    else
        echo "0.0" > avg_predict_${MODE}_feature_time.txt
        echo "No valid data for feature extraction time"
    fi
    
    if [ -s predict_${MODE}_knn_time.txt ]; then
        echo "scale=6; $(cat predict_${MODE}_knn_time.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_knn_time.txt
        echo "Average KNN computation time: $(cat avg_predict_${MODE}_knn_time.txt) seconds"
    else
        echo "0.0" > avg_predict_${MODE}_knn_time.txt
        echo "No valid data for KNN computation time"
    fi
    
    if [ -s predict_${MODE}_memory.txt ]; then
        echo "scale=2; $(cat predict_${MODE}_memory.txt | paste -sd+ | bc) / $ITERATIONS" | bc > avg_predict_${MODE}_memory.txt
        echo "Average memory usage: $(cat avg_predict_${MODE}_memory.txt) MB"
    else
        echo "0.0" > avg_predict_${MODE}_memory.txt
        echo "No valid data for memory usage"
    fi
}

# Main benchmark
echo "Starting $MODE benchmarks..."
run_training
run_prediction
echo "$MODE benchmarking complete." 