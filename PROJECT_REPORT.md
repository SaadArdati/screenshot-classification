# Screenshot Classification Project Report

## Executive Summary

This project implements a machine learning solution for automatically classifying images as screenshots or regular photographs. We developed a CPU-based implementation using the K-Nearest Neighbors (KNN) algorithm with grayscale histogram features.

Key achievements:
- Successfully trained a model on 25,932 images with 93.69% accuracy on a test set of 6,481 images
- Developed a memory-efficient implementation requiring only 3.10 MB during training
- Achieved fast inference time of under 2 milliseconds per image prediction
- Implemented an intuitive two-phase workflow: a training component that builds the model and a prediction component for classifying new images

Our performance analysis revealed that the image loading and feature extraction phase is the main computational bottleneck, accounting for 96% of the total training time. The classification algorithm itself is highly efficient, requiring only 0.21 milliseconds for inference against nearly 26,000 training examples.

## Problem Definition and AI Application Description

This project implements a machine learning-based image classifier that can distinguish between screenshots and regular photographs. The classification task is important for various applications including content moderation, image organization, and automated image analysis. Our approach uses K-Nearest Neighbors (KNN) with grayscale histogram features extracted from images.

## Flowchart for CPU Implementation

### Training Process:
Step 1: Load Training Images → 
Step 2: Extract Features from Each Image → 
Step 3: Save Model to File → 
Step 4: Load Test Images → 
Step 5: Extract Features from Test Images → 
Step 6: Evaluate Model Accuracy → 
Step 7: Print Results and Performance Metrics

### Classification Process (Prediction):
Step 1: Load Model from File → 
Step 2: Extract Features from Query Image → 
Step 3: Calculate Distance to All Training Samples → 
Step 4: Find K Nearest Neighbors → 
Step 5: Majority Vote for Classification → 
Step 6: Return Result (Screenshot/Non-Screenshot)

## Algorithm/Pseudocode for CPU Implementation

### Feature Extraction Algorithm

```
FUNCTION extract_feature(image_path, feature):
    img = load_image(image_path)
    IF img IS NULL THEN
        RETURN error
    END IF
    
    total_pixels = width * height
    Initialize histogram_bins[16] to zeros
    
    FOR each pixel in img:
        gray_value = (red + green + blue) / 3
        bin_index = gray_value * 16 / 256
        histogram_bins[bin_index]++
    END FOR
    
    FOR i = 0 TO 15:
        feature.bins[i] = histogram_bins[i] / total_pixels
    END FOR
    
    RETURN success
END FUNCTION
```

### K-Nearest Neighbors Classification Algorithm

```
FUNCTION classify_knn(training_set, training_size, query_feature):
    Initialize neighbors[training_size] array
    
    // Calculate distance to all training examples
    FOR i = 0 TO training_size - 1:
        neighbors[i].distance = euclidean_distance(training_set[i], query_feature)
        neighbors[i].label = training_set[i].label
    END FOR
    
    // Find K nearest neighbors
    Sort neighbors by distance (ascending)
    
    // Majority vote
    votes = 0
    FOR i = 0 TO K_NEIGHBORS - 1:
        votes += neighbors[i].label
    END FOR
    
    IF votes >= (K_NEIGHBORS / 2 + 1) THEN
        RETURN 1 (Screenshot)
    ELSE
        RETURN 0 (Non-screenshot)
    END IF
END FUNCTION
```

### Training Process Algorithm

```
FUNCTION train_model():
    // Load training data
    train_data = empty array
    train_count = 0
    
    // Load screenshot training images
    load_folder("screenshots_train_dir", label=1, train_data, train_count)
    
    // Load non-screenshot training images
    load_folder("non_screenshots_train_dir", label=0, train_data, train_count)
    
    // Load test data
    test_data = empty array
    test_count = 0
    
    // Load screenshot test images
    load_folder("screenshots_test_dir", label=1, test_data, test_count)
    
    // Load non-screenshot test images
    load_folder("non_screenshots_test_dir", label=0, test_data, test_count)
    
    // Evaluate model
    correct = 0
    FOR i = 0 TO test_count - 1:
        prediction = classify_knn(train_data, train_count, test_data[i])
        IF prediction == test_data[i].label THEN
            correct++
        END IF
    END FOR
    
    accuracy = correct / test_count
    
    // Save model to file
    save_model(model_path, train_data, train_count)
    
    RETURN accuracy
END FUNCTION
```

## Optimization Strategies & Efficiency

The sequential CPU implementation focuses on several optimization strategies:

1. **Memory Efficiency**: Using a 16-bin histogram as a compact feature representation instead of raw pixel data, reducing memory needs by several orders of magnitude.

2. **Efficient Distance Calculation**: The Euclidean distance calculation focuses only on the 16 histogram bins rather than raw pixel data, significantly reducing computation.

3. **Partial Sorting**: Instead of sorting the entire list of distances, we perform a partial sort to find just the K nearest neighbors, reducing complexity from O(n log n) to O(n K).

4. **Normalized Histograms**: Features are normalized by dividing by the total number of pixels, ensuring invariance to image size.

5. **Binary Format Model Storage**: The model is saved in an efficient binary format rather than text-based formats for faster loading and smaller file size.

## Performance Results

### CPU Implementation Performance

The sequential CPU implementation was tested on a dataset of approximately 26,000 training images and 6,500 test images. The performance metrics are as follows:

1. **Training Phase**:
   - Data Loading Time: 14.18 seconds
   - Model Evaluation Time: 0.45 seconds
   - Model Saving Time: <0.01 seconds
   - Total Processing Time: 14.63 seconds
   - Approximate Memory Usage: 3.10 MB

2. **Prediction Phase**:
   - Model Loading Time: 0.00049 seconds
   - Feature Extraction Time: 0.00127 seconds
   - Classification Time: 0.00021 seconds
   - Total Processing Time: 0.00197 seconds
   - Memory Usage (Model): 1.68 MB

3. **Classification Performance**:
   - Accuracy: 93.69% on the test set

### Critical Performance Bottlenecks

1. **Feature Extraction**: Loading and processing images is the most time-consuming operation, especially when dealing with thousands of images. Our measurements show that data loading accounts for over 96% of the total training time.

2. **KNN Classification**: The classification time grows linearly with the number of training examples, making it inefficient for very large training sets. However, our implementation shows good performance, with only 0.21 milliseconds needed for classification against 25,932 training examples.

3. **Memory Usage**: The entire training set must be loaded into memory, limiting scalability for extremely large datasets. Our model requires about 1.68 MB for storing the trained model, which is very reasonable for modern systems.

## Insights and Conclusions

The sequential CPU implementation demonstrates that grayscale histogram features are effective for distinguishing between screenshots and regular photographs, achieving high accuracy (93.69%). 

The simplicity of the KNN algorithm makes it interpretable and straightforward to implement, but it faces scalability challenges with large datasets due to its need to compare against all training samples.

Our performance analysis indicates that:
1. The feature extraction process is the main bottleneck in training, taking the majority of the processing time.
2. The model is quite memory-efficient, using only 3.10 MB during training and 1.68 MB for storing the model.
3. Inference (prediction) is very fast, taking less than 2 milliseconds per image.

Future improvements could focus on:
1. Implementing feature indexing structures like KD-trees to accelerate nearest neighbor searches
2. Exploring more compact feature representations
3. Developing batch processing capabilities to handle larger datasets

## Project Management

### Team Members and Roles

| Name          | Role         | Responsibilities                    |
|---------------|--------------|-------------------------------------|
| Saad Ardati   | Project Lead | C implementation, project organizer |
| Ryan Kyrillos | Developer    | CUDA implementation, Analysis       |
| Tarek Zaher   | Developer    | Dataset preparation, Analysis       |

### Project Timeline

| Week                      | Tasks                                        | Status      |
|---------------------------|----------------------------------------------|-------------|
| Week 1 (March 1-7)        | Project planning, requirement analysis       | Completed   |
| Week 2 (March 8-14)       | Dataset preparation, feature design          | Completed   |
| Week 3 (March 15-21)      | CPU implementation development               | Completed   |
| Week 4 (March 22-28)      | Testing and optimization                     | Completed   |
| Week 5 (March 29-April 4) | Performance measurement and analysis         | Completed   |
| Week 6 (April 5-11)       | Documentation and report writing             | Completed   |
| Week 7 (April 12-18)      | Final testing and presentation preparation   | In Progress |
| Week 8 (April 19-25)      | Presentation rehearsal and final adjustments | Planned     |
| April 30, 2025            | Final submission deadline                    | Deadline    |

## Appendix: Source Code

The complete source code for the project can be found in the GitHub repository: [Project Repository URL]

Key files include:
- `training.c`: Implements the training procedure with feature extraction and model evaluation
- `predict.c`: Implements the prediction functionality using the trained model
- `stb_image.h`: External library for image loading and processing 