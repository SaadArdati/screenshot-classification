# Screenshot Classification

A machine learning-based image classifier that can identify whether an image is a screenshot or a regular photograph.

## Overview

This project implements a K-nearest neighbors (KNN) classification algorithm to distinguish between screenshots and non-screenshots using grayscale histogram features extracted from images.

## Requirements

- C compiler (gcc recommended)
- make or CMake
- Image dataset split into screenshots and non-screenshots

## Installation

Clone the repository and build the project using one of the following methods:

### Using Make

```bash
make all
```

### Using CMake

```bash
mkdir -p cmake-build-debug
cd cmake-build-debug
cmake ..
make
```

This will compile three executables:
- `classifier`: For classifying individual images
- `training`: For training the model on the dataset
- `predict`: For using the trained model to classify new images

## Data Structure

The project expects data to be organized as follows:

```
split_data/
├── screenshots_256x256/
│   ├── train/
│   │   └── [screenshot images]
│   └── test/
│       └── [screenshot images]
└── non_screenshot_256x256/
    ├── train/
    │   └── [non-screenshot images]
    └── test/
        └── [non-screenshot images]
```

## Training the Model

To train the model on your dataset:

```bash
./training
```

This will:
1. Load all training images from both screenshot and non-screenshot directories
2. Extract grayscale histogram features from each image
3. Train a K-Nearest Neighbors model (with K=3)
4. Evaluate the model on the test set
5. Output the accuracy
6. Save the trained model to `trained_model.bin`

You can also specify a custom output path for the model:

```bash
./training my_custom_model.bin
```

## Using the Model to Classify Images

Once you have a trained model, you can classify new images using:

```bash
./predict trained_model.bin path/to/your/image.jpg
```

The program will output whether the image is classified as a SCREENSHOT or NON-SCREENSHOT.

### Examples

Classify a screenshot:
```bash
./predict trained_model.bin split_data/screenshots_256x256/test/GAME_530.jpg
```

Classify a regular photograph:
```bash
./predict trained_model.bin split_data/non_screenshot_256x256/test/photo-1634291934402-7968f44a2939.jpg
```

## Technical Details

### Feature Extraction
- Images are converted to grayscale
- A histogram with 16 bins is computed
- Histogram is normalized by the total number of pixels

### Classification Algorithm
- K-Nearest Neighbors (K=3)
- Euclidean distance metric
- Majority vote for final classification

### Model File Format
The model is saved as a binary file containing:
- Number of training examples (integer)
- Array of Feature structures, each containing:
  - 16-bin histogram (float array)
  - Label (1 for screenshot, 0 for non-screenshot)

## Performance

On the test set, the model achieves approximately 93.69% accuracy, demonstrating that grayscale histogram features are effective for distinguishing between screenshots and regular photographs.

## Limitations

- The model only considers grayscale information
- Performance may vary with different types of screenshots or photographs
- The current implementation loads all training data into memory, which may be limiting for very large datasets 