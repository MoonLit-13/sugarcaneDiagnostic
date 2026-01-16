# Sugarcane Leaf Disease Classification

This project implements a lightweight Convolutional Neural Network (CNN) model for classifying sugarcane leaf images into six disease categories: Healthy, Bacterial Blight, Mosaic, RedRot, Rust, and Yellow.

## Overview

The Sugarcane_Health_Diagnostic_ML.ipynb provides a complete pipeline for:
- Downloading the dataset from Kaggle
- Data preprocessing and augmentation
- Building and training a CNN model
- Model evaluation
- Converting the model to TensorFlow Lite (TFLite) for deployment
- Testing inference with the TFLite model

## Dataset

The dataset used is the [Sugarcane Plant Diseases Dataset](https://www.kaggle.com/datasets/akilesh253/sugarcane-plant-diseases-dataset) from Kaggle, containing images of sugarcane leaves categorized into the six classes mentioned above.

## Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenDatasets (for Kaggle dataset download)

## Installation

1. Install the required packages:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opendatasets
   ```

2. Set up Kaggle API credentials (if downloading dataset):
   - Obtain your Kaggle username and API key from your Kaggle account settings.
   - The notebook will prompt for these during execution.

## Usage

1. Open the [Sugarcane_Health_Diagnostic_ML.ipynb]() notebook in Jupyter.

2. Run the cells in order:
   - Import libraries
   - Download dataset (requires Kaggle credentials)
   - Data loading and preprocessing
   - Model building
   - Training
   - Evaluation
   - Model conversion to TFLite
   - Testing inference

3. For testing with a pre-trained model:
   - Skip the training section
   - Download the pre-trained model as instructed in the notebook
   - Proceed to the testing sections

## Model Architecture

The model is a lightweight CNN built using Keras, potentially utilizing transfer learning with MobileNetV2 for better performance.

## Evaluation

The notebook includes evaluation metrics such as classification report and confusion matrix to assess model performance.

## Deployment

The trained model is converted to TensorFlow Lite format for efficient deployment on edge devices.

## Testing

The notebook provides two testing modes:
- Batch testing with multiple images
- Single image upload and classification

## Files
- `requirements.txt`: Python dependencies for training
- `sugarcane_classification.ipynb`: Jupyter notebook for data loading, model training, evaluation, and TFLite conversion
- `sugarcaneDiagnostic.ipynb`: Notebook for testing TFLite inference locally
- `rasp_diagnostics.py`: Python script for Raspberry Pi deployment with camera
- `sugarcane_model.tflite`: Converted TensorFlow Lite model for inference
- `Sugarcane_leafs/`: Dataset directory with subfolders for each class

---
- Dataset source: Kaggle (akilesh253/sugarcane-plant-diseases-dataset)
- TensorFlow and Keras for deep learning framework</content>

