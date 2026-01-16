
Objective:
Train a lightweight machine learning model to classify sugarcane leaf images into six categories:
- Healthy
- Bacterial Blight
- Mosaic
- RedRot
- Rust
- Yellow

Dataset:
Use the labeled sugarcane leaf dataset from Kaggle. Ensure images are preprocessed:
- Data set source: https://www.kaggle.com/datasets/akilesh253/sugarcane-plant-diseases-dataset
- Resize to 224x224 pixels
- Normalize pixel values (0–1)
- Apply augmentation (rotation, flip, zoom) for robustness

Model Requirements:
- Lightweight CNN architecture (MobileNetV2 or EfficientNet-lite preferred)
- Balance between training speed and accuracy
- Prioritize accuracy over inference speed
- Train entirely on laptop with 15.6 GB RAM (6.3 GB available)

Training Setup:
- Use Python 3.11.9ironment
- Framework: TensorFlow/Keras
- Batch size: 32
- Epochs: 20–30 with early stopping
- Optimizer: Adam
- Loss: categorical crossentropy
- Metrics: accuracy

Deployment:
- Convert trained model to TensorFlow Lite format
- Apply quantization for smaller size and faster inference
- Deploy on Raspberry Pi with AI Hat+ and camera sensor module
- Framework: TensorFlow Lite Runtime (easier setup)
- Inference pipeline: capture image → preprocess → run model → output diagnosis

Performance Expectations:
- Real-time inference not required; short delay acceptable
- Accuracy is priority

## Environment Setup

1. Ensure Python 3.11.9 is installed.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Launch Jupyter: `jupyter notebook`
6. Open `sugarcane_classification.ipynb` and run the cells.

## Files
- `requirements.txt`: Python dependencies for training
- `sugarcane_classification.ipynb`: Jupyter notebook for data loading, model training, evaluation, and TFLite conversion
- `sugarcaneDiagnostic.ipynb`: Notebook for testing TFLite inference locally
- `rasp_diagnostics.py`: Python script for Raspberry Pi deployment with camera
- `sugarcane_model.tflite`: Converted TensorFlow Lite model for inference
- `Sugarcane_leafs/`: Dataset directory with subfolders for each class

## Raspberry Pi Deployment
1. Transfer `sugarcane_model.tflite` and `rasp_diagnostics.py` to Raspberry Pi.
2. Install dependencies: `pip3 install picamera numpy pillow tflite-runtime`
3. Enable camera: `sudo raspi-config` → Interfacing Options → Camera → Enable
4. Run: `python3 rasp_diagnostics.py`
   - Captures image, preprocesses, runs inference, prints diagnosis.

For AI Hat+ integration, modify the script to use the hat's camera API if different from picamera.

## Files
- `requirements.txt`: Python dependencies
- `sugarcane_classification.ipynb`: Main notebook for training and evaluation
- `Sugarcane_leafs/`: Dataset directory

