# Space Anomaly Detection System

A machine learning system for detecting anomalies in space imagery using convolutional autoencoders. This system analyzes astronomical images to identify unusual patterns and objects with high confidence.

## 🎯 Project Goal

The goal of this project is to recognize space anomalies through analysis of space imagery using machine learning techniques. The system provides:

- **Input**: Space imagery (SDSS, NASA, etc.)
- **Output**: 
  - List of images with potential anomalies (≥80% confidence)
  - List of identified anomalies with metadata

## 🏗️ System Architecture

### Core Components

1. **Data Preprocessor** (`DataPreprocessor`)
   - Loads and preprocesses space imagery
   - Resizes images to 512x512 pixels
   - Converts to grayscale and normalizes
   - Splits data into training/testing sets

2. **Convolutional Autoencoder** (`ConvAutoencoder`)
   - Encoder: 3 convolutional layers with max pooling
   - Decoder: 3 transpose convolutional layers
   - Learns normal patterns in space imagery

3. **Anomaly Detector** (`AnomalyDetector`)
   - Trains autoencoder on normal images
   - Detects anomalies based on reconstruction error
   - Calculates confidence scores
   - Exports anomaly images with metadata

4. **Complete System** (`SpaceAnomalyDetectionSystem`)
   - Orchestrates the entire pipeline
   - Handles data preprocessing, training, and detection
   - Provides comprehensive results

## 🚀 Quick Start

### Option 1: GUI (Recommended for End Users)
```bash
python launch_gui.py
```

### Option 2: Command Line

#### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

#### Basic Usage
```python
from space_anomaly_detector import SpaceAnomalyDetectionSystem

# Initialize the system
system = SpaceAnomalyDetectionSystem(data_dir="sdss_images")

# Run complete pipeline
results = system.run_complete_pipeline(
    confidence_threshold=0.8,  # 80% confidence threshold
    train_epochs=50
)

print(f"Found {results['high_confidence_anomalies']} high-confidence anomalies")
```

#### Command Line Usage
```bash
# Run the complete system
python space_anomaly_detector.py

# Run tests
python test_system.py

# Use pre-trained model
python use_pretrained_model.py
```

## 🖥️ GUI Application

The system includes a comprehensive GUI for easy interaction:

### Features
- **Main Tab**: System status and quick actions
- **Training Tab**: Train models with visual progress
- **Detection Tab**: Detect anomalies with adjustable parameters
- **Results Tab**: View and browse detected anomalies
- **Settings Tab**: Configure system settings and view logs

### Launch Options
```bash
# Recommended: Use launcher with dependency checking
python launch_gui.py

# Direct launch
python gui_app.py

# Demo mode
python demo_gui.py
```

For detailed GUI documentation, see [README_GUI.md](README_GUI.md).

## 📁 Project Structure

```
├── space_anomaly_detector.py    # Main system implementation
├── gui_app.py                  # GUI application
├── launch_gui.py               # GUI launcher
├── demo_gui.py                 # GUI demo
├── config.py                   # Configuration parameters
├── test_system.py              # Testing and validation
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── README_GUI.md              # GUI documentation
├── sdss_images/               # Input space imagery
├── preprocessed_data/         # Preprocessed datasets
├── models/                    # Trained models
├── anomalies_export/          # Exported anomaly images
└── autoencoder.ipynb         # Original notebook implementation
```

## ⚙️ Configuration

Edit `config.py` to customize system parameters:

```python
# Data Configuration
DATA_CONFIG = {
    'data_directory': 'sdss_images',
    'image_size': (512, 512),
    'grayscale': True,
    'test_ratio': 0.2
}

# Detection Configuration
DETECTION_CONFIG = {
    'confidence_threshold': 0.8,  # 80% confidence
    'error_percentile': 95        # Anomaly threshold
}
```

## 🔧 System Features

### Data Preprocessing
- **Image Loading**: Supports JPG, PNG formats
- **Resizing**: Standardizes to 512x512 pixels
- **Normalization**: Scales pixel values to [0,1]
- **Grayscale Conversion**: Reduces complexity for better training

### Model Architecture
- **Encoder**: 3 convolutional layers (1→32→64→128 channels)
- **Decoder**: 3 transpose convolutional layers (128→64→32→1 channels)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Loss**: Mean Squared Error for reconstruction quality

### Anomaly Detection
- **Training**: Learns normal patterns from space imagery
- **Inference**: High reconstruction error indicates anomalies
- **Confidence Scoring**: Normalized confidence based on error distribution
- **Thresholding**: Dynamic threshold based on error percentile

### Output Generation
- **Anomaly Images**: Exported with error and confidence metadata
- **Results JSON**: Comprehensive detection results
- **Visualizations**: Original vs reconstructed image comparisons
- **Metadata**: Error scores, confidence levels, and thresholds

## 📊 Results Format

The system generates comprehensive results:

```python
{
    'high_confidence_anomalies': 3,           # Number of anomalies ≥80% confidence
    'total_anomalies': 5,                     # Total anomalies detected
    'total_images_processed': 18,             # Total test images
    'confidence_threshold': 0.8,              # Applied confidence threshold
    'exported_files': ['anomaly_0003_err0.01244_conf0.850.png', ...],
    'anomaly_indices': [3, 7, 12],           # Indices of high-confidence anomalies
    'error_threshold': 0.005441               # Calculated error threshold
}
```

## 🧪 Testing

Run comprehensive tests:

```bash
python test_system.py
```

Tests include:
- ✅ Data preprocessing validation
- ✅ Model training verification
- ✅ Anomaly detection accuracy
- ✅ Complete pipeline testing
- 📊 Performance visualizations

## 📈 Performance Metrics

- **Reconstruction Loss**: Tracks training progress
- **Anomaly Detection Rate**: Percentage of anomalies correctly identified
- **Confidence Distribution**: Spread of confidence scores
- **Error Threshold**: Dynamic threshold based on data distribution

## 🔍 Example Output

```
==================================================
SPACE ANOMALY DETECTION RESULTS
==================================================
Total images processed: 18
High-confidence anomalies: 3
Total anomalies detected: 5
Confidence threshold: 0.8
Error threshold: 0.005441
Exported files: 3
==================================================
```

## 🛠️ Advanced Usage

### Custom Model Training

```python
from space_anomaly_detector import AnomalyDetector
import numpy as np

# Load your data
X_train = np.load("preprocessed_data/X_train.npy")

# Initialize detector
detector = AnomalyDetector()

# Train with custom parameters
losses = detector.train(
    X_train,
    epochs=100,
    batch_size=64,
    learning_rate=1e-4
)
```

### Batch Processing

```python
# Process multiple directories
directories = ["sdss_images", "nasa_images", "custom_data"]

for data_dir in directories:
    system = SpaceAnomalyDetectionSystem(data_dir=data_dir)
    results = system.run_complete_pipeline()
    print(f"Results for {data_dir}: {results}")
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use CPU instead
   detector = AnomalyDetector(device="cpu")
   ```

2. **No Images Found**
   ```bash
   # Check data directory
   ls sdss_images/
   ```

3. **Low Anomaly Detection**
   ```python
   # Adjust confidence threshold
   results = system.run_complete_pipeline(confidence_threshold=0.7)
   ```

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualizations
- **scikit-learn**: Data splitting and utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Sloan Digital Sky Survey (SDSS) for space imagery
- PyTorch community for deep learning tools
- OpenCV for image processing capabilities

---

**Note**: This system is designed for research and educational purposes. For production use, consider additional validation and testing procedures. 