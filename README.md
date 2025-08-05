# Space Anomaly Detection & Classification System

A comprehensive machine learning system for detecting anomalies and classifying astronomical objects in space imagery.

## 🌟 Features

- **Anomaly Detection**: Identify unusual patterns in space imagery using convolutional autoencoders
- **Object Classification**: Classify astronomical objects (galaxies, stars, nebulae, etc.)
- **Combined Analysis**: Run both detection and classification in a unified pipeline
- **Progress Tracking**: Visual progress bars for long-running operations
- **Export Results**: Save detected anomalies and classifications with metadata
- **Command-Line Interface**: Easy-to-use CLI for all operations

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd space-anomaly-detector
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables** (macOS):
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

### Basic Usage

#### Test the System
```bash
python space_analyzer.py test
```

#### Run Anomaly Detection
```bash
python space_analyzer.py detect --input-dir test_dataset/images/ --output-dir results/
```

#### Run Object Classification
```bash
python space_analyzer.py classify --input-dir test_dataset/images/ --output-dir results/
```

#### Run Combined Analysis
```bash
python space_analyzer.py analyze --input-dir test_dataset/images/ --epochs 20
```

#### Train Models
```bash
python space_analyzer.py train --input-dir test_dataset/images/ --model both --epochs 50
```

## 📋 Command-Line Interface

The system provides a comprehensive command-line interface through `space_analyzer.py`:

### Available Commands

- **`test`**: Test system components and data availability
- **`detect`**: Run anomaly detection on images
- **`classify`**: Run astronomical object classification
- **`analyze`**: Run combined anomaly detection and classification
- **`train`**: Train anomaly detector and/or classifier models

### Common Options

- `--device`: Choose computation device (`cpu` or `cuda`)
- `--verbose`: Enable detailed logging
- `--input-dir`: Directory containing input images
- `--output-dir`: Directory for saving results
- `--epochs`: Number of training epochs
- `--threshold`: Confidence threshold for detections/classifications

### Examples

```bash
# Test the system
python space_analyzer.py test

# Detect anomalies with training
python space_analyzer.py detect --input-dir images/ --output-dir results/ --train --epochs 100

# Classify objects with custom threshold
python space_analyzer.py classify --input-dir images/ --output-dir results/ --threshold 0.9

# Run complete analysis
python space_analyzer.py analyze --input-dir images/ --output-dir results/ --epochs 50

# Train only the anomaly detector
python space_analyzer.py train --input-dir images/ --model anomaly --epochs 75
```

## 🏗️ System Architecture

### Core Components

1. **Anomaly Detector** (`space_anomaly_detector.py`)
   - Convolutional autoencoder for anomaly detection
   - Data preprocessing pipeline
   - Export functionality for results

2. **Astronomical Classifier** (`astronomical_classifier.py`)
   - CNN-based classifier for astronomical objects
   - Multi-class classification support
   - Confidence scoring

3. **Enhanced System** (`enhanced_space_system.py`)
   - Combined anomaly detection and classification
   - Unified analysis pipeline
   - Comprehensive reporting

4. **Command-Line Interface** (`space_analyzer.py`)
   - User-friendly CLI for all operations
   - Progress tracking and logging
   - Error handling and validation

### Data Flow

```
Input Images → Preprocessing → Model Inference → Results → Export
```

## 📊 Output Format

### Anomaly Detection Results
- **Anomaly images**: High-confidence anomaly detections
- **Metadata**: JSON files with detection details
- **Visualizations**: Plots showing anomaly scores

### Classification Results
- **Classified objects**: Images with classification labels
- **Metadata**: JSON files with classification details
- **Statistics**: Summary of object types found

### Combined Analysis
- **Comprehensive report**: Both anomaly and classification results
- **Cross-referenced data**: Objects that are both anomalous and classified
- **Export directories**: Organized results by analysis type

## 🔧 Configuration

The system uses `config.py` for centralized configuration:

- **Data paths**: Input/output directories
- **Model parameters**: Architecture settings
- **Detection thresholds**: Confidence levels
- **System settings**: Device selection, logging

## 📁 Directory Structure

```
space-anomaly-detector/
├── space_analyzer.py          # Main CLI interface
├── space_anomaly_detector.py  # Anomaly detection system
├── astronomical_classifier.py  # Object classification system
├── enhanced_space_system.py   # Combined analysis system
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── test_dataset/             # Test data
│   └── images/              # Sample images
├── models/                   # Trained models
├── results/                  # Output directory
├── enhanced_analysis_export/ # Combined analysis results
├── anomalies_export/         # Anomaly detection results
├── classification_export/    # Classification results
└── preprocessed_data/        # Preprocessed datasets
```

## 🧪 Testing

### System Tests
```bash
# Basic system test
python space_analyzer.py test
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenMP Errors** (macOS): Set environment variable
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

3. **CUDA Issues**: Use CPU if CUDA is not available
   ```bash
   python space_analyzer.py test --device cpu
   ```

4. **Memory Issues**: Reduce batch size or image resolution in config.py

### Getting Help

- Check the log file: `space_analyzer.log`
- Run with verbose mode: `--verbose`
- Test system components: `python space_analyzer.py test`

## 📈 Performance

- **Anomaly Detection**: ~0.1-0.5 seconds per image (CPU)
- **Object Classification**: ~0.2-0.8 seconds per image (CPU)
- **Combined Analysis**: ~0.3-1.3 seconds per image (CPU)
- **Training**: 5-30 minutes depending on dataset size and epochs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SDSS (Sloan Digital Sky Survey) for astronomical data
- PyTorch team for the deep learning framework
- Scientific Python community for supporting libraries 