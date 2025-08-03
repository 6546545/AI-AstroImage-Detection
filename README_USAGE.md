# Using the Pre-trained Space Anomaly Detection Model

This guide shows you how to use the trained model to detect anomalies in new space images.

## 🚀 Quick Start

### 1. Command Line Usage

The easiest way to use the model is with the command-line script:

```bash
# Analyze images in a directory
python use_pretrained_model.py path/to/your/images

# Use custom confidence threshold
python use_pretrained_model.py path/to/your/images --confidence 0.7

# Specify output directory
python use_pretrained_model.py path/to/your/images --output my_results

# Skip visualization
python use_pretrained_model.py path/to/your/images --no-viz
```

### 2. Run Examples

See the model in action with example usage:

```bash
python example_usage.py
```

## 📋 Requirements

- ✅ Trained model file: `models/anomaly_detector.pth`
- ✅ Python dependencies (see `requirements.txt`)
- ✅ Images to analyze (JPG, PNG format)

## 🔧 Usage Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `image_dir` | Directory containing images to analyze | Required |
| `--confidence, -c` | Confidence threshold (0.0-1.0) | 0.8 |
| `--output, -o` | Output directory for results | `new_anomalies` |
| `--no-viz` | Skip visualization generation | False |

### Example Commands

```bash
# Basic usage
python use_pretrained_model.py sdss_images

# More sensitive detection (lower threshold)
python use_pretrained_model.py sdss_images --confidence 0.6

# Less sensitive detection (higher threshold)
python use_pretrained_model.py sdss_images --confidence 0.9

# Custom output directory
python use_pretrained_model.py sdss_images --output my_anomalies

# Quick detection without visualization
python use_pretrained_model.py sdss_images --no-viz
```

## 📊 Understanding Results

### Output Files

The script creates several output files:

- **`detection_results.json`**: Detailed results in JSON format
- **`detection_visualization.png`**: Charts showing error and confidence distributions
- **Anomaly images**: Individual images of detected anomalies

### Result Interpretation

```json
{
  "total_images_processed": 43,
  "high_confidence_anomalies": 2,
  "total_anomalies": 3,
  "confidence_threshold": 0.8,
  "error_threshold": 0.011510,
  "anomaly_details": [
    {
      "index": 3,
      "filename": "ra30p0000_dec65p0000.jpg",
      "error": 0.014780,
      "confidence": 0.839,
      "is_high_confidence": true
    }
  ]
}
```

### Key Metrics

- **High-confidence anomalies**: Anomalies with confidence ≥ threshold (default 80%)
- **Total anomalies**: All anomalies above error threshold
- **Error threshold**: 95th percentile of reconstruction errors
- **Confidence**: Normalized score based on error magnitude

## 🐍 Programmatic Usage

### Basic Detection

```python
from space_anomaly_detector import AnomalyDetector, DataPreprocessor

# Initialize
detector = AnomalyDetector(model_path="models/anomaly_detector.pth")
preprocessor = DataPreprocessor()

# Load images
X_images, filenames = preprocessor.load_and_preprocess_images("path/to/images")

# Detect anomalies
results = detector.detect_anomalies(X_images, confidence_threshold=0.8)

# Check results
high_conf_anomalies = len(results['high_confidence_anomalies'])
total_anomalies = len(results['all_anomalies'])

print(f"Found {high_conf_anomalies} high-confidence anomalies")
print(f"Total anomalies: {total_anomalies}")
```

### Export Anomalies

```python
# Export anomaly images
exported_files = detector.export_anomalies(
    X_images, results, filenames, "output_directory"
)

print(f"Exported {len(exported_files)} anomaly images")
```

### Custom Analysis

```python
# Analyze individual images
for i, (image, filename) in enumerate(zip(X_images, filenames)):
    single_result = detector.detect_anomalies(image[np.newaxis, ...])
    
    is_anomaly = len(single_result['high_confidence_anomalies']) > 0
    error = single_result['errors'][0]
    confidence = single_result['confidences'][0]
    
    print(f"{filename}: Anomaly={is_anomaly}, Error={error:.6f}, Confidence={confidence:.3f}")
```

## ⚙️ Configuration

### Adjusting Sensitivity

Modify `config.py` to change detection behavior:

```python
# More sensitive detection
DETECTION_CONFIG = {
    'confidence_threshold': 0.6,  # Lower threshold
    'error_percentile': 90,       # Lower percentile
}

# Less sensitive detection
DETECTION_CONFIG = {
    'confidence_threshold': 0.9,  # Higher threshold
    'error_percentile': 98,       # Higher percentile
}
```

### Image Processing

```python
# Change image size for faster processing
DATA_CONFIG = {
    'image_size': (256, 256),  # Smaller images
    'grayscale': True,
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Model not found**
   ```
   ❌ Error: Model file 'models/anomaly_detector.pth' not found!
   ```
   **Solution**: Run the training pipeline first: `python test_system.py`

2. **No images found**
   ```
   ❌ Error: Image directory 'path/to/images' not found!
   ```
   **Solution**: Check the image directory path and ensure it contains JPG/PNG files

3. **Memory issues**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or image size in config

4. **Slow processing**
   **Solution**: 
   - Use smaller image sizes
   - Reduce batch size
   - Use CPU instead of GPU (change device in config)

### Performance Tips

- **Faster processing**: Use smaller image sizes (256x256 instead of 512x512)
- **More memory**: Reduce batch size in `MODEL_CONFIG['batch_size']`
- **CPU usage**: Set `SYSTEM_CONFIG['device'] = 'cpu'`

## 📈 Model Performance

Based on test results:
- **Accuracy**: Detects anomalies with ≥80% confidence
- **Speed**: ~412 seconds for 43 images (512x512)
- **Memory**: ~2GB GPU memory usage
- **Output**: Exports anomaly images and detailed reports

## 🎯 Best Practices

1. **Image Quality**: Use high-quality, well-exposed images
2. **Consistent Format**: Ensure all images are the same size and format
3. **Threshold Tuning**: Start with 0.8 confidence, adjust based on results
4. **Validation**: Manually verify high-confidence anomalies
5. **Batch Processing**: Process large datasets in batches

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure the model file exists: `ls models/anomaly_detector.pth`
4. Test with the example script: `python example_usage.py` 