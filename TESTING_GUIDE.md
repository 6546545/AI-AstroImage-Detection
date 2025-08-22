# CosmosAI Systems Testing Guide

## 🧪 **Comprehensive Unit Testing Suite**

This guide covers the complete testing framework for the CosmosAI Space Anomaly Detection System.

## 📋 **Test Overview**

### **Test Coverage:**
- ✅ **Data Preprocessing** - Image loading, resizing, normalization
- ✅ **Anomaly Detection** - Model architecture, inference, results validation
- ✅ **Astronomical Classification** - Object classification, confidence scoring
- ✅ **Multi-Object Detection** - Object detection, bounding boxes, confidence
- ✅ **Enhanced Analysis** - Integrated system testing
- ✅ **Result Export** - JSON export, data formatting
- ✅ **System Integration** - Module imports, initialization
- ✅ **Performance Benchmarks** - Speed and efficiency testing
- ✅ **Error Handling** - Edge cases, invalid inputs
- ✅ **Data Validation** - Input validation, size handling
- ✅ **Output Consistency** - Format validation across systems

## 🚀 **Running Tests**

### **Quick Start:**
```bash
# Run all tests
python run_tests.py

# Or run directly
python test_cosmosai_systems.py
```

### **Individual Test Categories:**
```bash
# Run specific test categories
python -m unittest test_cosmosai_systems.TestCosmosAISystems.test_01_data_preprocessor
python -m unittest test_cosmosai_systems.TestCosmosAISystems.test_03_anomaly_detection_inference
python -m unittest test_cosmosai_systems.TestCosmosAISystems.test_05_astronomical_classification_inference
```

## 📊 **Test Details**

### **1. Data Preprocessor Tests**
- **Purpose**: Validate image loading and preprocessing
- **Tests**: 
  - Image loading from directories
  - Resizing to specified dimensions
  - Normalization to [0, 1] range
  - Handling different image formats
- **Expected Output**: Normalized numpy arrays with correct dimensions

### **2. Anomaly Detection Tests**
- **Purpose**: Validate anomaly detection system
- **Tests**:
  - Model architecture validation
  - Forward pass testing
  - Anomaly detection inference
  - Result format validation
- **Expected Output**: Dictionary with anomaly results and confidence scores

### **3. Astronomical Classification Tests**
- **Purpose**: Validate object classification system
- **Tests**:
  - Model architecture validation
  - Classification inference
  - Confidence scoring
  - Class probability distribution
- **Expected Output**: Classification results with predicted classes and confidence

### **4. Multi-Object Detection Tests**
- **Purpose**: Validate object detection system
- **Tests**:
  - Model architecture validation
  - Object detection inference
  - Bounding box generation
  - Multi-object handling
- **Expected Output**: Detection results with objects, classes, and bounding boxes

### **5. Enhanced Analysis Tests**
- **Purpose**: Validate integrated system functionality
- **Tests**:
  - Combined analysis pipeline
  - Result aggregation
  - Summary generation
- **Expected Output**: Comprehensive analysis results with all system outputs

### **6. Performance Benchmark Tests**
- **Purpose**: Validate system performance
- **Tests**:
  - Processing speed validation
  - Memory usage monitoring
  - Scalability testing
- **Expected Output**: Performance metrics within acceptable limits

### **7. Error Handling Tests**
- **Purpose**: Validate robust error handling
- **Tests**:
  - Empty data handling
  - Invalid input validation
  - Threshold validation
  - Exception handling
- **Expected Output**: Graceful error handling without crashes

## 🔧 **Test Configuration**

### **Environment Variables:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### **Test Parameters:**
- **Device**: CPU (for consistent testing)
- **Image Size**: 512x512 pixels
- **Anomaly Threshold**: 0.8 (80% confidence)
- **Classification Threshold**: 0.7 (70% confidence)
- **Expected Classes**: 10 astronomical object classes

### **Test Data:**
- **Synthetic Images**: Randomly generated test images
- **Sample Images**: Real astronomical images from sample_images/
- **Test Dataset**: Comprehensive test dataset with metadata

## 📈 **Expected Results**

### **Success Criteria:**
- ✅ All 15 test categories pass
- ✅ Performance benchmarks within limits
- ✅ Output formats consistent across systems
- ✅ Error handling works correctly
- ✅ Data validation functions properly

### **Performance Benchmarks:**
- **Anomaly Detection**: < 30 seconds for 10 images
- **Classification**: < 30 seconds for 10 images
- **Object Detection**: < 30 seconds for 10 images
- **Memory Usage**: < 2GB RAM
- **CPU Usage**: < 80% during testing

## 🐛 **Troubleshooting**

### **Common Issues:**

#### **1. Import Errors**
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### **2. Memory Issues**
```bash
# Solution: Reduce test data size or increase system memory
# Modify test parameters in test_cosmosai_systems.py
```

#### **3. Performance Failures**
```bash
# Solution: Check system resources or adjust time limits
# Modify performance thresholds in test_11_performance_benchmarks
```

#### **4. OpenMP Conflicts**
```bash
# Solution: Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
```

### **Debug Mode:**
```bash
# Run with verbose output
python -m unittest test_cosmosai_systems -v

# Run specific test with debug info
python -c "
import test_cosmosai_systems
test = test_cosmosai_systems.TestCosmosAISystems()
test.setUp()
test.test_03_anomaly_detection_inference()
"
```

## 📄 **Test Reports**

### **Generated Reports:**
- **test_report.json**: Comprehensive test results
- **Console Output**: Real-time test progress
- **Error Logs**: Detailed failure information

### **Report Structure:**
```json
{
  "test_suite": "CosmosAI Systems Unit Tests",
  "timestamp": "2025-08-11T10:30:00",
  "total_tests": 15,
  "passed_tests": 15,
  "failed_tests": 0,
  "test_results": {
    "data_preprocessor": true,
    "anomaly_detector_model": true,
    // ... all test results
  },
  "system_info": {
    "device": "cpu",
    "expected_classes": ["asteroid", "black", "comet", ...],
    "anomaly_threshold": 0.8,
    "classification_threshold": 0.7
  }
}
```

## 🎯 **Quality Assurance**

### **Test Validation:**
- ✅ **Unit Tests**: Individual component testing
- ✅ **Integration Tests**: System interaction testing
- ✅ **Performance Tests**: Speed and efficiency validation
- ✅ **Error Tests**: Robustness and error handling
- ✅ **Format Tests**: Output consistency validation

### **Continuous Integration:**
```bash
# Automated testing script
#!/bin/bash
set -e
python run_tests.py
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Tests failed!"
    exit 1
fi
```

## 📚 **Additional Resources**

### **Related Documentation:**
- `README.md` - Main project documentation
- `PRODUCTION_README.md` - Production deployment guide
- `PRODUCTION_READY.md` - Production readiness summary

### **Test Files:**
- `test_cosmosai_systems.py` - Main test suite
- `run_tests.py` - Test runner script
- `TESTING_GUIDE.md` - This testing guide

### **Sample Data:**
- `sample_images/` - Test astronomical images
- `test_dataset/` - Comprehensive test dataset
- `models/` - Pre-trained model files

---

**🎉 Ready to test your CosmosAI systems! Run `python run_tests.py` to get started.**
