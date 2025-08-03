# Space Anomaly Detection System - GUI

A user-friendly graphical interface for the Space Anomaly Detection System that allows users to easily train models, detect anomalies, and view results without using the command line.

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
python launch_gui.py
```

### Option 2: Direct Launch
```bash
python gui_app.py
```

## 📋 Features

### 🏠 Main Tab
- **System Status**: Check if trained model exists
- **Quick Actions**: One-click access to common operations
  - Run Complete Pipeline
  - Load Existing Model
  - Detect Anomalies
  - View Results

### 🎯 Training Tab
- **Training Parameters**:
  - Data Directory selection
  - Training Epochs (1-100)
  - Batch Size (1-128)
  - Learning Rate
- **Training Controls**:
  - Start/Stop Training
  - Progress Bar
  - Real-time Training Log

### 🔍 Detection Tab
- **Detection Parameters**:
  - Confidence Threshold (0.0-1.0)
  - Model Path selection
- **Detection Controls**:
  - Detect Anomalies
  - Export Results
  - Real-time Detection Log

### 📊 Results Tab
- **Detection Summary**: Overview of results
- **Image Viewer**: Browse detected anomaly images
  - Previous/Next navigation
  - Image counter
  - Thumbnail display

### ⚙️ Settings Tab
- **System Settings**:
  - Device selection (auto/cpu/cuda/mps)
  - Output directory
- **Settings Management**:
  - Save/Load settings
  - Reset to defaults
- **System Log**: Real-time logging

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- All dependencies from `requirements.txt`

### Automatic Installation
The launcher script will automatically check and install missing dependencies:

```bash
python launch_gui.py
```

### Manual Installation
```bash
pip install -r requirements.txt
```

### Required Dependencies
- `tkinter` (usually included with Python)
- `matplotlib>=3.5.0`
- `numpy`
- `torch`
- `opencv-python`
- `Pillow`
- `scikit-learn`
- `tqdm`

## 📖 Usage Guide

### First Time Setup

1. **Launch the GUI**:
   ```bash
   python launch_gui.py
   ```

2. **Check System Status**:
   - The main tab shows if a trained model exists
   - If no model is found, you'll need to train one

3. **Configure Settings** (Optional):
   - Go to Settings tab
   - Adjust device, output directory, etc.
   - Save settings if needed

### Training a Model

1. **Go to Training Tab**

2. **Set Parameters**:
   - **Data Directory**: Path to your image data (default: `sdss_images`)
   - **Training Epochs**: Number of training cycles (recommended: 10-50)
   - **Batch Size**: Number of images per batch (default: 32)
   - **Learning Rate**: How fast the model learns (default: 0.001)

3. **Start Training**:
   - Click "Start Training"
   - Monitor progress in the Training Log
   - Training runs in background (GUI remains responsive)

4. **Training Complete**:
   - Model is automatically saved to `models/anomaly_detector.pth`
   - Status updates to show model is ready

### Detecting Anomalies

1. **Go to Detection Tab**

2. **Set Parameters**:
   - **Confidence Threshold**: Minimum confidence for anomalies (0.0-1.0)
   - **Model Path**: Path to trained model (auto-detected)

3. **Run Detection**:
   - Click "Detect Anomalies"
   - Monitor progress in Detection Log
   - Results are automatically exported

### Viewing Results

1. **Go to Results Tab**

2. **View Summary**:
   - Total images processed
   - Number of anomalies found
   - Confidence and error thresholds

3. **Browse Images**:
   - Use Previous/Next buttons to navigate
   - View detected anomaly images
   - Image counter shows current position

### Exporting Results

1. **From Detection Tab**:
   - Click "Export Results" after detection
   - Choose save location
   - Results saved as JSON file

2. **From Results Tab**:
   - View current results
   - Export to file if needed

## 🎨 GUI Features

### Modern Interface
- **Tabbed Layout**: Organized into logical sections
- **Real-time Logging**: See what's happening as it happens
- **Progress Indicators**: Visual feedback for long operations
- **Error Handling**: User-friendly error messages

### Responsive Design
- **Background Processing**: GUI stays responsive during operations
- **Threading**: Long operations don't freeze the interface
- **Status Updates**: Real-time status information

### User-Friendly Controls
- **File Browsers**: Easy selection of directories and files
- **Sliders and Spinboxes**: Intuitive parameter adjustment
- **Dropdown Menus**: Predefined options for common settings

## 🔧 Troubleshooting

### Common Issues

#### "No module named 'tkinter'"
**Solution**: Install Python with tkinter support
```bash
# On Ubuntu/Debian
sudo apt-get install python3-tk

# On macOS with Homebrew
brew install python-tk

# On Windows
# Usually included with Python installation
```

#### "Model not found"
**Solution**: Train a model first using the Training tab

#### "Data directory not found"
**Solution**: 
1. Place your images in a directory
2. Update the Data Directory in Training tab
3. Or use the default `sdss_images` directory

#### "CUDA/MPS not available"
**Solution**: 
1. Go to Settings tab
2. Change Device to "cpu"
3. Or install appropriate GPU drivers

### Performance Tips

1. **Use GPU if available**:
   - Set Device to "auto" or "cuda" in Settings
   - Significantly faster training and detection

2. **Adjust batch size**:
   - Larger batch sizes = faster training
   - Limited by available memory

3. **Monitor system resources**:
   - Training uses significant CPU/GPU
   - Close other applications if needed

### Error Messages

#### "Pipeline failed"
- Check data directory exists and contains images
- Ensure sufficient disk space
- Verify all dependencies are installed

#### "Detection failed"
- Ensure trained model exists
- Check model file is not corrupted
- Verify test data is available

#### "GUI not responding"
- Long operations run in background
- Check the log tabs for progress
- Wait for operation to complete

## 📁 File Structure

```
Python/
├── gui_app.py              # Main GUI application
├── launch_gui.py           # GUI launcher with dependency checking
├── space_anomaly_detector.py  # Core system (used by GUI)
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── README_GUI.md          # This file
├── sdss_images/           # Input data directory
├── models/                # Trained models
├── preprocessed_data/     # Processed data
└── anomalies_export/      # Detection results
```

## 🔄 Integration with Command Line

The GUI uses the same underlying system as the command-line tools:

- **Same Models**: Models trained via GUI work with command-line tools
- **Same Data**: Data processed via GUI works with command-line tools
- **Same Results**: Results from GUI are compatible with command-line tools

You can switch between GUI and command-line as needed.

## 🆘 Support

### Getting Help

1. **Check the logs**: Each tab has a log section showing detailed information
2. **Read error messages**: GUI provides specific error information
3. **Check system status**: Main tab shows current system state

### Reporting Issues

When reporting issues, please include:
- Operating system and Python version
- Error messages from the log tabs
- Steps to reproduce the issue
- System specifications (CPU, GPU, memory)

## 🎯 Advanced Usage

### Custom Settings
- Save custom settings in Settings tab
- Load settings for different projects
- Reset to defaults when needed

### Batch Processing
- Train multiple models with different parameters
- Compare results using different confidence thresholds
- Export results for further analysis

### Integration
- Use GUI for interactive work
- Use command-line for automation
- Combine both approaches as needed

---

**Happy Anomaly Detection! 🚀** 