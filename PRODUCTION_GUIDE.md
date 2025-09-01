# 🚀 Space Anomaly Detection System - Production Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment Options](#deployment-options)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)
9. [Security](#security)
10. [Performance Optimization](#performance-optimization)

---

## 🚀 Quick Start

### Option 1: Automated Deployment
```bash
# Clone and deploy
git clone <repository>
cd space-anomaly-detector
python3 deploy.py

# Start the system
./start_production.sh
```

### Option 2: Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Check status
docker-compose ps
```

### Option 3: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run system
python3 start_analysis.py
```

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 5GB free space
- **CPU**: 2 cores minimum, 4+ cores recommended

### Recommended Requirements
- **OS**: Ubuntu 20.04+ or macOS 10.15+
- **Python**: 3.9 or higher
- **RAM**: 8GB or more
- **Storage**: 20GB+ SSD
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with CUDA support (optional)

### Dependencies
- PyTorch 2.0+
- OpenCV 4.5+
- NumPy 1.21+
- scikit-learn 1.0+
- Rich (for UI)
- Structlog (for logging)

---

## 📦 Installation

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies
```bash
# Install production dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch, cv2, numpy; print('✅ Dependencies installed')"
```

### 3. Verify Installation
```bash
# Test system
python3 space_analyzer.py test

# Check configuration
python3 config_manager.py
```

---

## ⚙️ Configuration

### Production Configuration File
Edit `production_config.yaml` to customize settings:

```yaml
# System Configuration
system:
  name: "Space Anomaly Detection & Classification System"
  version: "1.0.0"
  environment: "production"
  debug: false
  log_level: "INFO"

# Model Configuration
models:
  classifier:
    path: "models/astronomical_classifier.pth"
    input_size: [512, 512]
    num_classes: 10
    confidence_threshold: 0.8
    device: "auto"
  
  anomaly_detector:
    path: "models/anomaly_detector.pth"
    input_size: [512, 512]
    threshold: 0.15
    device: "auto"

# Processing Configuration
processing:
  batch_size: 8
  num_workers: 4
  max_memory_usage: 0.8
  image_formats: ["jpg", "jpeg", "png", "tiff", "fits"]

# Output Configuration
output:
  results_dir: "results"
  export_formats: ["json", "csv", "html"]
  save_images: true
  create_visualizations: true

# Logging Configuration
logging:
  file: "logs/space_analyzer.log"
  max_file_size: "100MB"
  backup_count: 5
  format: "json"
```

### Environment Variables
```bash
# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LOG_LEVEL=INFO
```

---

## 🚀 Deployment Options

### 1. Local Deployment
```bash
# Deploy locally
python3 deploy.py

# Start production
./start_production.sh
```

### 2. Docker Deployment
```bash
# Build image
docker build -t space-anomaly-detector .

# Run container
docker run -d \
  --name space-analyzer \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  space-anomaly-detector

# Or use Docker Compose
docker-compose up -d
```

### 3. Cloud Deployment

#### AWS EC2
```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair

# SSH and deploy
ssh -i your-key.pem ubuntu@your-instance-ip
git clone <repository>
cd space-anomaly-detector
python3 deploy.py
```

#### Google Cloud Platform
```bash
# Create VM instance
gcloud compute instances create space-analyzer \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2004-lts

# SSH and deploy
gcloud compute ssh space-analyzer
git clone <repository>
cd space-anomaly-detector
python3 deploy.py
```

### 4. Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: space-anomaly-detector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: space-analyzer
  template:
    metadata:
      labels:
        app: space-analyzer
    spec:
      containers:
      - name: space-analyzer
        image: space-anomaly-detector:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: results-volume
          mountPath: /app/results
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: results-volume
        persistentVolumeClaim:
          claimName: results-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
```

---

## 📊 Monitoring

### 1. System Monitoring
```bash
# Start monitoring dashboard
python3 monitor.py

# Run health check
python3 monitor.py --once

# Continuous monitoring
python3 monitor.py --interval 60
```

### 2. Log Monitoring
```bash
# View logs
tail -f logs/space_analyzer.log

# Search logs
grep "ERROR" logs/space_analyzer.log

# Log rotation
logrotate -f /etc/logrotate.d/space-analyzer
```

### 3. Performance Metrics
```bash
# Check system metrics
python3 -c "
from monitor import SystemMonitor
monitor = SystemMonitor()
print(monitor.get_system_metrics())
"

# Performance statistics
python3 -c "
from monitor import SystemMonitor
monitor = SystemMonitor()
print(monitor.get_performance_stats())
"
```

### 4. Grafana Dashboard
```bash
# Start Grafana (included in docker-compose)
docker-compose up monitoring

# Access dashboard
open http://localhost:3000
# Username: admin
# Password: admin
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check model files
ls -la models/

# Re-download models
python3 space_analyzer.py train --model both --epochs 10
```

#### 2. Memory Issues
```bash
# Reduce batch size in production_config.yaml
processing:
  batch_size: 4  # Reduce from 8
  max_memory_usage: 0.6  # Reduce from 0.8
```

#### 3. GPU Issues
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or set in config
models:
  classifier:
    device: "cpu"
  anomaly_detector:
    device: "cpu"
```

#### 4. Permission Issues
```bash
# Fix permissions
chmod +x start_production.sh
chmod +x deploy.py
chmod +x monitor.py

# Create directories with proper permissions
mkdir -p logs results models data
chmod 755 logs results models data
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python3 space_analyzer.py test --verbose

# Check system status
python3 space_analyzer.py test
```

### Performance Issues
```bash
# Profile performance
python3 -m cProfile -o profile.stats space_analyzer.py test

# Analyze profile
python3 -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

---

## 📚 API Reference

### Command Line Interface

#### Basic Commands
```bash
# Test system
python3 space_analyzer.py test

# Classify images
python3 space_analyzer.py classify \
  --input-dir /path/to/images \
  --output-dir /path/to/results

# Detect anomalies
python3 space_analyzer.py detect \
  --input-dir /path/to/images \
  --output-dir /path/to/results

# Combined analysis
python3 space_analyzer.py analyze \
  --input-dir /path/to/images \
  --epochs 20

# Train models
python3 space_analyzer.py train \
  --input-dir /path/to/training/data \
  --model both \
  --epochs 50
```

#### Interactive Mode
```bash
# Start interactive menu
python3 start_analysis.py
```

### Configuration API
```python
from config_manager import config

# Get configuration
classifier_config = config.get_model_config('classifier')
processing_config = config.get_processing_config()
output_config = config.get_output_config()

# Validate configuration
config.validate()

# Print summary
config.print_summary()
```

### Monitoring API
```python
from monitor import SystemMonitor

# Create monitor
monitor = SystemMonitor()

# Get metrics
metrics = monitor.get_system_metrics()
health = monitor.check_system_health()
stats = monitor.get_performance_stats()

# Save metrics
monitor.save_metrics(metrics)
```

---

## 🔒 Security

### 1. File Security
```bash
# Secure file permissions
chmod 600 production_config.yaml
chmod 700 logs/
chmod 700 models/

# Validate file integrity
sha256sum models/*.pth > models.sha256
sha256sum -c models.sha256
```

### 2. Network Security
```bash
# Firewall rules
sudo ufw allow 8000/tcp  # If API enabled
sudo ufw enable

# SSL/TLS (if API enabled)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
```

### 3. Input Validation
```python
# Validate input files
import os
from pathlib import Path

def validate_input_file(file_path: str) -> bool:
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.fits']
    max_size = 100 * 1024 * 1024  # 100MB
    
    path = Path(file_path)
    if not path.exists():
        return False
    
    if path.suffix.lower() not in allowed_extensions:
        return False
    
    if path.stat().st_size > max_size:
        return False
    
    return True
```

### 4. Environment Security
```bash
# Use environment variables for secrets
export API_KEY="your-secret-key"
export DATABASE_URL="your-database-url"

# Don't commit secrets
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore
echo "secrets/" >> .gitignore
```

---

## ⚡ Performance Optimization

### 1. Hardware Optimization
```bash
# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Optimize CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 2. Memory Optimization
```yaml
# production_config.yaml
processing:
  batch_size: 4  # Reduce for memory constraints
  max_memory_usage: 0.7  # Conservative memory usage
  num_workers: 2  # Reduce worker threads
```

### 3. Storage Optimization
```bash
# Use SSD storage
# Enable compression
# Regular cleanup of old results

# Cleanup script
find results/ -name "*.json" -mtime +30 -delete
find logs/ -name "*.log" -mtime +7 -delete
```

### 4. Network Optimization
```bash
# Use local storage for models
# Compress results
# Batch processing
```

### 5. Caching
```python
# Enable model caching
import torch

# Cache models in memory
@lru_cache(maxsize=1)
def load_classifier_model():
    return torch.load('models/astronomical_classifier.pth')

@lru_cache(maxsize=1)
def load_anomaly_model():
    return torch.load('models/anomaly_detector.pth')
```

---

## 📞 Support

### Getting Help
1. **Check logs**: `tail -f logs/space_analyzer.log`
2. **Run diagnostics**: `python3 space_analyzer.py test`
3. **Check configuration**: `python3 config_manager.py`
4. **Monitor system**: `python3 monitor.py --once`

### Common Commands
```bash
# System status
python3 space_analyzer.py test

# Health check
python3 monitor.py --once

# Configuration validation
python3 config_manager.py

# Performance check
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Log Locations
- **Application logs**: `logs/space_analyzer.log`
- **Metrics**: `logs/metrics.json`
- **System logs**: `/var/log/syslog` (Linux)

### Contact
For production support:
- Check the logs first
- Run system diagnostics
- Review this guide
- Create an issue with detailed information

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

**🎉 Your Space Anomaly Detection System is now production-ready!**
