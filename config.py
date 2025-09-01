"""
Configuration file for Space Anomaly Detection System
"""

# Data Configuration
DATA_CONFIG = {
    'data_directory': 'sdss_images',
    'image_size': (512, 512),
    'grayscale': True,
    'test_ratio': 0.2,
    'random_seed': 42
}

# Model Configuration
MODEL_CONFIG = {
    'input_channels': 1,
    'latent_dim': 128,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 50,
    'model_save_path': 'models/anomaly_detector.pth'
}

# Anomaly Detection Configuration
DETECTION_CONFIG = {
    'confidence_threshold': 0.8,  # Minimum confidence for high-confidence anomalies
    'error_percentile': 95,       # Percentile for anomaly threshold
    'min_anomaly_confidence': 0.8  # Minimum confidence to be considered anomaly
}

# Output Configuration
OUTPUT_CONFIG = {
    'anomalies_export_dir': 'anomalies_export',
    'preprocessed_data_dir': 'preprocessed_data',
    'visualization_enabled': True,
    'num_visualization_samples': 5
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_file': 'anomaly_detection.log'
}

# System Configuration
SYSTEM_CONFIG = {
    'device': 'auto',  # 'auto', 'cpu', or 'cuda'
    'num_workers': 4,
    'pin_memory': True
}

# Validation Configuration
VALIDATION_CONFIG = {
    'validation_split': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 1e-6
}