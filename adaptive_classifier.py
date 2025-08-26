#!/usr/bin/env python3
"""
Adaptive Astronomical Object Classifier
Supports both binary (star vs galaxy) and full multi-class classification
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
import json
from typing import List, Tuple, Dict, Optional, Union
import logging
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveClassifier(nn.Module):
    """Adaptive CNN classifier for astronomical object classification."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Second convolutional block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

class AdaptiveAstronomicalClassification:
    """Adaptive classification system that can handle both binary and multi-class scenarios."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.MODEL_DIR = Path("models")
        self.MODEL_DIR.mkdir(exist_ok=True)
        
        # Define both model paths
        self.BINARY_MODEL_PATH = self.MODEL_DIR / "binary_classifier.pth"
        self.FULL_MODEL_PATH = self.MODEL_DIR / "full_classifier.pth"
        
        # Will be set during initialization
        self.classifier = None
        self.binary_mode = None
        self.active_classes = None
        
        # Initialize class mappings
        self.BINARY_CLASSES = ['star', 'galaxy']
        self.FULL_CLASSES = [
            'star', 'galaxy', 'nebula', 'planet', 'asteroid',
            'comet', 'quasar', 'pulsar', 'black_hole'
        ]
    
    def get_available_classes(self, data_dir: str = "clean_dataset/processed_dataset/train") -> List[str]:
        """Determine which classes have data available."""
        available = []
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"Data directory {data_dir} not found!")
            return available
        
        for class_name in self.FULL_CLASSES:
            class_dir = data_path / class_name
            if class_dir.exists() and any(class_dir.iterdir()):
                available.append(class_name)
        return available
    
    def initialize_classifier(self):
        """Initialize the classifier based on available data."""
        # Check available classes
        available_classes = self.get_available_classes()
        if not available_classes:
            raise ValueError("No classes found in the dataset!")
        
        # Determine operating mode
        self.binary_mode = len(available_classes) <= 2
        self.active_classes = self.BINARY_CLASSES if self.binary_mode else self.FULL_CLASSES
        
        # Create appropriate model
        num_classes = len(self.active_classes)
        self.classifier = AdaptiveClassifier(num_classes=num_classes).to(self.device)
        
        # Load existing model if available
        model_path = self.BINARY_MODEL_PATH if self.binary_mode else self.FULL_MODEL_PATH
        if model_path.exists():
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"✅ Loaded existing {'binary' if self.binary_mode else 'full'} model")
        
        # Log configuration
        logger.info(f"🔧 Operating Mode: {'Binary' if self.binary_mode else 'Full'} Classification")
        logger.info(f"📊 Active Classes: {self.active_classes}")
        
        self.classifier.eval()
    
    def classify_image(self, image_path: str) -> List[Dict]:
        """Classify a single image."""
        if self.classifier is None:
            self.initialize_classifier()
        
        # Read and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize and normalize
        img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 255.0
        
        # Prepare for PyTorch
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Convert to numpy for processing
        probs = probabilities.cpu().numpy()[0]
        
        # Format results
        results = []
        for i, (prob, class_name) in enumerate(zip(probs, self.active_classes)):
            results.append({
                'class_name': class_name,
                'confidence': float(prob),
                'type': 'binary' if self.binary_mode else 'multi-class'
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

# Example usage
if __name__ == "__main__":
    classifier = AdaptiveAstronomicalClassification()
    classifier.initialize_classifier()
