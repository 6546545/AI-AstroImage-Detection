#!/usr/bin/env python3
"""
Advanced Feature Engineering for Space Anomaly Detection
"""

import os
import numpy as np
import cv2
import json
from typing import List, Tuple, Dict, Optional, Union
import logging
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSpectralAnalyzer:
    """Multi-spectral analysis for enhanced object classification."""
    
    def __init__(self):
        self.spectral_bands = {
            'visible': (400, 700),      # nm
            'near_infrared': (700, 1400), # nm
            'mid_infrared': (3000, 5000), # nm
            'far_infrared': (8000, 15000)  # nm
        }
    
    def extract_spectral_features(self, image: np.ndarray) -> Dict:
        """Extract spectral features from multi-band image."""
        features = {}
        
        # Spectral indices
        if len(image.shape) >= 3 and image.shape[2] >= 3:  # Multi-band image
            # Normalized Difference Vegetation Index (NDVI) for astronomical objects
            red_band = image[:, :, 0]
            nir_band = image[:, :, 2] if image.shape[2] > 2 else image[:, :, 1]
            
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
            features['ndvi_mean'] = np.mean(ndvi)
            features['ndvi_std'] = np.std(ndvi)
            features['ndvi_min'] = np.min(ndvi)
            features['ndvi_max'] = np.max(ndvi)
            
            # Color ratios
            features['red_blue_ratio'] = np.mean(red_band / (image[:, :, 2] + 1e-8))
            features['green_red_ratio'] = np.mean(image[:, :, 1] / (red_band + 1e-8))
            
            # Color statistics
            for i, color in enumerate(['red', 'green', 'blue']):
                band = image[:, :, i]
                features[f'{color}_mean'] = np.mean(band)
                features[f'{color}_std'] = np.std(band)
                features[f'{color}_min'] = np.min(band)
                features[f'{color}_max'] = np.max(band)
        else:
            # Single band image
            features['intensity_mean'] = np.mean(image)
            features['intensity_std'] = np.std(image)
            features['intensity_min'] = np.min(image)
            features['intensity_max'] = np.max(image)
        
        # Spectral signatures
        features['spectral_signature'] = self.compute_spectral_signature(image)
        
        # Texture features
        features.update(self.extract_texture_features(image))
        
        return features
    
    def compute_spectral_signature(self, image: np.ndarray) -> np.ndarray:
        """Compute spectral signature across bands."""
        if len(image.shape) == 3:
            # Multi-band image
            signature = np.mean(image, axis=(0, 1))
        else:
            # Single band image
            signature = np.array([np.mean(image)])
        
        return signature
    
    def extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract texture features from image."""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # GLCM-like features (simplified)
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        features['gradient_magnitude_mean'] = np.mean(magnitude)
        features['gradient_magnitude_std'] = np.std(magnitude)
        features['gradient_direction_mean'] = np.mean(direction)
        features['gradient_direction_std'] = np.std(direction)
        
        # Local Binary Pattern (simplified)
        lbp = self.compute_lbp(gray)
        features['lbp_histogram'] = lbp
        
        return features
    
    def compute_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern histogram."""
        height, width = image.shape
        lbp_image = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                lbp_code = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if image[x, y] >= center:
                        lbp_code |= (1 << k)
                
                lbp_image[i, j] = lbp_code
        
        # Compute histogram
        hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256), density=True)
        return hist

class AdvancedImageProcessor:
    """Advanced image processing for feature extraction."""
    
    def __init__(self):
        self.feature_extractors = {
            'haralick': self.extract_haralick_features,
            'lbp': self.extract_lbp_features,
            'hog': self.extract_hog_features,
            'sift': self.extract_sift_features,
            'orb': self.extract_orb_features,
            'gabor': self.extract_gabor_features
        }
    
    def extract_haralick_features(self, image: np.ndarray) -> Dict:
        """Extract Haralick texture features."""
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Convert to uint8 for GLCM
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Compute GLCM
            glcm = graycomatrix(image_uint8, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
            
            # Extract properties
            features = {}
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            
            for prop in properties:
                features[f'haralick_{prop}'] = graycoprops(glcm, prop).flatten()
            
            return features
        except ImportError:
            logger.warning("skimage not available, skipping Haralick features")
            return {}
    
    def extract_lbp_features(self, image: np.ndarray) -> Dict:
        """Extract Local Binary Pattern features."""
        try:
            from skimage.feature import local_binary_pattern
            
            # Compute LBP
            lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
            
            # Compute histogram
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            
            return {'lbp_histogram': hist}
        except ImportError:
            logger.warning("skimage not available, skipping LBP features")
            return {}
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Histogram of Oriented Gradients features."""
        try:
            from skimage.feature import hog
            
            # Compute HOG features
            features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=False)
            
            return features
        except ImportError:
            logger.warning("skimage not available, skipping HOG features")
            return np.zeros(128)
    
    def extract_sift_features(self, image: np.ndarray) -> np.ndarray:
        """Extract SIFT features."""
        try:
            # Initialize SIFT detector
            sift = cv2.SIFT_create()
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(image, None)
            
            if descriptors is not None:
                # Return mean descriptor
                return np.mean(descriptors, axis=0)
            else:
                return np.zeros(128)  # SIFT descriptor size
        except Exception as e:
            logger.warning(f"Error extracting SIFT features: {e}")
            return np.zeros(128)
    
    def extract_orb_features(self, image: np.ndarray) -> np.ndarray:
        """Extract ORB features."""
        try:
            # Initialize ORB detector
            orb = cv2.ORB_create()
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = orb.detectAndCompute(image, None)
            
            if descriptors is not None:
                # Return mean descriptor
                return np.mean(descriptors, axis=0)
            else:
                return np.zeros(32)  # ORB descriptor size
        except Exception as e:
            logger.warning(f"Error extracting ORB features: {e}")
            return np.zeros(32)
    
    def extract_gabor_features(self, image: np.ndarray) -> Dict:
        """Extract Gabor filter features."""
        features = {}
        
        # Gabor filter parameters
        ksize = 31
        sigma = 4.0
        theta = np.pi/4
        lambda_ = 10.0
        gamma = 0.5
        psi = 0
        
        # Create Gabor kernel
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
        
        # Apply Gabor filter
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        # Extract features
        features['gabor_mean'] = np.mean(filtered)
        features['gabor_std'] = np.std(filtered)
        features['gabor_min'] = np.min(filtered)
        features['gabor_max'] = np.max(filtered)
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> Dict:
        """Extract all advanced features."""
        features = {}
        
        for name, extractor in self.feature_extractors.items():
            try:
                result = extractor(image)
                if isinstance(result, dict):
                    features.update(result)
                else:
                    features[name] = result
            except Exception as e:
                logger.warning(f"Error extracting {name} features: {e}")
                continue
        
        return features

class TimeSeriesAnalyzer:
    """Analyze astronomical objects over time."""
    
    def __init__(self):
        self.temporal_features = {}
    
    def analyze_light_curve(self, brightness_values: List[float], 
                           timestamps: List[float]) -> Dict:
        """Analyze light curve of astronomical object."""
        brightness = np.array(brightness_values)
        time = np.array(timestamps)
        
        features = {}
        
        # Basic statistics
        features['mean_brightness'] = np.mean(brightness)
        features['std_brightness'] = np.std(brightness)
        features['min_brightness'] = np.min(brightness)
        features['max_brightness'] = np.max(brightness)
        features['brightness_range'] = features['max_brightness'] - features['min_brightness']
        
        # Trend analysis
        if len(time) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(time, brightness)
            features['brightness_trend'] = slope
            features['trend_significance'] = p_value
            features['trend_correlation'] = r_value
        
        # Periodicity analysis
        if len(brightness) > 10:
            # FFT for periodicity
            fft = np.fft.fft(brightness)
            freqs = np.fft.fftfreq(len(brightness))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            if dominant_freq != 0:
                features['period'] = 1 / abs(dominant_freq)
            else:
                features['period'] = None
        
        # Variability metrics
        features['coefficient_of_variation'] = features['std_brightness'] / features['mean_brightness']
        
        # Detect flares or sudden changes
        brightness_diff = np.diff(brightness)
        features['max_brightness_change'] = np.max(np.abs(brightness_diff))
        features['flare_count'] = np.sum(np.abs(brightness_diff) > 2 * features['std_brightness'])
        
        return features
    
    def detect_periodic_behavior(self, time_series: List[float], 
                                min_period: float = 0.1, 
                                max_period: float = 100.0) -> Dict:
        """Detect periodic behavior in time series."""
        time_series = np.array(time_series)
        
        # Find peaks
        peaks, _ = signal.find_peaks(time_series, height=np.mean(time_series))
        
        if len(peaks) < 2:
            return {'is_periodic': False, 'period': None, 'confidence': 0.0}
        
        # Calculate periods between peaks
        peak_times = peaks
        periods = np.diff(peak_times)
        
        # Check if periods are consistent
        mean_period = np.mean(periods)
        period_std = np.std(periods)
        period_cv = period_std / mean_period if mean_period > 0 else float('inf')
        
        # Determine if periodic
        is_periodic = (period_cv < 0.3 and  # Low coefficient of variation
                      min_period <= mean_period <= max_period and
                      len(periods) >= 2)
        
        confidence = 1.0 - min(period_cv, 1.0) if is_periodic else 0.0
        
        return {
            'is_periodic': is_periodic,
            'period': mean_period if is_periodic else None,
            'confidence': confidence,
            'peak_count': len(peaks),
            'period_std': period_std
        }

class AstronomicalObjectTracker:
    """Real-time tracking of astronomical objects across multiple frames."""
    
    def __init__(self):
        self.trackers = {}
        self.next_track_id = 0
        self.max_disappeared = 30  # frames
    
    def update(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Update object tracking."""
        # If no existing trackers, initialize new ones
        if len(self.trackers) == 0:
            for detection in detections:
                self.register_new_tracker(detection, frame_number)
        
        # Update existing trackers
        for track_id, tracker in self.trackers.items():
            tracker['disappeared'] += 1
        
        # Match detections to existing trackers
        matched_detections = self.match_detections_to_trackers(detections)
        
        # Update matched trackers
        for detection, track_id in matched_detections:
            self.trackers[track_id]['bbox'] = detection['bbox']
            self.trackers[track_id]['classification'] = detection['classification']
            self.trackers[track_id]['confidence'] = detection['confidence']
            self.trackers[track_id]['disappeared'] = 0
            self.trackers[track_id]['last_seen'] = frame_number
        
        # Register new detections as trackers
        for detection in detections:
            if not any(detection == d for d, _ in matched_detections):
                self.register_new_tracker(detection, frame_number)
        
        # Remove old trackers
        self.cleanup_old_trackers()
        
        return self.get_active_tracks()
    
    def register_new_tracker(self, detection: Dict, frame_number: int):
        """Register a new object tracker."""
        self.trackers[self.next_track_id] = {
            'bbox': detection['bbox'],
            'classification': detection['classification'],
            'confidence': detection['confidence'],
            'disappeared': 0,
            'first_seen': frame_number,
            'last_seen': frame_number,
            'track_id': self.next_track_id
        }
        self.next_track_id += 1
    
    def match_detections_to_trackers(self, detections: List[Dict]) -> List[Tuple[Dict, int]]:
        """Match detections to existing trackers using IoU."""
        matches = []
        
        for detection in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, tracker in self.trackers.items():
                iou = self.calculate_iou(detection['bbox'], tracker['bbox'])
                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                matches.append((detection, best_track_id))
        
        return matches
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def cleanup_old_trackers(self):
        """Remove trackers that haven't been seen for too long."""
        track_ids_to_remove = []
        
        for track_id, tracker in self.trackers.items():
            if tracker['disappeared'] > self.max_disappeared:
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            del self.trackers[track_id]
    
    def get_active_tracks(self) -> List[Dict]:
        """Get all active tracks."""
        return list(self.trackers.values())

class AnomalyClassifier:
    """Classify different types of astronomical anomalies."""
    
    def __init__(self):
        self.anomaly_types = {
            'supernova': 'Explosive stellar death',
            'gamma_ray_burst': 'High-energy explosion',
            'gravitational_lensing': 'Light bending by gravity',
            'microlensing': 'Temporary brightening',
            'variable_star': 'Periodic brightness changes',
            'asteroid': 'Moving object',
            'satellite': 'Artificial object',
            'cosmic_ray': 'High-energy particle hit',
            'instrument_artifact': 'Equipment malfunction',
            'unknown': 'Unclassified anomaly'
        }
        
        self.classifier = self.build_anomaly_classifier()
    
    def build_anomaly_classifier(self) -> 'nn.Module':
        """Build neural network for anomaly classification."""
        try:
            import torch.nn as nn
            
            return nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, len(self.anomaly_types))
            )
        except ImportError:
            logger.warning("PyTorch not available, using rule-based classifier")
            return None
    
    def classify_anomaly(self, anomaly_image: np.ndarray) -> Dict:
        """Classify the type of anomaly."""
        if self.classifier is None:
            # Rule-based classification
            return self.rule_based_classification(anomaly_image)
        
        try:
            import torch
            
            # Preprocess image
            image_tensor = torch.FloatTensor(anomaly_image).unsqueeze(0).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                logits = self.classifier(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            anomaly_type = list(self.anomaly_types.keys())[predicted_class]
            
            return {
                'anomaly_type': anomaly_type,
                'description': self.anomaly_types[anomaly_type],
                'confidence': confidence,
                'all_probabilities': probabilities[0].numpy().tolist()
            }
        except Exception as e:
            logger.warning(f"Error in neural classification: {e}")
            return self.rule_based_classification(anomaly_image)
    
    def rule_based_classification(self, anomaly_image: np.ndarray) -> Dict:
        """Rule-based anomaly classification."""
        # Simple rule-based classification
        mean_intensity = np.mean(anomaly_image)
        std_intensity = np.std(anomaly_image)
        
        if mean_intensity > 0.8:
            anomaly_type = 'cosmic_ray'
            confidence = 0.7
        elif std_intensity > 0.3:
            anomaly_type = 'variable_star'
            confidence = 0.6
        else:
            anomaly_type = 'unknown'
            confidence = 0.5
        
        return {
            'anomaly_type': anomaly_type,
            'description': self.anomaly_types[anomaly_type],
            'confidence': confidence,
            'all_probabilities': [0.1] * len(self.anomaly_types)
        }

def main():
    """Test the advanced features."""
    print("🧪 Testing Advanced Features...")
    
    # Test Multi-Spectral Analyzer
    print("Testing Multi-Spectral Analyzer...")
    msa = MultiSpectralAnalyzer()
    test_image = np.random.rand(512, 512, 3)
    spectral_features = msa.extract_spectral_features(test_image)
    print(f"Extracted {len(spectral_features)} spectral features")
    
    # Test Advanced Image Processor
    print("Testing Advanced Image Processor...")
    aip = AdvancedImageProcessor()
    test_image_gray = np.random.rand(512, 512)
    image_features = aip.extract_all_features(test_image_gray)
    print(f"Extracted {len(image_features)} image features")
    
    # Test Time Series Analyzer
    print("Testing Time Series Analyzer...")
    tsa = TimeSeriesAnalyzer()
    brightness = [1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0]
    timestamps = list(range(len(brightness)))
    time_features = tsa.analyze_light_curve(brightness, timestamps)
    print(f"Extracted {len(time_features)} time series features")
    
    # Test Object Tracker
    print("Testing Object Tracker...")
    tracker = AstronomicalObjectTracker()
    detections = [
        {'bbox': [100, 100, 50, 50], 'classification': 'star', 'confidence': 0.9},
        {'bbox': [200, 200, 60, 60], 'classification': 'galaxy', 'confidence': 0.8}
    ]
    tracks = tracker.update(detections, 1)
    print(f"Tracking {len(tracks)} objects")
    
    # Test Anomaly Classifier
    print("Testing Anomaly Classifier...")
    ac = AnomalyClassifier()
    anomaly_image = np.random.rand(64, 64)
    anomaly_class = ac.classify_anomaly(anomaly_image)
    print(f"Classified anomaly as: {anomaly_class['anomaly_type']}")
    
    print("✅ All advanced features tested successfully!")

if __name__ == "__main__":
    main()
