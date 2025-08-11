#!/usr/bin/env python3
"""
Multi-Object Detection and Classification System for Astronomical Images
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
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy import ndimage
from scipy.signal import find_peaks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetectionModel(nn.Module):
    """CNN model for detecting multiple objects in astronomical images."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()
        
        # Feature extraction layers
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
            nn.Dropout2d(0.25),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Object detection head (regression for bounding boxes)
        self.detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # x, y, width, height
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the model."""
        features = self.features(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classification output
        class_output = self.classifier(flattened)
        
        # Detection output (bounding boxes)
        detection_output = self.detector(flattened)
        
        return class_output, detection_output

class MultiObjectDetector:
    """System for detecting and classifying multiple objects in astronomical images."""
    
    def __init__(self, device: str = "auto"):
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.detection_model = None
        self.class_names = [
            'star', 'galaxy', 'nebula', 'planet', 'asteroid', 
            'comet', 'quasar', 'pulsar', 'black_hole', 'unknown'
        ]
        self.num_classes = len(self.class_names)
        
        # Detection parameters
        self.min_object_size = 20  # Minimum object size in pixels
        self.max_objects = 50      # Maximum objects to detect per image
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.5   # Non-maximum suppression threshold
        
        logger.info("Multi-Object Detector initialized")
    
    def detect_objects_in_image(self, image: np.ndarray, 
                               confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Detect multiple objects in a single image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            confidence_threshold: Minimum confidence for object detection
            
        Returns:
            List of detected objects with bounding boxes and classifications
        """
        logger.info("Detecting objects in image...")
        
        # Preprocess image
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert RGB to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 1:
                image = image.squeeze()
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Method 1: Traditional computer vision approach
        objects_cv = self._detect_objects_cv(image)
        
        # Method 2: Deep learning approach (if model is available)
        objects_dl = self._detect_objects_dl(image, confidence_threshold)
        
        # Combine results
        all_objects = objects_cv + objects_dl
        
        # Apply non-maximum suppression
        filtered_objects = self._apply_nms(all_objects)
        
        # Limit number of objects
        if len(filtered_objects) > self.max_objects:
            filtered_objects = filtered_objects[:self.max_objects]
        
        logger.info(f"Detected {len(filtered_objects)} objects")
        return filtered_objects
    
    def _detect_objects_cv(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using traditional computer vision techniques."""
        objects = []
        
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Method 1: Threshold-based detection
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < self.min_object_size:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on area and shape
            confidence = min(area / 1000.0, 0.9)  # Simple confidence heuristic
            
            objects.append({
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'method': 'contour',
                'area': area
            })
        
        # Method 2: Peak detection
        # Find local maxima in the image
        peaks = self._find_peaks_2d(image)
        
        for peak in peaks:
            y, x = peak
            
            # Create bounding box around peak
            size = 30  # Fixed size for peak-based detection
            x1 = max(0, x - size // 2)
            y1 = max(0, y - size // 2)
            x2 = min(image.shape[1], x + size // 2)
            y2 = min(image.shape[0], y + size // 2)
            
            # Calculate confidence based on peak intensity
            confidence = image[y, x]
            
            if confidence > 0.3:  # Minimum intensity threshold
                objects.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'confidence': confidence,
                    'method': 'peak',
                    'center': [x, y]
                })
        
        # Method 3: Blob detection
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        
        # Find blobs using simple threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_object_size:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate confidence based on area and intensity
            region = image[y:y+h, x:x+w]
            avg_intensity = np.mean(region)
            confidence = min(avg_intensity * 0.8, 0.9)
            
            objects.append({
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'method': 'blob',
                'area': area,
                'intensity': avg_intensity
            })
        
        return objects
    
    def _detect_objects_dl(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """Detect objects using deep learning model (if available)."""
        objects = []
        
        if self.detection_model is None:
            return objects
        
        # Prepare image for model
        img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            class_output, detection_output = self.detection_model(img_tensor)
            
            # Process classification results
            probabilities = F.softmax(class_output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Process detection results (bounding boxes)
            bboxes = detection_output.cpu().numpy()[0]
            
            # Convert normalized coordinates to pixel coordinates
            h, w = image.shape
            x, y, box_w, box_h = bboxes
            
            # Scale to image dimensions
            x = int(x * w)
            y = int(y * h)
            box_w = int(box_w * w)
            box_h = int(box_h * h)
            
            if confidence.item() > confidence_threshold:
                objects.append({
                    'bbox': [x, y, box_w, box_h],
                    'confidence': confidence.item(),
                    'method': 'deep_learning',
                    'classification': self.class_names[predicted.item()],
                    'probabilities': probabilities.cpu().numpy()[0].tolist()
                })
        
        return objects
    
    def _find_peaks_2d(self, image: np.ndarray, min_distance: int = 10) -> List[Tuple[int, int]]:
        """Find 2D peaks in the image."""
        peaks = []
        
        # Find peaks in each row
        for i in range(image.shape[0]):
            row_peaks, _ = find_peaks(image[i], distance=min_distance)
            for j in row_peaks:
                peaks.append((i, j))
        
        # Find peaks in each column
        for j in range(image.shape[1]):
            col_peaks, _ = find_peaks(image[:, j], distance=min_distance)
            for i in col_peaks:
                if (i, j) not in peaks:
                    peaks.append((i, j))
        
        return peaks
    
    def _apply_nms(self, objects: List[Dict]) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not objects:
            return objects
        
        # Sort by confidence
        objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)
        
        filtered_objects = []
        
        for obj in objects:
            should_keep = True
            
            for kept_obj in filtered_objects:
                # Calculate intersection over union
                iou = self._calculate_iou(obj['bbox'], kept_obj['bbox'])
                
                if iou > self.nms_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                filtered_objects.append(obj)
        
        return filtered_objects
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
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
    
    def classify_detected_objects(self, image: np.ndarray, objects: List[Dict],
                                classifier_model_path: str = "models/astronomical_classifier.pth") -> List[Dict]:
        """
        Classify detected objects using the existing classifier.
        
        Args:
            image: Original image
            objects: List of detected objects with bounding boxes
            classifier_model_path: Path to trained classifier model
            
        Returns:
            List of objects with classifications
        """
        logger.info(f"Classifying {len(objects)} detected objects...")
        
        # Load classifier if available
        classifier = None
        if os.path.exists(classifier_model_path):
            try:
                from astronomical_classifier import AstronomicalClassificationSystem
                classifier = AstronomicalClassificationSystem(device=str(self.device))
                classifier.load_classifier(classifier_model_path)
                logger.info("Loaded existing classifier for object classification")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}")
        
        classified_objects = []
        
        for i, obj in enumerate(tqdm(objects, desc="Classifying objects")):
            x, y, w, h = obj['bbox']
            
            # Extract object region
            object_region = image[y:y+h, x:x+w]
            
            if object_region.size == 0:
                continue
            
            # Resize to standard size
            object_region = cv2.resize(object_region, (64, 64))
            
            # Normalize
            if object_region.dtype != np.float32:
                object_region = object_region.astype(np.float32) / 255.0
            
            # Classify object
            classification_result = self._classify_single_object(object_region, classifier)
            
            # Combine detection and classification results
            classified_obj = {
                **obj,
                'classification': classification_result['classification'],
                'classification_confidence': classification_result['confidence'],
                'probabilities': classification_result.get('probabilities', {}),
                'object_id': i
            }
            
            classified_objects.append(classified_obj)
        
        return classified_objects
    
    def _classify_single_object(self, object_region: np.ndarray, 
                               classifier) -> Dict:
        """Classify a single object region."""
        if classifier is None:
            # Fallback classification based on image properties
            return self._fallback_classification(object_region)
        
        try:
            # Prepare input for classifier
            if len(object_region.shape) == 2:
                object_region = object_region.reshape(1, 1, 64, 64)
            else:
                object_region = object_region.reshape(1, 64, 64, 1)
                object_region = np.transpose(object_region, (0, 3, 1, 2))
            
            # Classify using the existing classifier
            results = classifier.classify_objects(object_region, confidence_threshold=0.5)
            
            if results['predictions']:
                pred = results['predictions'][0]
                return {
                    'classification': pred['classification'],
                    'confidence': pred['confidence'],
                    'probabilities': dict(zip(classifier.class_names, pred['probabilities']))
                }
        
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
        
        # Fallback classification
        return self._fallback_classification(object_region)
    
    def _fallback_classification(self, object_region: np.ndarray) -> Dict:
        """Fallback classification based on image properties."""
        # Calculate basic features
        mean_intensity = np.mean(object_region)
        std_intensity = np.std(object_region)
        area = object_region.shape[0] * object_region.shape[1]
        
        # Simple rule-based classification
        if mean_intensity > 0.8:
            classification = 'star'
            confidence = 0.6
        elif std_intensity > 0.3:
            classification = 'galaxy'
            confidence = 0.5
        elif area > 1000:
            classification = 'nebula'
            confidence = 0.4
        else:
            classification = 'unknown'
            confidence = 0.3
        
        return {
            'classification': classification,
            'confidence': confidence,
            'probabilities': {classification: confidence}
        }
    
    def visualize_detections(self, image: np.ndarray, objects: List[Dict],
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected objects on the image.
        
        Args:
            image: Original image
            objects: List of detected and classified objects
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()
        
        # Define colors for different classes
        colors = {
            'star': (255, 0, 0),      # Red
            'galaxy': (0, 255, 0),    # Green
            'nebula': (0, 0, 255),    # Blue
            'planet': (255, 255, 0),  # Yellow
            'asteroid': (255, 0, 255), # Magenta
            'comet': (0, 255, 255),   # Cyan
            'quasar': (128, 0, 128),  # Purple
            'pulsar': (255, 165, 0),  # Orange
            'black_hole': (0, 0, 0),  # Black
            'unknown': (128, 128, 128) # Gray
        }
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            classification = obj.get('classification', 'unknown')
            confidence = obj.get('confidence', 0)
            
            # Get color for this class
            color = colors.get(classification, colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{classification} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save visualization if path provided
        if output_path:
            cv2.imwrite(output_path, (vis_image * 255).astype(np.uint8))
            logger.info(f"Visualization saved to {output_path}")
        
        return vis_image
    
    def export_multi_object_results(self, image: np.ndarray, objects: List[Dict],
                                  filename: str, output_dir: str = "multi_object_results") -> str:
        """
        Export multi-object detection and classification results.
        
        Args:
            image: Original image
            objects: List of detected and classified objects
            filename: Original image filename
            output_dir: Output directory
            
        Returns:
            Path to exported results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        vis_image = self.visualize_detections(image, objects)
        
        # Save visualization
        vis_filename = f"detected_{filename}"
        vis_path = os.path.join(output_dir, vis_filename)
        cv2.imwrite(vis_path, (vis_image * 255).astype(np.uint8))
        
        # Prepare results for export
        export_objects = []
        for obj in objects:
            export_obj = {
                'object_id': obj.get('object_id', 0),
                'bbox': obj['bbox'],
                'detection_confidence': obj['confidence'],
                'classification': obj.get('classification', 'unknown'),
                'classification_confidence': obj.get('classification_confidence', 0),
                'detection_method': obj.get('method', 'unknown'),
                'probabilities': obj.get('probabilities', {}),
                'area': obj.get('area', 0),
                'center': obj.get('center', [0, 0])
            }
            export_objects.append(export_obj)
        
        # Create summary
        summary = {
            'filename': filename,
            'total_objects': len(objects),
            'detection_methods': list(set(obj.get('method', 'unknown') for obj in objects)),
            'classifications': {},
            'confidence_ranges': {
                'high': len([obj for obj in objects if obj.get('confidence', 0) > 0.8]),
                'medium': len([obj for obj in objects if 0.5 < obj.get('confidence', 0) <= 0.8]),
                'low': len([obj for obj in objects if obj.get('confidence', 0) <= 0.5])
            }
        }
        
        # Count classifications
        for obj in objects:
            classification = obj.get('classification', 'unknown')
            summary['classifications'][classification] = summary['classifications'].get(classification, 0) + 1
        
        # Save results
        results = {
            'summary': summary,
            'objects': export_objects,
            'visualization_path': vis_path
        }
        
        results_filename = f"results_{filename.replace('.', '_')}.json"
        results_path = os.path.join(output_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Multi-object results exported to {results_path}")
        return results_path

def main():
    """Demo function for multi-object detection."""
    # Initialize detector
    detector = MultiObjectDetector()
    
    # Load sample image
    sample_image_path = "sample_images/star/star_sample_1.png"
    if os.path.exists(sample_image_path):
        image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        
        print("🔍 Detecting objects in sample image...")
        
        # Detect objects
        objects = detector.detect_objects_in_image(image)
        
        # Classify objects
        classified_objects = detector.classify_detected_objects(image, objects)
        
        # Visualize results
        vis_image = detector.visualize_detections(image, classified_objects)
        
        # Export results
        detector.export_multi_object_results(image, classified_objects, "sample_image.png")
        
        print(f"✅ Detected and classified {len(classified_objects)} objects")
        print("📁 Results saved to multi_object_results/")
        
        # Display summary
        for obj in classified_objects:
            print(f"  Object {obj['object_id']}: {obj['classification']} "
                  f"(confidence: {obj['confidence']:.2f})")
    else:
        print("❌ Sample image not found. Please run with your own images.")

if __name__ == "__main__":
    main()
