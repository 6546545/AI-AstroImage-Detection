#!/usr/bin/env python3
"""
Astronomical Object Classifier for Space Anomaly Detection System
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AstronomicalObjectClassifier(nn.Module):
    """CNN classifier for astronomical object classification."""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
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
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
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
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class AstronomicalClassificationSystem:
    """Complete system for astronomical object classification."""
    
    def __init__(self, device: str = "auto"):
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Known astronomical object classes
        self.known_classes = {
            0: "star",
            1: "galaxy", 
            2: "nebula",
            3: "planet",
            4: "asteroid",
            5: "comet",
            6: "quasar",
            7: "pulsar",
            8: "black_hole",
            9: "unknown"
        }
        
        self.num_classes = len(self.known_classes)
        self.classifier = None
        self.class_names = list(self.known_classes.values())
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Classes: {self.class_names}")
    
    def load_images_from_directory(self, directory: str) -> Tuple[np.ndarray, List[str]]:
        """Load and preprocess images from a directory (supports nested subdirectories)."""
        logger.info(f"Loading images from {directory}")
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        images = []
        filenames = []
        
        # Check if directory contains subdirectories (class-based organization)
        subdirs = [d for d in os.listdir(directory) 
                  if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')]
        
        if subdirs:
            # Load from subdirectories (class-based organization)
            logger.info(f"Found {len(subdirs)} subdirectories, loading from all classes")
            for subdir in subdirs:
                subdir_path = os.path.join(directory, subdir)
                image_files = [f for f in os.listdir(subdir_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for file in image_files:
                    path = os.path.join(subdir_path, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        logger.warning(f"Skipping unreadable file: {file}")
                        continue
                    
                    # Resize and normalize
                    img = cv2.resize(img, (512, 512))
                    img = img / 255.0  # Normalize to [0, 1]
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    
                    images.append(img)
                    filenames.append(f"{subdir}/{file}")
        else:
            # Load from flat directory
            image_files = [f for f in os.listdir(directory) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in image_files:
                path = os.path.join(directory, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    logger.warning(f"Skipping unreadable file: {file}")
                    continue
                
                # Resize and normalize
                img = cv2.resize(img, (512, 512))
                img = img / 255.0  # Normalize to [0, 1]
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                
                images.append(img)
                filenames.append(file)
        
        if not images:
            logger.warning(f"No image files found in {directory}")
            return np.array([]), []
        
        # Create progress bar for image loading
        pbar = tqdm(images, desc="Loading images", unit="image")
        
        # Update progress bar
        pbar.set_postfix({
            'Loaded': len(images),
            'Classes': len(set([f.split('/')[0] for f in filenames if '/' in f]))
        })
        
        pbar.close()
        
        images = np.array(images)
        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        
        return images, filenames
    
    def generate_training_data(self, images: np.ndarray, filenames: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data with labels based on filenames."""
        logger.info("Generating training data with labels...")
        
        labels = []
        valid_images = []
        valid_filenames = []
        
        # Create progress bar for label generation
        pbar = tqdm(zip(images, filenames), desc="Generating labels", unit="image")
        
        for img, filename in pbar:
            # Extract class from filename (assuming format: class_name_*.jpg)
            class_name = filename.split('_')[0].lower()
            
            # Map class name to class index
            if class_name in self.class_names:
                class_idx = self.class_names.index(class_name)
                labels.append(class_idx)
                valid_images.append(img)
                valid_filenames.append(filename)
            else:
                # If class not found, assign to 'unknown' class
                unknown_idx = self.class_names.index('unknown')
                labels.append(unknown_idx)
                valid_images.append(img)
                valid_filenames.append(filename)
            
            # Update progress bar
            pbar.set_postfix({
                'Processed': len(valid_images),
                'Current': class_name
            })
        
        pbar.close()
        
        valid_images = np.array(valid_images)
        labels = np.array(labels)
        
        logger.info(f"Generated training data: {len(valid_images)} images, {len(set(labels))} classes")
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.class_names[label]
            logger.info(f"  {class_name}: {count} images")
        
        return valid_images, labels
    
    def create_classifier(self):
        """Create and initialize the classifier model."""
        self.classifier = AstronomicalObjectClassifier(
            num_classes=self.num_classes,
            input_channels=1
        ).to(self.device)
        logger.info("Classifier model created and initialized")
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        epochs: int = 50, batch_size: int = 32, 
                        learning_rate: float = 1e-3, 
                        save_path: str = "models/astronomical_classifier.pth"):
        """Train the astronomical object classifier."""
        logger.info("Starting classifier training...")
        
        # Create model if not exists
        if self.classifier is None:
            self.create_classifier()
        
        # Prepare data
        if X_train.shape[-1] == 1:
            X_train = np.transpose(X_train, (0, 3, 1, 2))
        
        train_tensor = torch.tensor(X_train.astype(np.float32))
        train_labels = torch.tensor(y_train.astype(np.int64))
        train_dataset = TensorDataset(train_tensor, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Training loop
        self.classifier.train()
        losses = []
        accuracies = []
        start_time = time.time()
        
        epoch_pbar = tqdm(range(epochs), desc="Training Classifier", unit="epoch")
        
        for epoch in epoch_pbar:
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                             leave=False, unit="batch")
            
            for batch_idx, (images, labels) in enumerate(batch_pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classifier(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
                })
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct_predictions / total_predictions
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Loss': f'{avg_loss:.6f}',
                'Acc': f'{accuracy:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                          f"Accuracy: {accuracy:.2f}%")
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Final loss: {losses[-1]:.6f}")
        logger.info(f"Final accuracy: {accuracies[-1]:.2f}%")
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.classifier.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'training_history': {
                'losses': losses,
                'accuracies': accuracies
            }
        }, save_path)
        logger.info(f"Classifier saved to {save_path}")
        
        return {'losses': losses, 'accuracies': accuracies}
    
    def classify_objects(self, X_test: np.ndarray, confidence_threshold: float = 0.7) -> Dict:
        """Classify astronomical objects in test images."""
        logger.info("Classifying astronomical objects...")
        
        # Try to load existing model if classifier is None
        if self.classifier is None:
            model_path = "models/astronomical_classifier.pth"
            if os.path.exists(model_path):
                logger.info(f"Loading existing classifier from {model_path}")
                self.load_classifier(model_path)
            else:
                raise ValueError("Classifier not trained. Please train the classifier first.")
        
        # Prepare data
        if X_test.shape[-1] == 1:
            X_test = np.transpose(X_test, (0, 3, 1, 2))
        
        self.classifier.eval()
        test_tensor = torch.tensor(X_test.astype(np.float32))
        
        predictions = []
        confidences = []
        class_probabilities = []
        
        with torch.no_grad():
            for i in tqdm(range(len(test_tensor)), desc="Classifying objects"):
                image = test_tensor[i:i+1].to(self.device)
                outputs = self.classifier(image)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get prediction and confidence
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                # Determine if object is known or unknown
                if confidence_score >= confidence_threshold:
                    object_type = "known"
                    classification = self.class_names[predicted_class]
                else:
                    object_type = "unknown"
                    classification = "unknown"
                
                predictions.append({
                    'index': i,
                    'predicted_class': predicted_class,
                    'classification': classification,
                    'confidence': confidence_score,
                    'object_type': object_type,
                    'probabilities': probabilities.cpu().numpy()[0].tolist()
                })
                
                confidences.append(confidence_score)
                class_probabilities.append(probabilities.cpu().numpy()[0])
        
        # Compile results
        results = {
            'predictions': predictions,
            'confidences': confidences,
            'class_probabilities': class_probabilities,
            'known_objects': [p for p in predictions if p['object_type'] == 'known'],
            'unknown_objects': [p for p in predictions if p['object_type'] == 'unknown'],
            'confidence_threshold': confidence_threshold,
            'total_objects': len(predictions),
            'known_count': len([p for p in predictions if p['object_type'] == 'known']),
            'unknown_count': len([p for p in predictions if p['object_type'] == 'unknown'])
        }
        
        logger.info(f"Classification completed:")
        logger.info(f"  Total objects: {results['total_objects']}")
        logger.info(f"  Known objects: {results['known_count']}")
        logger.info(f"  Unknown objects: {results['unknown_count']}")
        
        return results
    
    def load_classifier(self, model_path: str):
        """Load a trained classifier."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create classifier if not exists
        if self.classifier is None:
            self.classifier = AstronomicalObjectClassifier(
                num_classes=checkpoint['num_classes'],
                input_channels=1
            ).to(self.device)
        
        # Load state dict
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        # Update class names
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        
        logger.info(f"Classifier loaded from {model_path}")
        logger.info(f"Classes: {self.class_names}")
    
    def export_classifications(self, X_test: np.ndarray, results: Dict, 
                             filenames: Optional[List[str]] = None,
                             output_dir: str = "classification_export") -> List[str]:
        """Export classification results with metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        predictions = results['predictions']
        
        # Create progress bar for export
        pbar = tqdm(predictions, desc="Exporting classifications", unit="object")
        
        for pred in pbar:
            idx = pred['index']
            classification = pred['classification']
            confidence = pred['confidence']
            object_type = pred['object_type']
            
            # Create filename
            if filenames and idx < len(filenames):
                base_name = Path(filenames[idx]).stem
            else:
                base_name = f"object_{idx:04d}"
            
            filename = f"{base_name}_{classification}_conf{confidence:.3f}_{object_type}.png"
            out_path = os.path.join(output_dir, filename)
            
            # Save image
            img = (X_test[idx][0] * 255).astype('uint8')
            cv2.imwrite(out_path, img)
            exported_files.append(out_path)
            
            # Update progress bar
            pbar.set_postfix({
                'Exported': len(exported_files),
                'Current': filename[:20] + '...' if len(filename) > 20 else filename
            })
        
        pbar.close()
        
        # Save results metadata
        metadata_path = os.path.join(output_dir, "classification_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(metadata_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Exported {len(exported_files)} classified objects to {output_dir}")
        logger.info(f"Results metadata saved to: {metadata_path}")
        
        return exported_files
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def generate_classification_report(self, results: Dict) -> str:
        """Generate a detailed classification report."""
        report = []
        report.append("=" * 60)
        report.append("ASTRONOMICAL OBJECT CLASSIFICATION REPORT")
        report.append("=" * 60)
        
        # Summary statistics
        report.append(f"\nSUMMARY:")
        report.append(f"  Total objects analyzed: {results['total_objects']}")
        report.append(f"  Known objects: {results['known_count']}")
        report.append(f"  Unknown objects: {results['unknown_count']}")
        report.append(f"  Confidence threshold: {results['confidence_threshold']}")
        
        # Known object breakdown
        if results['known_objects']:
            report.append(f"\nKNOWN OBJECTS ({len(results['known_objects'])}):")
            class_counts = {}
            for obj in results['known_objects']:
                class_name = obj['classification']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                report.append(f"  {class_name}: {count}")
        
        # Unknown objects
        if results['unknown_objects']:
            report.append(f"\nUNKNOWN OBJECTS ({len(results['unknown_objects'])}):")
            for obj in results['unknown_objects'][:10]:  # Show first 10
                report.append(f"  Object {obj['index']}: confidence {obj['confidence']:.3f}")
            if len(results['unknown_objects']) > 10:
                report.append(f"  ... and {len(results['unknown_objects']) - 10} more")
        
        # Confidence statistics
        confidences = results['confidences']
        report.append(f"\nCONFIDENCE STATISTICS:")
        report.append(f"  Average confidence: {np.mean(confidences):.3f}")
        report.append(f"  Median confidence: {np.median(confidences):.3f}")
        report.append(f"  Min confidence: {np.min(confidences):.3f}")
        report.append(f"  Max confidence: {np.max(confidences):.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def main():
    """Main function to demonstrate the astronomical classifier."""
    # Initialize the system
    classifier_system = AstronomicalClassificationSystem()
    
    # Create classifier
    classifier_system.create_classifier()
    
    print("Astronomical Object Classification System")
    print("=" * 50)
    print(f"Classes: {classifier_system.class_names}")
    print(f"Device: {classifier_system.device}")
    print("=" * 50)

if __name__ == "__main__":
    main() 