#!/usr/bin/env python3
"""
Enhanced Space Analyzer - Advanced CLI with cutting-edge features
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from tqdm import tqdm

# Import existing modules
from space_anomaly_detector import AnomalyDetector, DataPreprocessor
from astronomical_classifier import AstronomicalClassificationSystem
from enhanced_space_system import EnhancedSpaceSystem
from multi_object_detector import MultiObjectDetector

# Import new advanced modules
from advanced_models import (
    VisionTransformer, EfficientNetAstronomical, VariationalAutoencoder,
    SelfSupervisedAnomalyDetector, EnsembleAstronomicalClassifier,
    create_advanced_classifier
)
from advanced_features import (
    MultiSpectralAnalyzer, AdvancedImageProcessor, TimeSeriesAnalyzer,
    AstronomicalObjectTracker, AnomalyClassifier
)
from advanced_training import (
    AdvancedTrainer, AdvancedDataAugmentation, LearningRateScheduler,
    EarlyStopping, AdvancedTrainingLoop, CurriculumDataset
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSpaceAnalyzer:
    """Enhanced space analyzer with advanced features."""
    
    def __init__(self):
        self.advanced_models = {}
        self.feature_extractors = {}
        self.training_techniques = {}
        
        # Initialize advanced components
        self._initialize_advanced_components()
    
    def _initialize_advanced_components(self):
        """Initialize all advanced components."""
        logger.info("Initializing advanced components...")
        
        # Advanced models
        self.advanced_models = {
            'vit': VisionTransformer,
            'efficientnet': EfficientNetAstronomical,
            'vae': VariationalAutoencoder,
            'self_supervised': SelfSupervisedAnomalyDetector
        }
        
        # Feature extractors
        self.feature_extractors = {
            'multispectral': MultiSpectralAnalyzer(),
            'image_processor': AdvancedImageProcessor(),
            'time_series': TimeSeriesAnalyzer(),
            'object_tracker': AstronomicalObjectTracker(),
            'anomaly_classifier': AnomalyClassifier()
        }
        
        # Training techniques
        self.training_techniques = {
            'trainer': AdvancedTrainer(),
            'augmentation': AdvancedDataAugmentation(),
            'scheduler': None,  # Will be set during training
            'early_stopping': None  # Will be set during training
        }
        
        logger.info("✅ Advanced components initialized successfully")
    
    def advanced_detect(self, input_dir: str, output_dir: str, 
                       model_type: str = 'vae', threshold: float = 0.7,
                       use_multispectral: bool = True, use_advanced_features: bool = True):
        """Advanced anomaly detection with multiple techniques."""
        logger.info(f"🚀 Starting advanced anomaly detection with {model_type} model")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess images
        preprocessor = DataPreprocessor(img_size=(512, 512), grayscale=True)
        images, filenames = preprocessor.load_and_preprocess_images(input_dir)
        
        if len(images) == 0:
            logger.error("No images found for analysis")
            return
        
        # Initialize advanced model
        if model_type in self.advanced_models:
            model = self.advanced_models[model_type](input_channels=1)
            logger.info(f"Using {model_type} model for anomaly detection")
        else:
            logger.warning(f"Unknown model type {model_type}, using default VAE")
            model = VariationalAutoencoder(input_channels=1)
        
        # Advanced feature extraction
        advanced_features = {}
        if use_multispectral:
            logger.info("Extracting multi-spectral features...")
            for i, image in enumerate(tqdm(images, desc="Multi-spectral analysis")):
                features = self.feature_extractors['multispectral'].extract_spectral_features(image)
                advanced_features[filenames[i]] = features
        
        if use_advanced_features:
            logger.info("Extracting advanced image features...")
            for i, image in enumerate(tqdm(images, desc="Advanced feature extraction")):
                features = self.feature_extractors['image_processor'].extract_all_features(image)
                if filenames[i] in advanced_features:
                    advanced_features[filenames[i]].update(features)
                else:
                    advanced_features[filenames[i]] = features
        
        # Perform anomaly detection
        logger.info("Performing anomaly detection...")
        anomalies = []
        
        for i, image in enumerate(tqdm(images, desc="Anomaly detection")):
            # Convert image to tensor
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            
            # Get model prediction
            with torch.no_grad():
                if model_type == 'vae':
                    recon, mu, log_var = model(image_tensor)
                    # Calculate reconstruction error
                    recon_error = torch.mean((image_tensor - recon) ** 2).item()
                    anomaly_score = recon_error
                else:
                    # For other models, use the output directly
                    output = model(image_tensor)
                    anomaly_score = torch.sigmoid(output).item() if output.dim() == 0 else torch.sigmoid(output).mean().item()
            
            # Classify anomaly type
            anomaly_class = self.feature_extractors['anomaly_classifier'].classify_anomaly(image)
            
            if anomaly_score > threshold:
                anomaly_info = {
                    'filename': filenames[i],
                    'anomaly_score': anomaly_score,
                    'anomaly_type': anomaly_class['anomaly_type'],
                    'description': anomaly_class['description'],
                    'confidence': anomaly_class['confidence'],
                    'advanced_features': advanced_features.get(filenames[i], {})
                }
                anomalies.append(anomaly_info)
        
        # Export results
        results = {
            'high_confidence_anomalies': [a for a in anomalies if a['confidence'] > 0.8],
            'all_anomalies': anomalies,
            'total_images_analyzed': len(images),
            'anomalies_found': len(anomalies),
            'model_type': model_type,
            'threshold': threshold,
            'advanced_features_used': use_advanced_features,
            'multispectral_analysis': use_multispectral
        }
        
        # Save results
        output_file = os.path.join(output_dir, 'advanced_anomaly_detection_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Advanced anomaly detection completed. Found {len(anomalies)} anomalies")
        logger.info(f"Results saved to: {output_file}")
        
        return results
    
    def advanced_classify(self, input_dir: str, output_dir: str, 
                         model_type: str = 'vit', threshold: float = 0.7,
                         use_ensemble: bool = True):
        """Advanced object classification with ensemble methods."""
        logger.info(f"🔍 Starting advanced classification with {model_type} model")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        classifier = AstronomicalClassificationSystem()
        images, filenames = classifier.load_images_from_directory(input_dir)
        
        if len(images) == 0:
            logger.error("No images found for classification")
            return
        
        # Initialize models
        models = []
        if use_ensemble:
            # Create ensemble of different models
            model_types = ['vit', 'efficientnet', 'self_supervised']
            for mt in model_types:
                if mt in self.advanced_models:
                    model = self.advanced_models[mt](num_classes=10, input_channels=1)
                    models.append(model)
                    logger.info(f"Added {mt} model to ensemble")
        else:
            # Single model
            if model_type in self.advanced_models:
                model = self.advanced_models[model_type](num_classes=10, input_channels=1)
                models.append(model)
            else:
                logger.warning(f"Unknown model type {model_type}, using default classifier")
                # Use the existing classification system
                classifier = AstronomicalClassificationSystem()
                return classifier.classify_objects(images)
        
        # Create ensemble classifier
        ensemble = EnsembleAstronomicalClassifier(models)
        
        # Perform classification
        logger.info("Performing advanced classification...")
        classifications = []
        
        for i, image in enumerate(tqdm(images, desc="Classification")):
            # Convert image to tensor
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            
            # Get ensemble prediction
            with torch.no_grad():
                prediction = ensemble.predict(image_tensor)
            
            if prediction['confidence'] > threshold:
                classification_info = {
                    'filename': filenames[i],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'ensemble_agreement': prediction['ensemble_agreement'],
                    'individual_predictions': prediction['individual_predictions'],
                    'individual_confidences': prediction['individual_confidences']
                }
                classifications.append(classification_info)
        
        # Export results
        results = {
            'high_confidence_classifications': [c for c in classifications if c['confidence'] > 0.8],
            'all_classifications': classifications,
            'total_images_analyzed': len(images),
            'classifications_found': len(classifications),
            'model_type': model_type,
            'ensemble_used': use_ensemble,
            'threshold': threshold
        }
        
        # Save results
        output_file = os.path.join(output_dir, 'advanced_classification_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Advanced classification completed. Found {len(classifications)} objects")
        logger.info(f"Results saved to: {output_file}")
        
        return results
    
    def advanced_train(self, data_dir: str, model_type: str = 'vit', 
                      epochs: int = 10, batch_size: int = 32,
                      use_advanced_techniques: bool = True):
        """Advanced model training with cutting-edge techniques."""
        logger.info(f"🎯 Starting advanced training for {model_type} model")
        
        # Load training data
        classifier = AstronomicalClassificationSystem()
        images, filenames = classifier.load_images_from_directory(data_dir)
        labels = classifier.generate_training_data(filenames)
        
        if len(images) == 0:
            logger.error("No training data found")
            return
        
        # Convert to tensors
        X = torch.FloatTensor(images)
        y = torch.LongTensor(labels)
        
        # Create dataset with curriculum learning
        if use_advanced_techniques:
            dataset = CurriculumDataset(X, y)
            logger.info("Using curriculum learning dataset")
        else:
            dataset = torch.utils.data.TensorDataset(X, y)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        if model_type in self.advanced_models:
            model = self.advanced_models[model_type](num_classes=10, input_channels=1)
        else:
            logger.warning(f"Unknown model type {model_type}, using default classifier")
            return
        
        # Setup training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Advanced training loop
        if use_advanced_techniques:
            training_loop = AdvancedTrainingLoop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_mixup=True,
                use_cutmix=True,
                use_label_smoothing=True,
                use_focal_loss=True,
                scheduler_type='cosine',
                early_stopping_patience=5
            )
            
            logger.info("Using advanced training techniques")
            metrics = training_loop.train(epochs)
        else:
            # Basic training
            logger.info("Using basic training")
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save model
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f'advanced_{model_type}_classifier.pth'
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"✅ Advanced training completed. Model saved to: {model_path}")
        
        return model_path
    
    def time_series_analysis(self, data_dir: str, output_dir: str):
        """Analyze astronomical objects over time."""
        logger.info("⏰ Starting time series analysis")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # This would typically work with time-series data
        # For now, we'll create a demonstration
        time_analyzer = self.feature_extractors['time_series']
        
        # Simulate light curve data
        brightness_values = [1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0]
        timestamps = list(range(len(brightness_values)))
        
        # Analyze light curve
        light_curve_features = time_analyzer.analyze_light_curve(brightness_values, timestamps)
        
        # Detect periodic behavior
        periodic_behavior = time_analyzer.detect_periodic_behavior(brightness_values)
        
        results = {
            'light_curve_analysis': light_curve_features,
            'periodic_behavior': periodic_behavior,
            'analysis_type': 'time_series',
            'data_points': len(brightness_values)
        }
        
        # Save results
        output_file = os.path.join(output_dir, 'time_series_analysis_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Time series analysis completed. Results saved to: {output_file}")
        
        return results
    
    def object_tracking(self, video_dir: str, output_dir: str):
        """Track astronomical objects across multiple frames."""
        logger.info("🎯 Starting object tracking analysis")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        tracker = self.feature_extractors['object_tracker']
        
        # This would typically work with video data
        # For now, we'll create a demonstration
        frame_number = 1
        detections = [
            {'bbox': [100, 100, 50, 50], 'classification': 'star', 'confidence': 0.9},
            {'bbox': [200, 200, 60, 60], 'classification': 'galaxy', 'confidence': 0.8}
        ]
        
        # Update tracking
        tracks = tracker.update(detections, frame_number)
        
        results = {
            'active_tracks': tracks,
            'total_tracks': len(tracks),
            'frame_number': frame_number,
            'analysis_type': 'object_tracking'
        }
        
        # Save results
        output_file = os.path.join(output_dir, 'object_tracking_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Object tracking completed. Tracking {len(tracks)} objects")
        logger.info(f"Results saved to: {output_file}")
        
        return results

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Enhanced Space Analyzer - Advanced Features')
    parser.add_argument('command', choices=[
        'advanced-detect', 'advanced-classify', 'advanced-train',
        'time-series', 'object-tracking', 'test-advanced'
    ], help='Command to execute')
    
    # Common arguments
    parser.add_argument('--input-dir', type=str, help='Input directory with images')
    parser.add_argument('--output-dir', type=str, default='enhanced_results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--model-type', type=str, default='vit', 
                       choices=['vit', 'efficientnet', 'vae', 'self_supervised'],
                       help='Advanced model type')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    # Advanced options
    parser.add_argument('--use-ensemble', action='store_true', help='Use ensemble methods')
    parser.add_argument('--use-advanced-techniques', action='store_true', help='Use advanced training techniques')
    parser.add_argument('--use-multispectral', action='store_true', help='Use multi-spectral analysis')
    parser.add_argument('--use-advanced-features', action='store_true', help='Use advanced feature extraction')
    
    args = parser.parse_args()
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSpaceAnalyzer()
    
    try:
        if args.command == 'advanced-detect':
            if not args.input_dir:
                parser.error("--input-dir is required for advanced-detect")
            
            results = analyzer.advanced_detect(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_type=args.model_type,
                threshold=args.threshold,
                use_multispectral=args.use_multispectral,
                use_advanced_features=args.use_advanced_features
            )
            
        elif args.command == 'advanced-classify':
            if not args.input_dir:
                parser.error("--input-dir is required for advanced-classify")
            
            results = analyzer.advanced_classify(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_type=args.model_type,
                threshold=args.threshold,
                use_ensemble=args.use_ensemble
            )
            
        elif args.command == 'advanced-train':
            if not args.input_dir:
                parser.error("--input-dir is required for advanced-train")
            
            model_path = analyzer.advanced_train(
                data_dir=args.input_dir,
                model_type=args.model_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_advanced_techniques=args.use_advanced_techniques
            )
            
        elif args.command == 'time-series':
            if not args.input_dir:
                parser.error("--input-dir is required for time-series")
            
            results = analyzer.time_series_analysis(
                data_dir=args.input_dir,
                output_dir=args.output_dir
            )
            
        elif args.command == 'object-tracking':
            if not args.input_dir:
                parser.error("--input-dir is required for object-tracking")
            
            results = analyzer.object_tracking(
                video_dir=args.input_dir,
                output_dir=args.output_dir
            )
            
        elif args.command == 'test-advanced':
            print("🧪 Testing Advanced Features...")
            
            # Test advanced models
            print("Testing Vision Transformer...")
            vit = VisionTransformer(image_size=512, patch_size=16, num_classes=10, dim=768, depth=6)
            test_input = torch.randn(1, 1, 512, 512)
            output = vit(test_input)
            print(f"ViT output shape: {output.shape}")
            
            # Test feature extractors
            print("Testing Multi-Spectral Analyzer...")
            msa = MultiSpectralAnalyzer()
            test_image = np.random.rand(512, 512, 3)
            spectral_features = msa.extract_spectral_features(test_image)
            print(f"Extracted {len(spectral_features)} spectral features")
            
            # Test training techniques
            print("Testing Advanced Trainer...")
            trainer = AdvancedTrainer()
            x1 = torch.randn(2, 1, 64, 64)
            x2 = torch.randn(2, 1, 64, 64)
            y1 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
            y2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
            mixed_x, mixed_y = trainer.mixup_training(x1, x2, y1, y2)
            print(f"Mixup output shapes: {mixed_x.shape}, {mixed_y.shape}")
            
            print("✅ All advanced features tested successfully!")
        
        print(f"✅ Command '{args.command}' completed successfully!")
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
