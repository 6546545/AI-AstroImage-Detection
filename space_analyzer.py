#!/usr/bin/env python3
"""
Space Anomaly Detection & Classification System
Command-Line Interface

This script provides a comprehensive command-line interface for:
- Anomaly detection in space imagery
- Astronomical object classification
- Combined analysis
- Model training
- Data preprocessing
- Results export
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set environment variables for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('space_analyzer.log')
        ]
    )

def anomaly_detection(args):
    """Run anomaly detection on images."""
    from space_anomaly_detector import AnomalyDetector, DataPreprocessor
    
    print("🔍 Running Anomaly Detection...")
    
    # Create detector
    detector = AnomalyDetector(device=args.device)
    
    # Load and preprocess images
    preprocessor = DataPreprocessor(img_size=(512, 512), grayscale=True)
    images, filenames = preprocessor.load_and_preprocess_images(args.input_dir)
    
    if len(images) == 0:
        print("❌ No images found in input directory")
        return False
    
    print(f"📊 Loaded {len(images)} images")
    
    # Train detector if needed
    if args.train:
        print("🏋️  Training anomaly detector...")
        detector.train(images, epochs=args.epochs)
        print("✅ Training completed")
    
    # Detect anomalies
    print("🔍 Detecting anomalies...")
    results = detector.detect_anomalies(images, confidence_threshold=args.threshold)
    
    # Export results
    if args.output_dir:
        print(f"📁 Exporting results to {args.output_dir}")
        detector.export_anomalies(images, results, filenames, args.output_dir)
    
    # Print summary
    high_confidence_anomalies = results['high_confidence_anomalies']
    all_anomalies = results['all_anomalies']
    print(f"\n📈 Detection Summary:")
    print(f"   Total images: {len(images)}")
    print(f"   High-confidence anomalies: {len(high_confidence_anomalies)}")
    print(f"   Total anomalies: {len(all_anomalies)}")
    print(f"   Anomaly rate: {len(all_anomalies)/len(images)*100:.1f}%")
    print(f"   Confidence threshold: {results['confidence_threshold']}")
    print(f"   Error threshold: {results['threshold']:.6f}")
    
    return True

def classification(args):
    """Run astronomical object classification."""
    from astronomical_classifier import AstronomicalClassificationSystem
    
    print("🌌 Running Astronomical Classification...")
    
    # Create classifier
    classifier = AstronomicalClassificationSystem(device=args.device)
    
    # Load images
    images, filenames = classifier.load_images_from_directory(args.input_dir)
    
    if len(images) == 0:
        print("❌ No images found in input directory")
        return False
    
    print(f"📊 Loaded {len(images)} images")
    
    # Train classifier if needed
    if args.train:
        print("🏋️  Training classifier...")
        # Generate training data with labels
        X_train, y_train = classifier.generate_training_data(images, filenames)
        classifier.train_classifier(X_train, y_train, epochs=args.epochs)
        print("✅ Training completed")
    
    # Classify objects
    print("🔍 Classifying objects...")
    results = classifier.classify_objects(images, confidence_threshold=args.threshold)
    
    # Export results
    if args.output_dir:
        print(f"📁 Exporting results to {args.output_dir}")
        classifier.export_classifications(images, results, filenames, args.output_dir)
    
    # Print summary
    classifications = results['predictions']
    print(f"\n📈 Classification Summary:")
    print(f"   Total objects: {len(classifications)}")
    
    # Count classifications
    class_counts = {}
    for pred in classifications:
        class_name = pred['classification']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count}")
    
    return True

def combined_analysis(args):
    """Run combined anomaly detection and classification."""
    from enhanced_space_system import EnhancedSpaceSystem
    
    print("🚀 Running Combined Space Analysis...")
    
    # Create enhanced system with custom data directory
    system = EnhancedSpaceSystem(data_dir=args.input_dir, device=args.device)
    
    # Run complete analysis
    results = system.run_complete_analysis(
        train_epochs=args.epochs,
        confidence_threshold=args.threshold
    )
    
    print("✅ Combined analysis completed")
    print(f"📊 Results summary:")
    print(f"   Total images: {results['summary']['total_images']}")
    print(f"   Anomalies detected: {results['summary']['anomalies_detected']}")
    print(f"   Known objects: {results['summary']['known_objects']}")
    print(f"   Unknown objects: {results['summary']['unknown_objects']}")
    
    return True

def train_models(args):
    """Train both anomaly detector and classifier."""
    print("🏋️  Training Models...")
    
    if args.model == 'anomaly' or args.model == 'both':
        print("Training anomaly detector...")
        from space_anomaly_detector import AnomalyDetector, DataPreprocessor
        
        detector = AnomalyDetector(device=args.device)
        preprocessor = DataPreprocessor(img_size=(512, 512), grayscale=True)
        images, _ = preprocessor.load_and_preprocess_images(args.input_dir)
        
        if len(images) > 0:
            detector.train(images, epochs=args.epochs)
            print("✅ Anomaly detector trained")
        else:
            print("❌ No images found for training")
    
    if args.model == 'classifier' or args.model == 'both':
        print("Training classifier...")
        from astronomical_classifier import AstronomicalClassificationSystem
        
        classifier = AstronomicalClassificationSystem(device=args.device)
        images, filenames = classifier.load_images_from_directory(args.input_dir)
        
        if len(images) > 0:
            # Generate training data with labels
            X_train, y_train = classifier.generate_training_data(images, filenames)
            classifier.train_classifier(X_train, y_train, epochs=args.epochs)
            print("✅ Classifier trained")
        else:
            print("❌ No images found for training")
    
    print("✅ Training completed")
    return True

def test_system(args):
    """Test the system with available data."""
    print("🧪 Testing System...")
    
    # Test imports
    try:
        from space_anomaly_detector import AnomalyDetector
        from astronomical_classifier import AstronomicalClassificationSystem
        from enhanced_space_system import EnhancedSpaceSystem
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data availability
    test_path = Path("test_dataset/images")
    if test_path.exists():
        image_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        print(f"✅ Found {len(image_files)} test images")
    else:
        print("⚠️  Test dataset not found")
    
    # Test model creation
    try:
        detector = AnomalyDetector(device=args.device)
        classifier = AstronomicalClassificationSystem(device=args.device)
        print("✅ Models created successfully")
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False
    
    print("✅ System test passed")
    return True

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Space Anomaly Detection & Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run anomaly detection
  python space_analyzer.py detect --input-dir images/ --output-dir results/
  
  # Run classification
  python space_analyzer.py classify --input-dir images/ --output-dir results/
  
  # Run combined analysis
  python space_analyzer.py analyze --input-dir images/ --output-dir results/
  
  # Train models
  python space_analyzer.py train --input-dir images/ --model both --epochs 50
  
  # Test system
  python space_analyzer.py test
        """
    )
    
    # Global arguments
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for computation (default: cpu)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Anomaly detection
    detect_parser = subparsers.add_parser('detect', help='Run anomaly detection')
    detect_parser.add_argument('--input-dir', required=True, help='Input directory with images')
    detect_parser.add_argument('--output-dir', help='Output directory for results')
    detect_parser.add_argument('--train', action='store_true', help='Train the detector')
    detect_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    detect_parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
    
    # Classification
    classify_parser = subparsers.add_parser('classify', help='Run object classification')
    classify_parser.add_argument('--input-dir', required=True, help='Input directory with images')
    classify_parser.add_argument('--output-dir', help='Output directory for results')
    classify_parser.add_argument('--train', action='store_true', help='Train the classifier')
    classify_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    classify_parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
    
    # Combined analysis
    analyze_parser = subparsers.add_parser('analyze', help='Run combined analysis')
    analyze_parser.add_argument('--input-dir', required=True, help='Input directory with images')
    analyze_parser.add_argument('--output-dir', help='Output directory for results')
    analyze_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    analyze_parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--input-dir', required=True, help='Input directory with images')
    train_parser.add_argument('--model', choices=['anomaly', 'classifier', 'both'], 
                             default='both', help='Which model to train')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    # Testing
    test_parser = subparsers.add_parser('test', help='Test the system')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'detect':
            success = anomaly_detection(args)
        elif args.command == 'classify':
            success = classification(args)
        elif args.command == 'analyze':
            success = combined_analysis(args)
        elif args.command == 'train':
            success = train_models(args)
        elif args.command == 'test':
            success = test_system(args)
        else:
            print(f"Unknown command: {args.command}")
            return
        
        if success:
            print("\n✅ Operation completed successfully!")
        else:
            print("\n❌ Operation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 