#!/usr/bin/env python3
"""
Smart entry point for astronomical classification
Automatically detects available data and uses appropriate classification mode
"""

import os
import logging
from pathlib import Path
import argparse
from adaptive_classifier import AdaptiveAstronomicalClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset(data_dir: str = "clean_dataset/processed_dataset"):
    """Analyze the available dataset and determine appropriate classification mode."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    train_dir = data_path / "train"
    available_classes = [d.name for d in train_dir.iterdir() if d.is_dir() and any(d.iterdir())]
    
    logger.info(f"📊 Found {len(available_classes)} classes with data:")
    for class_name in available_classes:
        class_dir = train_dir / class_name
        num_samples = len(list(class_dir.glob("*")))
        logger.info(f"   - {class_name}: {num_samples} samples")
    
    return available_classes

def main():
    parser = argparse.ArgumentParser(description="Smart Astronomical Classification")
    parser.add_argument("--mode", choices=["train", "classify"], required=True,
                      help="Operation mode: train or classify")
    parser.add_argument("--input", help="Input image or directory for classification")
    args = parser.parse_args()
    
    try:
        # Analyze available data
        logger.info("🔍 Analyzing dataset...")
        available_classes = analyze_dataset()
        
        # Initialize classifier
        classifier = AdaptiveAstronomicalClassification()
        classifier.initialize_classifier()
        
        if args.mode == "train":
            logger.info("🚀 Starting training...")
            # Training code will be implemented here
            pass
        
        elif args.mode == "classify":
            if not args.input:
                raise ValueError("--input is required for classification mode")
            
            input_path = Path(args.input)
            if input_path.is_file():
                # Classify single image
                results = classifier.classify_image(str(input_path))
                print("\n🔍 Classification Results:")
                for result in results:
                    print(f"   - {result['class_name']}: {result['confidence']:.2%}")
            
            elif input_path.is_dir():
                # Classify all images in directory
                for img_path in input_path.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        print(f"\n📸 Classifying {img_path.name}:")
                        results = classifier.classify_image(str(img_path))
                        for result in results:
                            print(f"   - {result['class_name']}: {result['confidence']:.2%}")
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
