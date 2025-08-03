#!/usr/bin/env python3
"""
Example: Using the Pre-trained Model for Space Anomaly Detection
This script demonstrates how to use the trained model programmatically.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from space_anomaly_detector import AnomalyDetector, DataPreprocessor
from config import *

def example_single_image_detection():
    """Example: Detect anomalies in a single image."""
    print("🔍 Example 1: Single Image Detection")
    print("-" * 40)
    
    # Initialize detector
    detector = AnomalyDetector(model_path=MODEL_CONFIG['model_save_path'])
    
    # Load a single image (using first image from sdss_images as example)
    preprocessor = DataPreprocessor()
    X_images, filenames = preprocessor.load_and_preprocess_images("sdss_images")
    
    if len(X_images) > 0:
        # Use first image
        single_image = X_images[0:1]  # Keep batch dimension
        filename = filenames[0]
        
        print(f"📸 Analyzing image: {filename}")
        
        # Detect anomalies
        results = detector.detect_anomalies(single_image, confidence_threshold=0.8)
        
        # Check results
        is_anomaly = len(results['high_confidence_anomalies']) > 0
        error = results['errors'][0]
        confidence = results['confidences'][0]
        
        print(f"🚨 Anomaly detected: {is_anomaly}")
        print(f"📊 Error: {error:.6f}")
        print(f"🎯 Confidence: {confidence:.3f}")
        print(f"📏 Threshold: {results['threshold']:.6f}")
    
    print()

def example_batch_detection():
    """Example: Detect anomalies in multiple images."""
    print("🔍 Example 2: Batch Detection")
    print("-" * 40)
    
    # Initialize components
    detector = AnomalyDetector(model_path=MODEL_CONFIG['model_save_path'])
    preprocessor = DataPreprocessor()
    
    # Load images
    X_images, filenames = preprocessor.load_and_preprocess_images("sdss_images")
    
    print(f"📸 Loaded {len(X_images)} images")
    
    # Detect anomalies
    results = detector.detect_anomalies(X_images, confidence_threshold=0.8)
    
    # Analyze results
    high_conf_anomalies = len(results['high_confidence_anomalies'])
    total_anomalies = len(results['all_anomalies'])
    
    print(f"🚨 High-confidence anomalies: {high_conf_anomalies}")
    print(f"⚠️  Total anomalies: {total_anomalies}")
    print(f"📊 Anomaly rate: {total_anomalies/len(X_images)*100:.1f}%")
    
    # Show details for each anomaly
    if results['all_anomalies']:
        print("\n📋 Anomaly Details:")
        for idx in results['all_anomalies']:
            filename = filenames[idx] if idx < len(filenames) else f"image_{idx}"
            error = results['errors'][idx]
            confidence = results['confidences'][idx]
            is_high_conf = idx in results['high_confidence_anomalies']
            
            status = "🔴 HIGH CONFIDENCE" if is_high_conf else "🟡 ANOMALY"
            print(f"  {status}: {filename}")
            print(f"    Error: {error:.6f}, Confidence: {confidence:.3f}")
    
    print()

def example_custom_threshold():
    """Example: Using different confidence thresholds."""
    print("🔍 Example 3: Custom Thresholds")
    print("-" * 40)
    
    detector = AnomalyDetector(model_path=MODEL_CONFIG['model_save_path'])
    preprocessor = DataPreprocessor()
    
    # Load images
    X_images, filenames = preprocessor.load_and_preprocess_images("sdss_images")
    
    # Test different confidence thresholds
    thresholds = [0.6, 0.7, 0.8, 0.9]
    
    print(f"📊 Testing {len(thresholds)} confidence thresholds on {len(X_images)} images:")
    print()
    
    for threshold in thresholds:
        results = detector.detect_anomalies(X_images, confidence_threshold=threshold)
        high_conf = len(results['high_confidence_anomalies'])
        total = len(results['all_anomalies'])
        
        print(f"🎯 Threshold {threshold}: {high_conf} high-conf, {total} total anomalies")
    
    print()

def example_export_anomalies():
    """Example: Export detected anomalies."""
    print("🔍 Example 4: Export Anomalies")
    print("-" * 40)
    
    detector = AnomalyDetector(model_path=MODEL_CONFIG['model_save_path'])
    preprocessor = DataPreprocessor()
    
    # Load images
    X_images, filenames = preprocessor.load_and_preprocess_images("sdss_images")
    
    # Detect anomalies
    results = detector.detect_anomalies(X_images, confidence_threshold=0.8)
    
    # Export anomalies
    output_dir = "example_anomalies"
    exported_files = detector.export_anomalies(X_images, results, filenames, output_dir)
    
    print(f"📁 Exported {len(exported_files)} anomaly images to '{output_dir}'")
    
    if exported_files:
        print("📋 Exported files:")
        for file in exported_files:
            print(f"  - {file}")
    
    print()

def main():
    """Run all examples."""
    print("🚀 Space Anomaly Detection - Example Usage")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(MODEL_CONFIG['model_save_path']):
        print(f"❌ Error: Model file '{MODEL_CONFIG['model_save_path']}' not found!")
        print("Please run the training pipeline first.")
        return
    
    # Check if images exist
    if not os.path.exists("sdss_images"):
        print("❌ Error: 'sdss_images' directory not found!")
        print("Please ensure you have images to analyze.")
        return
    
    # Run examples
    example_single_image_detection()
    example_batch_detection()
    example_custom_threshold()
    example_export_anomalies()
    
    print("✅ All examples completed!")

if __name__ == "__main__":
    main() 