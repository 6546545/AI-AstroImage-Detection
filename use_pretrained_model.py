#!/usr/bin/env python3
"""
Use Pre-trained Model for Space Anomaly Detection
This script allows you to use the trained model to detect anomalies in new images.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from space_anomaly_detector import AnomalyDetector, DataPreprocessor
from config import *

def load_and_detect_anomalies(image_dir, confidence_threshold=0.8, output_dir="new_anomalies"):
    """
    Load images from a directory and detect anomalies using the pre-trained model.
    
    Args:
        image_dir (str): Directory containing images to analyze
        confidence_threshold (float): Minimum confidence for high-confidence anomalies
        output_dir (str): Directory to save anomaly results
    
    Returns:
        dict: Results containing anomaly information
    """
    print(f"🔍 Loading images from: {image_dir}")
    
    # Initialize preprocessor and detector
    preprocessor = DataPreprocessor(
        img_size=DATA_CONFIG['image_size'],
        grayscale=DATA_CONFIG['grayscale']
    )
    
    detector = AnomalyDetector(
        model_path=MODEL_CONFIG['model_save_path'],
        device=SYSTEM_CONFIG['device']
    )
    
    # Load and preprocess images
    try:
        X_images, filenames = preprocessor.load_and_preprocess_images(image_dir)
        print(f"✅ Loaded {len(X_images)} images")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return None
    
    # Detect anomalies
    print(f"🔍 Detecting anomalies with confidence threshold: {confidence_threshold}")
    results = detector.detect_anomalies(
        X_images, 
        confidence_threshold=confidence_threshold,
        error_percentile=DETECTION_CONFIG['error_percentile']
    )
    
    # Export anomalies
    print(f"📁 Exporting results to: {output_dir}")
    exported_files = detector.export_anomalies(
        X_images, results, filenames, output_dir
    )
    
    # Compile results
    final_results = {
        'total_images_processed': len(X_images),
        'high_confidence_anomalies': len(results['high_confidence_anomalies']),
        'total_anomalies': len(results['all_anomalies']),
        'confidence_threshold': confidence_threshold,
        'error_threshold': results['threshold'],
        'exported_files': exported_files,
        'anomaly_details': []
    }
    
    # Add details for each anomaly
    for idx in results['all_anomalies']:
        anomaly_detail = {
            'index': idx,
            'filename': filenames[idx] if idx < len(filenames) else f"image_{idx}",
            'error': results['errors'][idx],
            'confidence': results['confidences'][idx],
            'is_high_confidence': idx in results['high_confidence_anomalies']
        }
        final_results['anomaly_details'].append(anomaly_detail)
    
    return final_results

def print_results(results):
    """Print formatted results."""
    print("\n" + "="*60)
    print("SPACE ANOMALY DETECTION RESULTS")
    print("="*60)
    print(f"📊 Total images processed: {results['total_images_processed']}")
    print(f"🚨 High-confidence anomalies: {results['high_confidence_anomalies']}")
    print(f"⚠️  Total anomalies detected: {results['total_anomalies']}")
    print(f"🎯 Confidence threshold: {results['confidence_threshold']}")
    print(f"📏 Error threshold: {results['error_threshold']:.6f}")
    print(f"📁 Exported files: {len(results['exported_files'])}")
    
    if results['anomaly_details']:
        print("\n📋 Anomaly Details:")
        for detail in results['anomaly_details']:
            status = "🔴 HIGH CONFIDENCE" if detail['is_high_confidence'] else "🟡 ANOMALY"
            print(f"  {status}: {detail['filename']}")
            print(f"    Error: {detail['error']:.6f}, Confidence: {detail['confidence']:.3f}")
    
    print("="*60)

def save_results_json(results, output_dir):
    """Save results to JSON file."""
    results_file = os.path.join(output_dir, "detection_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {results_file}")

def create_visualization(results, output_dir):
    """Create visualization of error distribution."""
    if not results['anomaly_details']:
        print("📊 No anomalies to visualize")
        return
    
    errors = [detail['error'] for detail in results['anomaly_details']]
    confidences = [detail['confidence'] for detail in results['anomaly_details']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error distribution
    ax1.hist(errors, bins=20, alpha=0.7, color='blue')
    ax1.axvline(results['error_threshold'], color='red', linestyle='--', 
                label=f'Threshold: {results["error_threshold"]:.5f}')
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    
    # Confidence distribution
    ax2.hist(confidences, bins=20, alpha=0.7, color='green')
    ax2.axvline(results['confidence_threshold'], color='red', linestyle='--', 
                label=f'Confidence Threshold: {results["confidence_threshold"]}')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    
    plt.tight_layout()
    viz_file = os.path.join(output_dir, "detection_visualization.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved to: {viz_file}")

def main():
    parser = argparse.ArgumentParser(description="Use pre-trained model for space anomaly detection")
    parser.add_argument('image_dir', help='Directory containing images to analyze')
    parser.add_argument('--confidence', '-c', type=float, default=0.8, 
                       help='Confidence threshold (default: 0.8)')
    parser.add_argument('--output', '-o', default='new_anomalies', 
                       help='Output directory for results (default: new_anomalies)')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: Image directory '{args.image_dir}' not found!")
        return
    
    # Check if model exists
    if not os.path.exists(MODEL_CONFIG['model_save_path']):
        print(f"❌ Error: Model file '{MODEL_CONFIG['model_save_path']}' not found!")
        print("Please run the training pipeline first or check the model path.")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"🚀 Starting anomaly detection...")
    print(f"📁 Input directory: {args.image_dir}")
    print(f"🎯 Confidence threshold: {args.confidence}")
    print(f"📁 Output directory: {args.output}")
    
    # Run detection
    results = load_and_detect_anomalies(
        args.image_dir, 
        confidence_threshold=args.confidence,
        output_dir=args.output
    )
    
    if results is None:
        print("❌ Detection failed!")
        return
    
    # Print and save results
    print_results(results)
    save_results_json(results, args.output)
    
    # Create visualization
    if not args.no_viz:
        create_visualization(results, args.output)
    
    print(f"\n✅ Detection complete! Check '{args.output}' directory for results.")

if __name__ == "__main__":
    main() 