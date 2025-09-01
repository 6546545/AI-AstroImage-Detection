#!/usr/bin/env python3
"""
Enhanced Space Anomaly Detection and Object Classification System
"""

import os
import numpy as np
import torch
import json
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2

# Import our existing systems
from space_anomaly_detector import SpaceAnomalyDetectionSystem, DataPreprocessor, AnomalyDetector
from astronomical_classifier import AstronomicalClassificationSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSpaceSystem:
    """Enhanced system combining anomaly detection and object classification."""
    
    def __init__(self, data_dir: str = "sdss_images",
                 anomaly_model_path: Optional[str] = None,
                 classifier_model_path: Optional[str] = None,
                 device: str = "auto"):
        
        self.data_dir = data_dir
        self.device = device
        
        # Initialize subsystems
        self.anomaly_system = SpaceAnomalyDetectionSystem(data_dir, anomaly_model_path)
        self.classifier_system = AstronomicalClassificationSystem(device)
        
        # Load classifier if model exists
        if classifier_model_path and os.path.exists(classifier_model_path):
            self.classifier_system.load_classifier(classifier_model_path)
        
        logger.info("Enhanced Space System initialized")
    
    def run_complete_analysis(self, confidence_threshold: float = 0.7,
                             train_epochs: int = 30) -> Dict:
        """Run complete analysis: anomaly detection + object classification."""
        logger.info("Starting complete space analysis...")
        
        # Create overall progress bar
        steps = [
            "Data Preprocessing",
            "Training Anomaly Detector",
            "Detecting Anomalies",
            "Classifying Objects",
            "Combining Results",
            "Exporting Results",
            "Generating Report"
        ]
        
        overall_pbar = tqdm(total=len(steps), desc="Complete Analysis Progress", unit="step")
        
        # Step 1: Data preprocessing
        logger.info("Step 1: Preprocessing data...")
        X_train, X_test, test_files = self.anomaly_system.preprocessor.prepare_dataset(self.data_dir)
        
        # Save preprocessed data
        self.anomaly_system.preprocessor.save_dataset(X_train, X_test,
                                                     [f"train_{i}" for i in range(len(X_train))],
                                                     test_files)
        overall_pbar.update(1)
        
        # Step 2: Train anomaly detector
        logger.info("Step 2: Training anomaly detector...")
        self.anomaly_system.detector.train(X_train, epochs=train_epochs)
        overall_pbar.update(1)
        
        # Step 3: Detect anomalies
        logger.info("Step 3: Detecting anomalies...")
        anomaly_results = self.anomaly_system.detector.detect_anomalies(X_test, confidence_threshold)
        overall_pbar.update(1)
        
        # Step 4: Classify objects
        logger.info("Step 4: Classifying astronomical objects...")
        classification_results = self.classifier_system.classify_objects(X_test, confidence_threshold)
        overall_pbar.update(1)
        
        # Step 5: Combine results
        logger.info("Step 5: Combining and analyzing results...")
        combined_results = self._combine_results(anomaly_results, classification_results, test_files)
        overall_pbar.update(1)
        
        # Step 6: Export results
        logger.info("Step 6: Exporting results...")
        exported_files = self._export_combined_results(X_test, combined_results, test_files)
        overall_pbar.update(1)
        
        # Step 7: Generate comprehensive report
        logger.info("Step 7: Generating comprehensive report...")
        report = self._generate_comprehensive_report(combined_results)
        overall_pbar.update(1)
        
        overall_pbar.close()
        
        # Compile final results
        final_results = {
            'anomaly_results': anomaly_results,
            'classification_results': classification_results,
            'combined_results': combined_results,
            'exported_files': exported_files,
            'report': report,
            'summary': {
                'total_images': len(X_test),
                'anomalies_detected': len(anomaly_results['high_confidence_anomalies']),
                'known_objects': combined_results['known_objects_count'],
                'unknown_objects': combined_results['unknown_objects_count'],
                'anomalous_known_objects': combined_results['anomalous_known_objects_count'],
                'anomalous_unknown_objects': combined_results['anomalous_unknown_objects_count']
            }
        }
        
        logger.info("Complete analysis finished!")
        logger.info(f"Results: {final_results['summary']}")
        
        return final_results
    
    def _combine_results(self, anomaly_results: Dict, classification_results: Dict,
                        filenames: List[str]) -> Dict:
        """Combine anomaly detection and classification results."""
        
        # Create mapping for quick lookup
        anomaly_indices = set(anomaly_results['high_confidence_anomalies'])
        
        combined_objects = []
        anomalous_known_objects = []
        anomalous_unknown_objects = []
        normal_known_objects = []
        normal_unknown_objects = []
        
        for pred in classification_results['predictions']:
            idx = pred['index']
            is_anomaly = idx in anomaly_indices
            object_type = pred['object_type']
            classification = pred['classification']
            
            # Create combined object entry
            combined_obj = {
                'index': idx,
                'filename': filenames[idx] if idx < len(filenames) else f"object_{idx}",
                'classification': classification,
                'confidence': pred['confidence'],
                'object_type': object_type,
                'is_anomaly': is_anomaly,
                'anomaly_error': anomaly_results['errors'][idx] if idx < len(anomaly_results['errors']) else None,
                'probabilities': pred['probabilities']
            }
            
            combined_objects.append(combined_obj)
            
            # Categorize object
            if is_anomaly:
                if object_type == "known":
                    anomalous_known_objects.append(combined_obj)
                else:
                    anomalous_unknown_objects.append(combined_obj)
            else:
                if object_type == "known":
                    normal_known_objects.append(combined_obj)
                else:
                    normal_unknown_objects.append(combined_obj)
        
        # Compile combined results
        combined_results = {
            'combined_objects': combined_objects,
            'anomalous_known_objects': anomalous_known_objects,
            'anomalous_unknown_objects': anomalous_unknown_objects,
            'normal_known_objects': normal_known_objects,
            'normal_unknown_objects': normal_unknown_objects,
            'anomalous_known_objects_count': len(anomalous_known_objects),
            'anomalous_unknown_objects_count': len(anomalous_unknown_objects),
            'normal_known_objects_count': len(normal_known_objects),
            'normal_unknown_objects_count': len(normal_unknown_objects),
            'known_objects_count': len(anomalous_known_objects) + len(normal_known_objects),
            'unknown_objects_count': len(anomalous_unknown_objects) + len(normal_unknown_objects),
            'total_anomalies': len(anomaly_results['high_confidence_anomalies']),
            'total_objects': len(combined_objects)
        }
        
        return combined_results
    
    def _export_combined_results(self, X_test: np.ndarray, combined_results: Dict,
                                filenames: List[str]) -> List[str]:
        """Export combined analysis results."""
        output_dir = "enhanced_analysis_export"
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        combined_objects = combined_results['combined_objects']
        
        # Create progress bar for export
        pbar = tqdm(combined_objects, desc="Exporting combined results", unit="object")
        
        for obj in pbar:
            idx = obj['index']
            classification = obj['classification']
            confidence = obj['confidence']
            object_type = obj['object_type']
            is_anomaly = obj['is_anomaly']
            anomaly_error = obj['anomaly_error']
            
            # Create descriptive filename
            base_name = Path(obj['filename']).stem
            anomaly_flag = "ANOMALY" if is_anomaly else "NORMAL"
            
            filename = f"{base_name}_{classification}_conf{confidence:.3f}_{object_type}_{anomaly_flag}.png"
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
        
        # Save combined results metadata
        metadata_path = os.path.join(output_dir, "enhanced_analysis_results.json")
        with open(metadata_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"Exported {len(exported_files)} analyzed objects to {output_dir}")
        logger.info(f"Results metadata saved to: {metadata_path}")
        
        return exported_files
    
    def _generate_comprehensive_report(self, combined_results: Dict) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED SPACE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        report.append(f"\nSUMMARY STATISTICS:")
        report.append(f"  Total objects analyzed: {combined_results['total_objects']}")
        report.append(f"  Total anomalies detected: {combined_results['total_anomalies']}")
        report.append(f"  Known objects: {combined_results['known_objects_count']}")
        report.append(f"  Unknown objects: {combined_results['unknown_objects_count']}")
        
        # Detailed breakdown
        report.append(f"\nDETAILED BREAKDOWN:")
        report.append(f"  Anomalous known objects: {combined_results['anomalous_known_objects_count']}")
        report.append(f"  Anomalous unknown objects: {combined_results['anomalous_unknown_objects_count']}")
        report.append(f"  Normal known objects: {combined_results['normal_known_objects_count']}")
        report.append(f"  Normal unknown objects: {combined_results['normal_unknown_objects_count']}")
        
        # Anomalous known objects
        if combined_results['anomalous_known_objects']:
            report.append(f"\nANOMALOUS KNOWN OBJECTS ({len(combined_results['anomalous_known_objects'])}):")
            class_counts = {}
            for obj in combined_results['anomalous_known_objects']:
                class_name = obj['classification']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                report.append(f"  {class_name}: {count}")
        
        # Anomalous unknown objects
        if combined_results['anomalous_unknown_objects']:
            report.append(f"\nANOMALOUS UNKNOWN OBJECTS ({len(combined_results['anomalous_unknown_objects'])}):")
            for obj in combined_results['anomalous_unknown_objects'][:10]:  # Show first 10
                report.append(f"  {obj['filename']}: confidence {obj['confidence']:.3f}, error {obj['anomaly_error']:.5f}")
            if len(combined_results['anomalous_unknown_objects']) > 10:
                report.append(f"  ... and {len(combined_results['anomalous_unknown_objects']) - 10} more")
        
        # Normal known objects
        if combined_results['normal_known_objects']:
            report.append(f"\nNORMAL KNOWN OBJECTS ({len(combined_results['normal_known_objects'])}):")
            class_counts = {}
            for obj in combined_results['normal_known_objects']:
                class_name = obj['classification']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                report.append(f"  {class_name}: {count}")
        
        # Key findings
        report.append(f"\nKEY FINDINGS:")
        
        # Calculate percentages
        total_objects = combined_results['total_objects']
        total_anomalies = combined_results['total_anomalies']
        known_objects = combined_results['known_objects_count']
        unknown_objects = combined_results['unknown_objects_count']
        
        anomaly_percentage = (total_anomalies / total_objects * 100) if total_objects > 0 else 0
        unknown_percentage = (unknown_objects / total_objects * 100) if total_objects > 0 else 0
        
        report.append(f"  Anomaly rate: {anomaly_percentage:.1f}%")
        report.append(f"  Unknown object rate: {unknown_percentage:.1f}%")
        
        if combined_results['anomalous_unknown_objects_count'] > 0:
            report.append(f"  ⚠️  {combined_results['anomalous_unknown_objects_count']} unknown objects are also anomalous!")
            report.append(f"     These may represent new astronomical phenomena or data artifacts.")
        
        if combined_results['anomalous_known_objects_count'] > 0:
            report.append(f"  🔍 {combined_results['anomalous_known_objects_count']} known objects are anomalous!")
            report.append(f"     These may represent unusual instances of known object types.")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def classify_known_objects(self, X_test: np.ndarray, confidence_threshold: float = 0.7) -> Dict:
        """Classify objects into known astronomical categories."""
        return self.classifier_system.classify_objects(X_test, confidence_threshold)
    
    def detect_anomalies_only(self, X_test: np.ndarray, confidence_threshold: float = 0.8) -> Dict:
        """Detect anomalies only (without classification)."""
        return self.anomaly_system.detector.detect_anomalies(X_test, confidence_threshold)
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        epochs: int = 50, batch_size: int = 32, 
                        learning_rate: float = 1e-3) -> Dict:
        """Train the astronomical object classifier."""
        return self.classifier_system.train_classifier(X_train, y_train, epochs, batch_size, learning_rate)
    
    def save_analysis_report(self, results: Dict, output_path: str = "enhanced_analysis_report.txt"):
        """Save the comprehensive analysis report to a file."""
        with open(output_path, 'w') as f:
            f.write(results['report'])
        logger.info(f"Analysis report saved to: {output_path}")

def main():
    """Main function to demonstrate the enhanced space system."""
    # Initialize the enhanced system
    enhanced_system = EnhancedSpaceSystem()
    
    print("Enhanced Space Anomaly Detection and Object Classification System")
    print("=" * 70)
    print("This system combines:")
    print("  - Anomaly detection (identifying unusual patterns)")
    print("  - Object classification (categorizing known astronomical objects)")
    print("  - Combined analysis (understanding what types of objects are anomalous)")
    print("=" * 70)
    
    # Run complete analysis
    results = enhanced_system.run_complete_analysis(train_epochs=10)  # Reduced for demo
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total images analyzed: {results['summary']['total_images']}")
    print(f"Anomalies detected: {results['summary']['anomalies_detected']}")
    print(f"Known objects: {results['summary']['known_objects']}")
    print(f"Unknown objects: {results['summary']['unknown_objects']}")
    print(f"Anomalous known objects: {results['summary']['anomalous_known_objects']}")
    print(f"Anomalous unknown objects: {results['summary']['anomalous_unknown_objects']}")
    print("=" * 70)
    
    # Save report
    enhanced_system.save_analysis_report(results)

if __name__ == "__main__":
    main()