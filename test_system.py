#!/usr/bin/env python3
"""
Test script for Space Anomaly Detection System
"""

import os
import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless mode
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import signal
from functools import wraps

# Set OMP/MKL thread env vars for stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("anomaly_test")

# Watchdog decorator for timeouts
class TimeoutException(Exception):
    pass

def timeout(seconds=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds.")
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from space_anomaly_detector import SpaceAnomalyDetectionSystem, DataPreprocessor, AnomalyDetector
from config import *

def log_file_summary():
    logger.info("Output file summary:")
    for folder in [OUTPUT_CONFIG['preprocessed_data_dir'], OUTPUT_CONFIG['anomalies_export_dir'], 'models']:
        if os.path.exists(folder):
            files = list(Path(folder).glob("**/*"))
            logger.info(f"{folder}: {len(files)} files")
            for f in files:
                logger.info(f"  - {f}")
        else:
            logger.info(f"{folder}: (not found)")

@timeout(180)
def test_data_preprocessing():
    logger.info("[TEST] Data Preprocessing started.")
    start = time.time()
    preprocessor = DataPreprocessor(
        img_size=DATA_CONFIG['image_size'],
        grayscale=DATA_CONFIG['grayscale']
    )
    if not os.path.exists(DATA_CONFIG['data_directory']):
        logger.error(f"Data directory '{DATA_CONFIG['data_directory']}' not found!")
        return False
    try:
        logger.info("Loading and preprocessing images...")
        X_train, X_test, test_files = preprocessor.prepare_dataset(
            DATA_CONFIG['data_directory'], 
            test_ratio=DATA_CONFIG['test_ratio']
        )
        logger.info(f"Loaded {len(X_train)} training images and {len(X_test)} test images. Shape: {X_train.shape}")
        preprocessor.save_dataset(X_train, X_test, [f"train_{i}" for i in range(len(X_train))], test_files)
        logger.info(f"Saved preprocessed data to {OUTPUT_CONFIG['preprocessed_data_dir']}")
        logger.info(f"[TEST] Data Preprocessing completed successfully in {time.time()-start:.2f}s.")
        log_file_summary()
        return True
    except Exception as e:
        logger.exception(f"Error in data preprocessing: {e}")
        return False

@timeout(300)
def test_model_training():
    logger.info("[TEST] Model Training started.")
    start = time.time()
    try:
        logger.info("Loading preprocessed training data...")
        X_train = np.load(os.path.join(OUTPUT_CONFIG['preprocessed_data_dir'], "X_train.npy"))
        detector = AnomalyDetector(device=SYSTEM_CONFIG['device'])
        logger.info("Training model...")
        losses = detector.train(
            X_train, 
            epochs=5,  # Reduced for testing
            batch_size=MODEL_CONFIG['batch_size'],
            learning_rate=MODEL_CONFIG['learning_rate']
        )
        logger.info(f"Model training completed. Final loss: {losses[-1]:.6f}, Loss reduction: {losses[0] - losses[-1]:.6f}")
        logger.info(f"[TEST] Model Training completed successfully in {time.time()-start:.2f}s.")
        log_file_summary()
        return True
    except Exception as e:
        logger.exception(f"Error in model training: {e}")
        return False

@timeout(180)
def test_anomaly_detection():
    logger.info("[TEST] Anomaly Detection started.")
    start = time.time()
    try:
        logger.info("Loading preprocessed test data...")
        X_test = np.load(os.path.join(OUTPUT_CONFIG['preprocessed_data_dir'], "X_test.npy"))
        detector = AnomalyDetector(
            model_path=MODEL_CONFIG['model_save_path'],
            device=SYSTEM_CONFIG['device']
        )
        logger.info("Detecting anomalies...")
        results = detector.detect_anomalies(
            X_test,
            confidence_threshold=DETECTION_CONFIG['confidence_threshold'],
            error_percentile=DETECTION_CONFIG['error_percentile']
        )
        logger.info(f"Anomaly detection completed. High-confidence anomalies: {len(results['high_confidence_anomalies'])}, Total anomalies: {len(results['all_anomalies'])}, Error threshold: {results['threshold']:.6f}")
        logger.info(f"[TEST] Anomaly Detection completed successfully in {time.time()-start:.2f}s.")
        log_file_summary()
        return True
    except Exception as e:
        logger.exception(f"Error in anomaly detection: {e}")
        return False

@timeout(600)
def test_complete_pipeline():
    logger.info("[TEST] Complete Pipeline started.")
    start = time.time()
    try:
        logger.info("Initializing SpaceAnomalyDetectionSystem...")
        system = SpaceAnomalyDetectionSystem(
            data_dir=DATA_CONFIG['data_directory']
        )
        logger.info("Running complete pipeline...")
        results = system.run_complete_pipeline(
            confidence_threshold=DETECTION_CONFIG['confidence_threshold'],
            train_epochs=4  # Reduced for testing
        )
        logger.info(f"Complete pipeline executed. Results: {results}")
        logger.info(f"[TEST] Complete Pipeline completed successfully in {time.time()-start:.2f}s.")
        log_file_summary()
        return True
    except Exception as e:
        logger.exception(f"Error in complete pipeline: {e}")
        return False

def generate_test_report(selected_test=None):
    logger.info("Generating test report...")
    print("="*60)
    print("SPACE ANOMALY DETECTION SYSTEM - TEST REPORT")
    print("="*60)
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Training", test_model_training),
        ("Anomaly Detection", test_anomaly_detection),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    results = {}
    for test_name, test_func in tests:
        if selected_test and test_name != selected_test:
            continue
        logger.info(f"Running {test_name} test...")
        print(f"\nRunning {test_name} test...")
        try:
            results[test_name] = test_func()
        except TimeoutException as te:
            logger.error(str(te))
            print(f"❌ {test_name} timed out.")
            results[test_name] = False
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name}: {status}")
    print(f"\nOverall: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    logger.info(f"Test report complete. {passed}/{total} tests passed.")
    log_file_summary()
    return results

def visualize_results():
    logger.info("Generating visualizations...")
    print("\nGenerating visualizations...")
    try:
        results_file = os.path.join(OUTPUT_CONFIG['anomalies_export_dir'], "anomaly_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            errors = np.array(results['errors'])
            plt.hist(errors, bins=20, alpha=0.7, color='blue')
            plt.axvline(results['threshold'], color='red', linestyle='--', 
                       label=f'Threshold: {results["threshold"]:.5f}')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.legend()
            plt.subplot(1, 2, 2)
            confidences = np.array(results['confidences'])
            plt.hist(confidences, bins=20, alpha=0.7, color='green')
            plt.axvline(DETECTION_CONFIG['confidence_threshold'], color='red', 
                       linestyle='--', label=f'Confidence Threshold: {DETECTION_CONFIG["confidence_threshold"]}')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution')
            plt.legend()
            plt.tight_layout()
            plt.savefig('test_results_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Visualization saved as 'test_results_visualization.png'")
        else:
            logger.warning("No results file found. Run the complete pipeline first.")
    except Exception as e:
        logger.exception(f"Error in visualization: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Space Anomaly Detection System with advanced debugging.")
    parser.add_argument('--test', type=str, help="Run only a specific test (Data Preprocessing, Model Training, Anomaly Detection, Complete Pipeline)")
    args = parser.parse_args()
    logger.info("Starting Space Anomaly Detection System Tests...")
    print("Starting Space Anomaly Detection System Tests...")
    print(f"Data directory: {DATA_CONFIG['data_directory']}")
    print(f"Image size: {DATA_CONFIG['image_size']}")
    print(f"Confidence threshold: {DETECTION_CONFIG['confidence_threshold']}")
    test_results = generate_test_report(selected_test=args.test)
    if sum(test_results.values()) >= 3 and (not args.test):
        visualize_results()
    logger.info("All tests complete.")
    return test_results

if __name__ == "__main__":
    main() 