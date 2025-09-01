#!/usr/bin/env python3
"""
Space Anomaly Detection & Classification System
Quick Start Script

This script provides an easy way to run the space analysis system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Set environment variables for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import numpy
        import cv2
        import matplotlib
        import sklearn
        import PIL
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {str(e)}")
        print("   Run: pip install -r requirements.txt")
        return False


def main():
    """Main startup function."""
    print("🚀 Space Anomaly Detection & Classification System")
    print("=" * 60)

    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected")
        print("   Consider activating: source .venv/bin/activate")

    # Check dependencies
    if not check_dependencies():
        return

    # Check if test data exists and create if necessary
    test_data_path = Path("test_dataset/images")
    if not test_data_path.exists():
        print("⚠️  Test dataset directory not found, creating it...")
        test_data_path.mkdir(parents=True, exist_ok=True)

    image_files = list(test_data_path.glob("*.jpg")) + list(test_data_path.glob("*.png"))
    if not image_files:
        print("📁 No test images found, copying sample images...")
        
        # Try to copy from clean_dataset first
        clean_dataset_path = Path("data/clean_dataset/test")
        if clean_dataset_path.exists():
            print("📁 Copying images from clean dataset...")
            import shutil
            for category_dir in clean_dataset_path.iterdir():
                if category_dir.is_dir():
                    for img_file in category_dir.glob("*.png"):
                        shutil.copy2(img_file, test_data_path)
                        break  # Only copy one image per category
        else:
            # Fallback to sample_images
            sample_path = Path("sample_images")
            if sample_path.exists():
                print("📁 Copying sample images to test dataset...")
                for img_type in ['*.jpg', '*.png']:
                    for img in sample_path.rglob(img_type):
                        if 'class_info.txt' not in str(img):
                            import shutil
                            shutil.copy2(img, test_data_path)
                            break  # Only copy one image per directory

    # Check again after copying
    image_files = list(test_data_path.glob("*.jpg")) + list(test_data_path.glob("*.png"))
    if not image_files:
        print("❌ No test images found")
        print("   Please add some images to test_dataset/images/")
        return

    print(f"✅ Found {len(image_files)} test images")

    print("\n📋 Available Operations:")
    print("1. Test system")
    print("2. Run anomaly detection")
    print("3. Run object classification")
    print("4. Run combined analysis")
    print("5. Train models")
    print("6. Show help")
    print("0. Exit")

    while True:
        try:
            print("\n" + "="*50)
            choice = input("Select operation (0-6): ").strip()
            print(f"Selected: '{choice}'")

            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                print("\n🧪 Testing system...")
                subprocess.run([sys.executable, "space_analyzer.py", "test"], 
                             capture_output=False, text=True)
                time.sleep(1)  # Brief pause to ensure terminal is ready
            elif choice == "2":
                print("\n🔍 Running anomaly detection...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "detect",
                    "--input-dir", "test_dataset/images/",
                    "--output-dir", "results/"
                ], capture_output=False, text=True)
            elif choice == "3":
                print("\n🌌 Running object classification...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "classify",
                    "--input-dir", "test_dataset/images/",
                    "--output-dir", "results/"
                ], capture_output=False, text=True)
            elif choice == "4":
                try:
                    epochs = input("Enter training epochs (default: 20): ").strip() or "20"
                    epochs = int(epochs)  # Validate input is a number
                    if epochs <= 0:
                        raise ValueError("Epochs must be positive")
                    print(f"\n🚀 Running combined analysis with {epochs} epochs...")
                    subprocess.run([
                        sys.executable, "space_analyzer.py", "analyze",
                        "--input-dir", "test_dataset/images/",
                        "--epochs", str(epochs)
                    ], check=True, capture_output=False, text=True)
                except ValueError as e:
                    print(f"❌ Invalid input: {str(e)}")
                except subprocess.CalledProcessError:
                    print("❌ Analysis failed. Check the error messages above.")
            elif choice == "5":
                model = input("Enter model to train (anomaly/classifier/both, default: both): ").strip() or "both"
                epochs = input("Enter training epochs (default: 50): ").strip() or "50"
                print(f"\n🏋️  Training {model} model with {epochs} epochs...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "train",
                    "--input-dir", "test_dataset/images/",
                    "--model", model,
                    "--epochs", epochs
                ], capture_output=False, text=True)
            elif choice == "6":
                print("\n📖 Help:")
                subprocess.run([sys.executable, "space_analyzer.py", "--help"], 
                             capture_output=False, text=True)
            else:
                print("❌ Invalid choice. Please select 0-6.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 