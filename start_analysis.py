#!/usr/bin/env python3
"""
Space Anomaly Detection & Classification System
Quick Start Script

This script provides an easy way to run the space analysis system.
"""

import os
import sys
import subprocess
from pathlib import Path

# Set environment variables for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def main():
    """Main startup function."""
    print("🚀 Space Anomaly Detection & Classification System")
    print("=" * 60)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected")
        print("   Consider activating: source .venv/bin/activate")
    
    # Check if test data exists
    test_data_path = Path("test_dataset/images")
    if not test_data_path.exists():
        print("❌ Test dataset not found")
        print("   Expected: test_dataset/images/")
        return
    
    image_files = list(test_data_path.glob("*.jpg")) + list(test_data_path.glob("*.png"))
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
            choice = input("\nSelect operation (0-6): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                print("\n🧪 Testing system...")
                subprocess.run([sys.executable, "space_analyzer.py", "test"])
            elif choice == "2":
                print("\n🔍 Running anomaly detection...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "detect",
                    "--input-dir", "test_dataset/images/",
                    "--output-dir", "results/"
                ])
            elif choice == "3":
                print("\n🌌 Running object classification...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "classify",
                    "--input-dir", "test_dataset/images/",
                    "--output-dir", "results/"
                ])
            elif choice == "4":
                epochs = input("Enter training epochs (default: 20): ").strip() or "20"
                print(f"\n🚀 Running combined analysis with {epochs} epochs...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "analyze",
                    "--input-dir", "test_dataset/images/",
                    "--epochs", epochs
                ])
            elif choice == "5":
                model = input("Enter model to train (anomaly/classifier/both, default: both): ").strip() or "both"
                epochs = input("Enter training epochs (default: 50): ").strip() or "50"
                print(f"\n🏋️  Training {model} model with {epochs} epochs...")
                subprocess.run([
                    sys.executable, "space_analyzer.py", "train",
                    "--input-dir", "test_dataset/images/",
                    "--model", model,
                    "--epochs", epochs
                ])
            elif choice == "6":
                print("\n📖 Help:")
                subprocess.run([sys.executable, "space_analyzer.py", "--help"])
            else:
                print("❌ Invalid choice. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 