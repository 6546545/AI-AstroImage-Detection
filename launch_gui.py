#!/usr/bin/env python3
"""
Launcher script for Space Anomaly Detection System GUI
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'tkinter',
        'matplotlib',
        'numpy',
        'torch',
        'cv2',
        'PIL',
        'sklearn',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sklearn':
                import sklearn
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages using pip."""
    print("Installing missing packages...")
    for package in packages:
        if package == 'tkinter':
            print("tkinter is usually included with Python. If missing, please install Python with tkinter support.")
            continue
        elif package == 'cv2':
            package = 'opencv-python'
        elif package == 'PIL':
            package = 'Pillow'
        elif package == 'sklearn':
            package = 'scikit-learn'
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def main():
    """Main launcher function."""
    print("🚀 Space Anomaly Detection System - GUI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('space_anomaly_detector.py'):
        print("❌ Error: Please run this script from the project directory")
        print("   Make sure you're in the directory containing space_anomaly_detector.py")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install missing packages? (y/n): ")
        if response.lower() == 'y':
            if not install_missing_packages(missing_packages):
                print("❌ Failed to install some packages. Please install them manually.")
                return
        else:
            print("❌ Cannot run GUI without required dependencies.")
            return
    else:
        print("✅ All dependencies are installed")
    
    # Check if data directory exists
    if not os.path.exists('sdss_images'):
        print("⚠️  Warning: sdss_images directory not found")
        print("   The GUI will work, but you'll need to specify a data directory")
    
    # Check if model exists
    if not os.path.exists('models/anomaly_detector.pth'):
        print("⚠️  Warning: No trained model found")
        print("   You can train a new model using the GUI")
    
    print("\n🎯 Starting GUI...")
    print("=" * 50)
    
    try:
        # Import and run the GUI
        from gui_app import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Error importing GUI: {e}")
        print("Make sure gui_app.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running GUI: {e}")
        print("Please check the error message above")

if __name__ == "__main__":
    main() 