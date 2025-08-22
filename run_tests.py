#!/usr/bin/env python3
"""
Quick Test Runner for CosmosAI Systems

This script provides an easy way to run all unit tests for the CosmosAI
Space Anomaly Detection System.

Usage:
    python run_tests.py
"""

import os
import sys
import subprocess
import time

def main():
    """Run all CosmosAI system tests."""
    print("🚀 CosmosAI Systems Test Runner")
    print("=" * 50)
    
    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Check if test file exists
    test_file = "test_cosmosai_systems.py"
    if not os.path.exists(test_file):
        print(f"❌ Test file '{test_file}' not found!")
        return False
    
    print(f"📁 Running tests from: {test_file}")
    print(f"🔧 Environment: CPU-only testing")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    try:
        # Run the tests
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=False, text=True)
        
        print("-" * 50)
        print(f"⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if result.returncode == 0:
            print("🎉 All tests completed successfully!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
