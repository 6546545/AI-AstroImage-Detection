#!/usr/bin/env python3
"""
Setup script for creating the test dataset structure.
This ensures proper validation during training and testing.
"""

import os
import shutil
from pathlib import Path

# Classes from the system test output
CLASSES = [
    'star', 'galaxy', 'nebula', 'planet', 'asteroid',
    'comet', 'quasar', 'pulsar', 'black_hole'
]

def setup_test_dataset():
    """Create test dataset directory structure."""
    base_dir = Path('clean_dataset')
    test_dir = base_dir / 'test'
    
    # Create test directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class subdirectories
    for class_name in CLASSES:
        class_dir = test_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create a placeholder .gitkeep file to maintain directory structure
        gitkeep = class_dir / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("✅ Test dataset structure created successfully")
    print(f"📁 Test directory: {test_dir}")
    print("\nDirectory structure:")
    for class_name in CLASSES:
        print(f"  └── {class_name}/")

if __name__ == "__main__":
    setup_test_dataset()
