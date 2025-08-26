#!/usr/bin/env python3
"""
Script to split data into train and test sets.
This ensures proper data separation for model validation.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files in directory."""
    return [
        f for f in directory.rglob("*")
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]

def split_data(source_dir: str, test_size: float = 0.2, random_seed: int = 42):
    """Split data into train and test sets."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory {source_dir} not found")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all image files
    image_files = get_image_files(source_path)
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * (1 - test_size))
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]
    
    # Create train/test directories
    train_dir = Path('clean_dataset/train')
    test_dir = Path('clean_dataset/test')
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    for f in train_files:
        rel_path = f.relative_to(source_path)
        dest = train_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    
    for f in test_files:
        rel_path = f.relative_to(source_path)
        dest = test_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    
    # Log results
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Training set: {len(train_files)} images")
    logger.info(f"Test set: {len(test_files)} images")
    
    return train_dir, test_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split data into train and test sets")
    parser.add_argument("--source", required=True, help="Source directory containing image data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    train_dir, test_dir = split_data(args.source, args.test_size, args.seed)
    print(f"\n✅ Data split complete!")
    print(f"📁 Training data: {train_dir}")
    print(f"📁 Test data: {test_dir}")
