#!/usr/bin/env python3
"""
Script to organize SDSS astronomical images into the proper directory structure
with image preprocessing.
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Class mapping from source to destination directories
CLASS_MAPPING = {
    'galaxies': 'galaxy',
    'stars': 'star',
    'quasars': 'quasar',
    'nebulae': 'nebula',
    'planets': 'planet',
    'asteroids': 'asteroid',
    'comets': 'comet',
    'pulsars': 'pulsar',
    'black_holes': 'black_hole'
}

def create_directory_structure():
    """Create the required directory structure."""
    base_dir = Path('clean_dataset/processed_dataset')
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_MAPPING.values():
            (base_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    logger.info("✅ Created directory structure")

def get_image_files(directory):
    """Get all image files in a directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.fits', '.fit'}
    files = []
    for ext in image_extensions:
        files.extend(Path(directory).glob(f'**/*{ext}'))
    return files

def split_and_copy_images(src_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split images into train/val/test sets and copy to appropriate directories."""
    src_dir = Path(src_dir)
    if not src_dir.exists():
        logger.error(f"Source directory {src_dir} does not exist!")
        return

    # Process each class directory
    total_processed = 0
    for src_class in CLASS_MAPPING:
        class_dir = src_dir / src_class
        if not class_dir.exists():
            logger.warning(f"Class directory {class_dir} not found, skipping...")
            continue

        # Get all images for this class
        images = get_image_files(class_dir)
        if not images:
            logger.warning(f"No images found in {class_dir}")
            continue

        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        # Copy files to respective directories
        dest_class = CLASS_MAPPING[src_class]
        base_dir = Path('clean_dataset/processed_dataset')
        
        for files, split in [(train_files, 'train'), 
                           (val_files, 'val'), 
                           (test_files, 'test')]:
            dest_dir = base_dir / split / dest_class
            for src_file in files:
                dest_file = dest_dir / src_file.name
                shutil.copy2(src_file, dest_file)
                total_processed += 1
                
        logger.info(f"✅ Processed {src_class}: "
                   f"Train={len(train_files)}, "
                   f"Val={len(val_files)}, "
                   f"Test={len(test_files)}")

    logger.info(f"🎉 Total images processed: {total_processed}")

def main():
    """Main function to organize the dataset."""
    logger.info("🚀 Starting dataset organization...")
    
    # Create directory structure
    create_directory_structure()
    
    # Process raw images
    raw_data_dir = Path('ai_dataset/sdss_raw')
    if raw_data_dir.exists():
        split_and_copy_images(raw_data_dir)
    else:
        logger.error(f"Raw data directory {raw_data_dir} not found!")

if __name__ == "__main__":
    main()
