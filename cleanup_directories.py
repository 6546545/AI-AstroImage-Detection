#!/usr/bin/env python3
"""
Clean up duplicate class directories and ensure consistent naming.
"""

import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory mapping (old -> new)
DIR_MAPPING = {
    'galaxies': 'galaxy',
    'stars': 'star'
}

def cleanup_directories():
    """Clean up duplicate directories and ensure consistent naming."""
    base_dir = Path('clean_dataset/processed_dataset')
    
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
            
        for old_name, new_name in DIR_MAPPING.items():
            old_dir = split_dir / old_name
            new_dir = split_dir / new_name
            
            if old_dir.exists():
                if new_dir.exists():
                    # Move contents if both directories exist
                    for file in old_dir.glob('*'):
                        shutil.move(str(file), str(new_dir / file.name))
                    old_dir.rmdir()
                    logger.info(f"Merged {old_dir} into {new_dir}")
                else:
                    # Rename if target doesn't exist
                    old_dir.rename(new_dir)
                    logger.info(f"Renamed {old_dir} to {new_dir}")

if __name__ == "__main__":
    logger.info("🧹 Starting directory cleanup...")
    cleanup_directories()
    logger.info("✅ Cleanup complete!")
