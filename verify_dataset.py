#!/usr/bin/env python3
"""
Verify image counts and distribution across classes.
"""

import os
from pathlib import Path
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_images(directory):
    """Count images in a directory."""
    image_extensions = {'.png', '.jpg', '.jpeg'}
    return sum(1 for f in Path(directory).glob('**/*') 
              if f.suffix.lower() in image_extensions)

def check_dataset_distribution():
    """Check the distribution of images across classes and splits."""
    base_dir = Path('clean_dataset/processed_dataset')
    splits = ['train', 'val', 'test']
    
    # Initialize counters
    counts = defaultdict(lambda: defaultdict(int))
    total_by_class = defaultdict(int)
    total_by_split = defaultdict(int)
    
    # Count images
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
            
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = count_images(class_dir)
                counts[split][class_dir.name] = count
                total_by_class[class_dir.name] += count
                total_by_split[split] += count
    
        # Sort classes by total count
    classes = sorted(total_by_class.keys(), 
                    key=lambda x: total_by_class[x], 
                    reverse=True)
    
    # Print header
    header = "{:<15}".format("Class")
    for split in splits:
        header += "{:<10}".format(split)
    header += "{:<10}".format("Total")
    
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    
    # Print rows
    for class_name in classes:
        row = "{:<15}".format(class_name)
        for split in splits:
            row += "{:<10}".format(counts[split][class_name])
        row += "{:<10}".format(total_by_class[class_name])
        print(row)
    
    # Print total row
    print("-" * len(header))
    total_row = "{:<15}".format("TOTAL")
    for split in splits:
        total_row += "{:<10}".format(total_by_split[split])
    total_row += "{:<10}".format(sum(total_by_class.values()))
    print(total_row)
    print("=" * len(header))
    
    # Calculate and display percentages
    total_images = sum(total_by_class.values())
    if total_images > 0:
        print("\n🔍 Dataset Distribution:")
        print("\n📊 Split Percentages:")
        for split in splits:
            percentage = (total_by_split[split] / total_images) * 100
            print(f"{split.capitalize()}: {percentage:.1f}%")
            
        print("\n📊 Class Distribution:")
        for class_name in classes:
            percentage = (total_by_class[class_name] / total_images) * 100
            print(f"{class_name}: {percentage:.1f}%")
            
        # Check for potential issues
        print("\n🔍 Dataset Analysis:")
        
        # Check minimum samples
        min_samples_train = 50
        min_samples_val = 10
        min_samples_test = 10
        
        for class_name in classes:
            issues = []
            if counts['train'][class_name] < min_samples_train:
                issues.append(f"Low training samples ({counts['train'][class_name]})")
            if counts['val'][class_name] < min_samples_val:
                issues.append(f"Low validation samples ({counts['val'][class_name]})")
            if counts['test'][class_name] < min_samples_test:
                issues.append(f"Low test samples ({counts['test'][class_name]})")
                
            if issues:
                print(f"⚠️  {class_name}: {', '.join(issues)}")
            
        # Check class imbalance
        avg_samples = total_images / len(classes)
        imbalance_threshold = 0.5  # 50% deviation from average
        
        imbalanced_classes = [
            c for c in classes
            if abs(total_by_class[c] - avg_samples) / avg_samples > imbalance_threshold
        ]
        
        if imbalanced_classes:
            print("\n⚠️  Class Imbalance Detected:")
            for class_name in imbalanced_classes:
                deviation = (total_by_class[class_name] - avg_samples) / avg_samples * 100
                print(f"{class_name}: {deviation:+.1f}% from average")
    else:
        print("❌ No images found in the dataset!")

if __name__ == "__main__":
    logger.info("🔍 Starting dataset verification...")
    check_dataset_distribution()
    logger.info("✅ Verification complete!")
