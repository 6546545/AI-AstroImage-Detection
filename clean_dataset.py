import os
import shutil
from sklearn.model_selection import train_test_split

# Directories
RAW_DIR = "ai_dataset/sdss_raw"
OUTPUT_DIR = "clean_dataset/processed_dataset"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ensure output directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def split_dataset():
    print(f"Processing dataset from {RAW_DIR}...")

    # Go through each category folder
    for category in os.listdir(RAW_DIR):
        category_path = os.path.join(RAW_DIR, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n[INFO] Processing category: {category}")
        
        # Collect image files for this category
        images = [os.path.join(category_path, f) for f in os.listdir(category_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            print(f"  [WARNING] No images found in {category_path}")
            continue

        # Split into train / temp
        train_files, temp_files = train_test_split(images, train_size=TRAIN_RATIO, random_state=42)
        # Split temp into val / test
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Copy files to new structure
        for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = os.path.join(OUTPUT_DIR, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for file_path in files:
                shutil.copy(file_path, split_dir)

        print(f"  [DONE] {len(train_files)} train | {len(val_files)} val | {len(test_files)} test")

    print("\n✅ Dataset splitting complete!")

if __name__ == "__main__":
    split_dataset()
