"""
Data augmentation utilities for astronomical images.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2
from scipy.ndimage import rotate
import albumentations as A
import random
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AstroImageAugmentor:
    def __init__(self):
        self.augmentation = A.Compose([
            # Spatial transformations
            A.RandomRotate90(p=0.7),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.7),
            
            # Noise and blur - simulating atmospheric distortion
            A.OneOf([
                A.IAAAdditiveGaussianNoise(scale=(10, 30)),
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True),
            ], p=0.3),
            
            # Blur variations - simulating different seeing conditions
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7), p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.MedianBlur(blur_limit=5, p=0.2),
            ], p=0.3),
            
            # Color and brightness - simulating different exposure conditions
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            
            # Astronomical specific transformations
            A.OneOf([
                A.Downscale(scale_min=0.8, scale_max=0.9),  # Simulate different resolutions
                A.ISONoise(intensity=(0.1, 0.5)),  # Simulate sensor noise
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1),  # Simulate atmospheric effects
            ], p=0.3),
            
            # Advanced distortions - simulating optical aberrations
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
                A.GridDistortion(num_steps=5, distort_limit=0.2),
                A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10),
            ], p=0.2),
            
            # Final adjustments
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),  # Enhance contrast in specific regions
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ])

    def check_image_quality(self, image):
        """Check if the image meets quality standards."""
        if image is None:
            return False
            
        # Check minimum size
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return False
            
        # Check if image is too dark or too bright
        mean_brightness = np.mean(image)
        if mean_brightness < 5 or mean_brightness > 250:
            return False
            
        # Check for excessive noise
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        noise_level = np.std(gray)
        if noise_level > 100:  # Adjust threshold as needed
            return False
            
        return True

    def augment_image(self, image):
        """Augment a single image."""
        if len(image.shape) == 2:
            # Convert grayscale to RGB for compatibility
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply augmentation
        augmented = self.augmentation(image=image)
        aug_image = augmented['image']
        
        # Ensure the augmented image maintains good quality
        if not self.check_image_quality(aug_image):
            return None
            
        return aug_image

    def generate_variations(self, image, num_variations=5):
        """Generate multiple variations of an image."""
        variations = []
        for _ in range(num_variations):
            aug_image = self.augment_image(image)
            variations.append(aug_image)
        return variations

def get_category_augmentations(category):
    """Get the number of augmentations based on the category."""
    augmentation_map = {
        'galaxy': 8,       # Galaxies can benefit from more variations due to complex structures
        'nebula': 8,       # Nebulae have varied appearances and can use more augmentations
        'star': 6,         # Stars are relatively simple but benefit from exposure variations
        'planet': 6,       # Planets can use moderate augmentation
        'asteroid': 5,     # Asteroids are simple objects
        'black_hole': 10,  # Black holes are rare, so generate more variations
        'pulsar': 8,       # Pulsars can benefit from temporal-like variations
        'quasar': 8,       # Quasars are rare and can use more variations
    }
    return augmentation_map.get(category, 6)  # Default to 6 if category not found

def process_dataset(input_dir, output_dir, augmentations_per_image=None):
    """Process entire dataset with augmentation."""
    category = os.path.basename(input_dir)
    if augmentations_per_image is None:
        augmentations_per_image = get_category_augmentations(category)
    
    augmentor = AstroImageAugmentor()
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n📁 Processing {len(image_files)} images in {os.path.basename(input_dir)}")
    
    # Process each image in the input directory
    for filename in tqdm(image_files, desc=f"Augmenting {os.path.basename(input_dir)}"):
        try:
            input_path = os.path.join(input_dir, filename)
            
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"⚠️ Warning: Could not read image {filename}")
                continue
                
            # Generate augmented versions
            variations = augmentor.generate_variations(image, augmentations_per_image)
            
            # Save augmented images
            for i, aug_image in enumerate(variations):
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
                cv2.imwrite(output_path, aug_image)
        
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            continue

def main():
    """Main function to run the augmentation process."""
    try:
        print("🎨 Starting Image Augmentation Process")
        print("=" * 60)
        
        # Define paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        raw_dir = os.path.join(base_dir, 'data', 'raw')
        processed_dir = os.path.join(base_dir, 'data', 'processed')
        
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get all category directories
        categories = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        
        if not categories:
            logging.warning("No category directories found in raw data directory")
            return
        
        for category in categories:
            try:
                input_dir = os.path.join(raw_dir, category)
                output_dir = os.path.join(processed_dir, category)
                
                logging.info(f"\n🔄 Processing category: {category}")
                process_dataset(input_dir, output_dir)
            except Exception as e:
                logging.error(f"Error processing category {category}: {str(e)}")
                continue
        
        logging.info("\n✨ Augmentation process completed successfully!")
        
    except FileNotFoundError as e:
        logging.error(f"Directory error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
