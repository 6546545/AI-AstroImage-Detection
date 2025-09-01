#!/usr/bin/env python3
"""
Dataset Downloader for Space Anomaly Detection System
Downloads and organizes astronomical images from various sources.
"""

import os
import sys
import time
import json
import requests
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from astroquery.sdss import SDSS
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import concurrent.futures
from PIL import Image
import io
import logging
import time
import random

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AstronomicalDatasetDownloader:
    """Downloads and organizes astronomical images for training."""
    
    def __init__(self, base_dir="data/raw"):
        self.base_dir = Path(base_dir)
        self.categories = {
            'star': {'min_samples': 1000, 'sources': ['sdss', 'mast']},
            'galaxy': {'min_samples': 1000, 'sources': ['sdss']},
            'nebula': {'min_samples': 500, 'sources': ['mast']},
            'planet': {'min_samples': 200, 'sources': ['nasa_planetary']},
            'asteroid': {'min_samples': 200, 'sources': ['minor_planet_center']},
            'comet': {'min_samples': 200, 'sources': ['minor_planet_center']},
            'quasar': {'min_samples': 300, 'sources': ['sdss']},
            'pulsar': {'min_samples': 200, 'sources': ['mast']},
            'black_hole': {'min_samples': 100, 'sources': ['event_horizon']}
        }
        
    def setup_directories(self):
        """Create necessary directories for each category."""
        for category in self.categories:
            category_dir = self.base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
    def download_sdss_images(self, category, count):
        """Download images from SDSS DR17."""
        logger.info(f"Downloading {count} SDSS images for {category}")
        
        try:
            if category == 'galaxy':
                query = "SELECT TOP {} ra, dec FROM Galaxy WHERE clean = 1".format(count)
            elif category == 'quasar':
                query = "SELECT TOP {} ra, dec FROM QSO WHERE clean = 1".format(count)
            elif category == 'star':
                query = "SELECT TOP {} ra, dec FROM Star WHERE clean = 1".format(count)
            else:
                return []

            results = SDSS.query_sql(query)
            
            if not results:
                logger.warning(f"No results found for {category} in SDSS")
                return []
                
            downloaded = []
            for idx, result in enumerate(tqdm(results, desc=f"Downloading {category}")):
                try:
                    im = SDSS.get_images(coordinates=SkyCoord(result['ra'], result['dec'],
                                                            unit=(u.degree, u.degree)),
                                       band='g')[0]
                    
                    # Save the image
                    filename = self.base_dir / category / f"{category}_{idx:04d}.fits"
                    im.writeto(filename, overwrite=True)
                    downloaded.append(str(filename))
                    
                    # Convert FITS to PNG for easier viewing
                    png_filename = filename.with_suffix('.png')
                    self.fits_to_png(filename, png_filename)
                    
                except Exception as e:
                    logger.error(f"Error downloading SDSS image {idx} for {category}: {e}")
                    continue
                    
            return downloaded
            
        except Exception as e:
            logger.error(f"Error in SDSS download for {category}: {e}")
            return []
            
    def download_nasa_planetary(self, category, count):
        """Download images from NASA's Planetary Database."""
        nasa_api_key = os.getenv('NASA_API_KEY')
        if not nasa_api_key:
            logger.error("NASA API key not found in environment variables")
            return []
            
        base_url = "https://images-api.nasa.gov/search"
        
        params = {
            'q': category,
            'media_type': 'image',
            'page_size': count
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if 'collection' not in data or 'items' not in data['collection']:
                logger.warning(f"No NASA images found for {category}")
                return []
                
            items = data['collection']['items']
            downloaded = []
            
            for idx, item in enumerate(tqdm(items, desc=f"Downloading NASA {category}")):
                try:
                    image_url = next(link['href'] for link in item['links'] 
                                   if link['render'] == 'image')
                    
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        filename = self.base_dir / category / f"nasa_{category}_{idx:04d}.jpg"
                        filename.write_bytes(response.content)
                        downloaded.append(str(filename))
                        
                except Exception as e:
                    logger.error(f"Error downloading NASA image {idx} for {category}: {e}")
                    continue
                    
            return downloaded
            
        except Exception as e:
            logger.error(f"Error in NASA download for {category}: {e}")
            return []
            
    def fits_to_png(self, fits_file, png_file):
        """Convert FITS file to PNG for easier viewing."""
        from astropy.visualization import astropy_mpl_style
        import matplotlib.pyplot as plt
        plt.style.use(astropy_mpl_style)
        
        try:
            from astropy.io import fits
            image_data = fits.getdata(fits_file)
            
            plt.figure()
            plt.imshow(image_data, cmap='gray')
            plt.axis('off')
            plt.savefig(png_file, bbox_inches='tight', pad_inches=0)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error converting FITS to PNG: {e}")

    def download_category(self, category):
        """Download images for a specific category."""
        logger.info(f"Starting download for category: {category}")
        
        category_info = self.categories[category]
        min_samples = category_info['min_samples']
        sources = category_info['sources']
        
        downloaded = []
        
        for source in sources:
            try:
                if source == 'sdss':
                    downloaded.extend(self.download_sdss_images(category, min_samples))
                elif source == 'nasa_planetary':
                    downloaded.extend(self.download_nasa_planetary(category, min_samples))
                # Add more sources as needed
                
            except Exception as e:
                logger.error(f"Error downloading from {source} for {category}: {e}")
                continue
                
        return downloaded
        
    def download_all(self):
        """Download datasets for all categories."""
        self.setup_directories()
        
        total_downloaded = {}
        for category in self.categories:
            logger.info(f"Processing category: {category}")
            downloaded = self.download_category(category)
            total_downloaded[category] = len(downloaded)
            
            # Log progress
            logger.info(f"Downloaded {len(downloaded)} images for {category}")
            
        return total_downloaded
        
def main():
    """Main function to run the dataset downloader."""
    print("🚀 Starting Astronomical Dataset Download")
    print("=" * 60)
    
    # Verify NASA API key
    nasa_api_key = os.getenv('NASA_API_KEY')
    if not nasa_api_key:
        print("❌ NASA API key not found! Please set the NASA_API_KEY environment variable.")
        sys.exit(1)
    print("✅ NASA API key found")
    
    # Configure download parameters
    categories = ['galaxy', 'nebula', 'star', 'planet', 'asteroid', 'black_hole', 'pulsar', 'quasar']
    base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    
    for category in categories:
        print(f"\n📥 Downloading {category} images...")
        output_dir = os.path.join(base_output_dir, category)
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure NASA API search parameters
        search_query = f"{category} space astronomical"
        api_url = f"https://images-api.nasa.gov/search"
        params = {
            'q': search_query,
            'media_type': 'image',
            'page': 1,
            'page_size': 100
        }
        
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'collection' in data and 'items' in data['collection']:
                items = data['collection']['items']
                print(f"Found {len(items)} {category} images")
                
                for idx, item in enumerate(items, 1):
                    if 'links' in item:
                        image_url = next((link['href'] for link in item['links'] if link['render'] == 'image'), None)
                        if image_url:
                            try:
                                image_response = requests.get(image_url, stream=True)
                                image_response.raise_for_status()
                                
                                # Generate a unique filename
                                filename = f"nasa_{category}_{idx}.jpg"
                                output_path = os.path.join(output_dir, filename)
                                
                                # Save the image
                                with open(output_path, 'wb') as f:
                                    for chunk in image_response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                print(f"✅ Downloaded: {filename}")
                            except Exception as e:
                                print(f"❌ Error downloading image {idx} for {category}: {str(e)}")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
            else:
                print(f"❌ No images found for category: {category}")
                
        except Exception as e:
            print(f"❌ Error processing category {category}: {str(e)}")
            continue
        
    print("\n🎉 Download process completed!")
    
    # Create downloader instance
    downloader = AstronomicalDatasetDownloader()
    
    try:
        # Download all datasets
        results = downloader.download_all()
        
        print("\n📊 Download Summary:")
        for category, count in results.items():
            print(f"{category}: {count} images")
            
        print("\n✅ Download complete!")
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        logger.exception("Download failed")
        
if __name__ == "__main__":
    main()
