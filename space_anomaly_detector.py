import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cv2
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for space anomaly detection."""
    
    def __init__(self, input_channels=1, latent_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DataPreprocessor:
    """Handles data preprocessing for space imagery."""
    
    def __init__(self, img_size=(512, 512), grayscale=True):
        self.img_size = img_size
        self.grayscale = grayscale
    
    def load_and_preprocess_images(self, folder: str) -> Tuple[np.ndarray, List[str]]:
        """Load and preprocess images from a folder (supports nested subdirectories)."""
        images = []
        filenames = []
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        
        logger.info(f"Loading images from {folder}")
        
        # Check if folder contains subdirectories (class-based organization)
        subdirs = [d for d in os.listdir(folder) 
                  if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
        
        if subdirs:
            # Load from subdirectories (class-based organization)
            logger.info(f"Found {len(subdirs)} subdirectories, loading from all classes")
            for subdir in subdirs:
                subdir_path = os.path.join(folder, subdir)
                image_files = [f for f in os.listdir(subdir_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for file in image_files:
                    path = os.path.join(subdir_path, file)
                    img = cv2.imread(path, flag)
                    
                    if img is None:
                        logger.warning(f"Skipping unreadable file: {file}")
                        continue
                    
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize to [0, 1]
                    images.append(img)
                    filenames.append(f"{subdir}/{file}")
        else:
            # Load from flat directory
            image_files = [f for f in os.listdir(folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in image_files:
                path = os.path.join(folder, file)
                img = cv2.imread(path, flag)
                
                if img is None:
                    logger.warning(f"Skipping unreadable file: {file}")
                    continue
                
                img = cv2.resize(img, self.img_size)
                img = img / 255.0  # Normalize to [0, 1]
                images.append(img)
                filenames.append(file)
        
        if not images:
            logger.warning(f"No image files found in {folder}")
            return np.array([]), []
        
        # Create progress bar for image loading
        pbar = tqdm(images, desc="Loading and preprocessing images", unit="image")
        
        # Update progress bar
        pbar.set_postfix({
            'Loaded': len(images),
            'Classes': len(set([f.split('/')[0] for f in filenames if '/' in f]))
        })
        
        pbar.close()
        
        images = np.array(images)
        if self.grayscale and images.ndim == 3:
            images = images[..., np.newaxis]  # Add channel dimension
        
        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        return images, filenames
    
    def prepare_dataset(self, folder: str, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare ML-ready dataset with train/test split."""
        from sklearn.model_selection import train_test_split
        
        X, filenames = self.load_and_preprocess_images(folder)
        X_train, X_test, train_files, test_files = train_test_split(
            X, filenames, test_size=test_ratio, random_state=42
        )
        
        return X_train, X_test, test_files
    
    def save_dataset(self, X_train: np.ndarray, X_test: np.ndarray, 
                    train_files: List[str], test_files: List[str], 
                    out_dir: str = "preprocessed_data"):
        """Save preprocessed dataset."""
        os.makedirs(out_dir, exist_ok=True)
        
        np.save(os.path.join(out_dir, "X_train.npy"), X_train)
        np.save(os.path.join(out_dir, "X_test.npy"), X_test)
        
        # Save filenames
        with open(os.path.join(out_dir, "train_files.json"), 'w') as f:
            json.dump(train_files, f)
        with open(os.path.join(out_dir, "test_files.json"), 'w') as f:
            json.dump(test_files, f)
        
        logger.info(f"Saved dataset to: {out_dir}")

class AnomalyDetector:
    """Main anomaly detection system."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        # Enhanced device selection: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model = ConvAutoencoder()
        self.model.to(self.device)
        logging.info(f"Using device: {self.device}")
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logging.info(f"Loaded model from {model_path}")
    
    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 1e-3, save_path: str = "models/anomaly_detector.pth"):
        """Train the autoencoder model."""
        logger.info("Starting model training...")
        
        # Prepare data
        # Transpose from NHWC to NCHW for PyTorch
        if X_train.shape[-1] == 1:
            X_train = np.transpose(X_train, (0, 3, 1, 2))
        train_tensor = torch.tensor(X_train.astype(np.float32))
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        losses = []
        start_time = time.time()
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            total_loss = 0
            num_batches = len(train_loader)
            
            # Create progress bar for batches within each epoch
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                             leave=False, unit="batch")
            
            for batch in batch_pbar:
                imgs = batch[0].to(self.device)
                outputs = self.model(imgs)
                loss = criterion(outputs, imgs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{total_loss / (batch_pbar.n + 1):.6f}'
                })
            
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            
            # Calculate time estimates
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Loss': f'{avg_loss:.6f}',
                'Elapsed': f'{elapsed_time/60:.1f}m',
                'ETA': f'{estimated_remaining/60:.1f}m'
            })
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                          f"Elapsed: {elapsed_time/60:.1f}m - ETA: {estimated_remaining/60:.1f}m")
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Final loss: {losses[-1]:.6f}")
        logger.info(f"Loss reduction: {losses[0] - losses[-1]:.6f}")
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
        
        return losses
    
    def detect_anomalies(self, X_test: np.ndarray, confidence_threshold: float = 0.8, 
                        error_percentile: float = 95) -> Dict:
        """Detect anomalies in test images."""
        logger.info("Detecting anomalies...")
        
        # Transpose from NHWC to NCHW for PyTorch
        if X_test.shape[-1] == 1:
            X_test = np.transpose(X_test, (0, 3, 1, 2))
        self.model.eval()
        test_tensor = torch.tensor(X_test.astype(np.float32))
        
        # Create progress bar for anomaly detection
        pbar = tqdm(total=len(X_test), desc="Detecting anomalies", unit="image")
        
        with torch.no_grad():
            # Process in batches for better progress tracking
            batch_size = 32
            reconstructions = []
            
            for i in range(0, len(X_test), batch_size):
                batch = test_tensor[i:i+batch_size].to(self.device)
                batch_reconstructions = self.model(batch).cpu().numpy()
                reconstructions.append(batch_reconstructions)
                pbar.update(len(batch))
            
            reconstructions = np.concatenate(reconstructions, axis=0)
        
        pbar.close()
        
        # Compute reconstruction errors
        logger.info("Computing reconstruction errors...")
        errors = np.mean((X_test - reconstructions) ** 2, axis=(1, 2, 3))
        
        # Calculate confidence scores
        logger.info("Calculating confidence scores...")
        confidences = self._calculate_confidence(errors)
        
        # Determine anomaly threshold
        threshold = np.percentile(errors, error_percentile)
        
        # Find anomalies with high confidence
        high_confidence_anomalies = np.where(
            (errors > threshold) & (confidences > confidence_threshold)
        )[0]
        
        # Get all anomalies for reference
        all_anomalies = np.where(errors > threshold)[0]
        
        results = {
            'high_confidence_anomalies': high_confidence_anomalies.tolist(),
            'all_anomalies': all_anomalies.tolist(),
            'errors': errors.tolist(),
            'confidences': confidences.tolist(),
            'threshold': float(threshold),
            'confidence_threshold': confidence_threshold
        }
        
        logger.info(f"Found {len(high_confidence_anomalies)} high-confidence anomalies")
        logger.info(f"Found {len(all_anomalies)} total anomalies")
        
        return results
    
    def _calculate_confidence(self, errors: np.ndarray) -> np.ndarray:
        """Calculate confidence scores based on error distribution."""
        e_min = np.min(errors)
        e_max = np.max(errors)
        confidences = 1 - (errors - e_min) / (e_max - e_min + 1e-8)
        return confidences
    
    def export_anomalies(self, X_test: np.ndarray, results: Dict, 
                        filenames: Optional[List[str]] = None,
                        output_dir: str = "anomalies_export") -> List[str]:
        """Export anomaly images with metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        anomalies = results['high_confidence_anomalies']
        errors = results['errors']
        confidences = results['confidences']
        
        # Create progress bar for export
        pbar = tqdm(anomalies, desc="Exporting anomalies", unit="image")
        
        for idx in pbar:
            img = (X_test[idx][0] * 255).astype('uint8')
            error = errors[idx]
            confidence = confidences[idx]
            
            # Create filename
            if filenames and idx < len(filenames):
                base_name = Path(filenames[idx]).stem
            else:
                base_name = f"anomaly_{idx:04d}"
            
            filename = f"{base_name}_err{error:.5f}_conf{confidence:.3f}.png"
            out_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(out_path, img)
            exported_files.append(out_path)
            
            # Update progress bar
            pbar.set_postfix({
                'Exported': len(exported_files),
                'Current': filename[:20] + '...' if len(filename) > 20 else filename
            })
        
        pbar.close()
        
        # Save results metadata
        metadata_path = os.path.join(output_dir, "anomaly_results.json")
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Exported {len(exported_files)} anomaly images to {output_dir}")
        logger.info(f"Results metadata saved to: {metadata_path}")
        
        return exported_files
        
        return exported_files
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def visualize_anomalies(self, X_test: np.ndarray, results: Dict, 
                           num_samples: int = 5):
        """Visualize anomaly detection results."""
        anomalies = results['high_confidence_anomalies'][:num_samples]
        errors = results['errors']
        
        if not anomalies:
            logger.info("No anomalies to visualize.")
            return
        
        self.model.eval()
        # Fix tensor format: PyTorch expects (batch, channels, height, width)
        test_tensor = torch.tensor(X_test.astype(np.float32)).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            reconstructions = self.model(test_tensor.to(self.device)).cpu().numpy()
            # Convert back to (batch, height, width, channels) for visualization
            reconstructions = reconstructions.transpose(0, 2, 3, 1)
        
        for i, idx in enumerate(anomalies):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            axs[0].imshow(X_test[idx].squeeze(), cmap='gray')
            axs[0].set_title(f"Original Image {idx}")
            axs[0].axis('off')
            
            axs[1].imshow(reconstructions[idx].squeeze(), cmap='gray')
            axs[1].set_title(f"Reconstruction\nError: {errors[idx]:.5f}")
            axs[1].axis('off')
            
            plt.tight_layout()
            # Save figure instead of showing it to avoid GUI backend issues
            plt.savefig(f'anomaly_visualization_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved anomaly visualization {i+1}/{len(anomalies)}")

class SpaceAnomalyDetectionSystem:
    """Complete space anomaly detection system."""
    
    def __init__(self, data_dir: str = "sdss_images", model_path: Optional[str] = None):
        self.data_dir = data_dir
        self.preprocessor = DataPreprocessor()
        self.detector = AnomalyDetector(model_path)
        
    def run_complete_pipeline(self, confidence_threshold: float = 0.8, 
                            train_epochs: int = 50) -> Dict:
        """Run the complete anomaly detection pipeline."""
        logger.info("Starting complete anomaly detection pipeline...")
        
        # Step 1: Data preprocessing
        logger.info("Step 1: Preprocessing data...")
        X_train, X_test, test_files = self.preprocessor.prepare_dataset(self.data_dir)
        
        # Save preprocessed data
        self.preprocessor.save_dataset(X_train, X_test, 
                                     [f"train_{i}" for i in range(len(X_train))], 
                                     test_files)
        
        # Step 2: Model training
        logger.info("Step 2: Training model...")
        self.detector.train(X_train, epochs=train_epochs)
        
        # Step 3: Anomaly detection
        logger.info("Step 3: Detecting anomalies...")
        results = self.detector.detect_anomalies(X_test, confidence_threshold)
        
        # Step 4: Export results
        logger.info("Step 4: Exporting results...")
        exported_files = self.detector.export_anomalies(X_test, results, test_files)
        
        # Step 5: Visualization
        logger.info("Step 5: Generating visualizations...")
        self.detector.visualize_anomalies(X_test, results)
        
        # Compile final results
        final_results = {
            'high_confidence_anomalies': len(results['high_confidence_anomalies']),
            'total_anomalies': len(results['all_anomalies']),
            'total_images_processed': len(X_test),
            'confidence_threshold': confidence_threshold,
            'exported_files': exported_files,
            'anomaly_indices': results['high_confidence_anomalies'],
            'error_threshold': results['threshold']
        }
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results: {final_results}")
        
        return final_results

def main():
    """Main function to run the space anomaly detection system."""
    # Initialize the system
    system = SpaceAnomalyDetectionSystem()
    
    # Run the complete pipeline
    results = system.run_complete_pipeline(confidence_threshold=0.8, train_epochs=30)
    
    # Print summary
    print("\n" + "="*50)
    print("SPACE ANOMALY DETECTION RESULTS")
    print("="*50)
    print(f"Total images processed: {results['total_images_processed']}")
    print(f"High-confidence anomalies: {results['high_confidence_anomalies']}")
    print(f"Total anomalies detected: {results['total_anomalies']}")
    print(f"Confidence threshold: {results['confidence_threshold']}")
    print(f"Error threshold: {results['error_threshold']:.6f}")
    print(f"Exported files: {len(results['exported_files'])}")
    print("="*50)

if __name__ == "__main__":
    main() 