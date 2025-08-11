#!/usr/bin/env python3
"""
Advanced Training Techniques for Space Anomaly Detection
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import json
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """Advanced training techniques for better model performance."""
    
    def __init__(self):
        self.techniques = {
            'mixup': self.mixup_training,
            'cutmix': self.cutmix_training,
            'label_smoothing': self.label_smoothing,
            'focal_loss': self.focal_loss,
            'curriculum_learning': self.curriculum_learning
        }
    
    def mixup_training(self, x1: torch.Tensor, x2: torch.Tensor, 
                      y1: torch.Tensor, y2: torch.Tensor, alpha: float = 0.2):
        """Mixup data augmentation."""
        lam = np.random.beta(alpha, alpha)
        
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    def cutmix_training(self, x1: torch.Tensor, x2: torch.Tensor, 
                       y1: torch.Tensor, y2: torch.Tensor, alpha: float = 1.0):
        """CutMix data augmentation."""
        lam = np.random.beta(alpha, alpha)
        
        # Random box
        W, H = x1.shape[2], x1.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_x = x1.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    def label_smoothing(self, y: torch.Tensor, smoothing: float = 0.1):
        """Apply label smoothing."""
        num_classes = y.size(-1)
        y_smooth = y * (1 - smoothing) + smoothing / num_classes
        return y_smooth
    
    def focal_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0):
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def curriculum_learning(self, dataset, difficulty_function):
        """Curriculum learning - start with easy samples."""
        # Sort dataset by difficulty
        difficulties = [difficulty_function(sample) for sample in dataset]
        sorted_indices = np.argsort(difficulties)
        
        return [dataset[i] for i in sorted_indices]

class AdvancedDataAugmentation:
    """Advanced data augmentation techniques."""
    
    def __init__(self):
        self.augmentations = {
            'basic': self.basic_augmentations,
            'advanced': self.advanced_augmentations,
            'astronomical': self.astronomical_augmentations
        }
    
    def basic_augmentations(self):
        """Basic augmentations for astronomical images."""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            ToTensorV2()
        ])
    
    def advanced_augmentations(self):
        """Advanced augmentations for better generalization."""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
                A.ISONoise(),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
                A.ElasticTransform(p=0.2),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
            ], p=0.4),
            A.HueSaturationValue(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
            ToTensorV2()
        ])
    
    def astronomical_augmentations(self):
        """Astronomical-specific augmentations."""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # Simulate different exposure times
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(gamma_limit=(50, 150)),
            ], p=0.4),
            # Simulate atmospheric conditions
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(color_shift=(0.01, 0.05)),
            ], p=0.3),
            # Simulate telescope movement
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.2),
            # Simulate cosmic ray hits
            A.CoarseDropout(max_holes=5, max_height=4, max_width=4, 
                          fill_value=255, p=0.1),
            # Simulate dust and artifacts
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
            ], p=0.1),
            ToTensorV2()
        ])

class LearningRateScheduler:
    """Advanced learning rate scheduling."""
    
    def __init__(self, optimizer, scheduler_type='cosine', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 100), eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=kwargs.get('factor', 0.1), 
                patience=kwargs.get('patience', 10), verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metrics=None):
        """Step the scheduler."""
        if self.scheduler_type == 'plateau' and metrics is not None:
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate."""
        return self.scheduler.get_last_lr()

class EarlyStopping:
    """Early stopping mechanism."""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class AdvancedTrainingLoop:
    """Advanced training loop with multiple techniques."""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device='cpu', use_mixup=False, use_cutmix=False, use_label_smoothing=False,
                 use_focal_loss=False, scheduler_type='cosine', early_stopping_patience=10):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Training techniques
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_label_smoothing = use_label_smoothing
        self.use_focal_loss = use_focal_loss
        
        # Advanced components
        self.trainer = AdvancedTrainer()
        self.scheduler = LearningRateScheduler(optimizer, scheduler_type)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply data augmentation techniques
            if self.use_mixup and np.random.random() < 0.5:
                # Mixup
                indices = torch.randperm(data.size(0))
                data2, target2 = data[indices], target[indices]
                data, target = self.trainer.mixup_training(data, data2, target, target2)
            elif self.use_cutmix and np.random.random() < 0.5:
                # CutMix
                indices = torch.randperm(data.size(0))
                data2, target2 = data[indices], target[indices]
                data, target = self.trainer.cutmix_training(data, data2, target, target2)
            
            # Apply label smoothing
            if self.use_label_smoothing:
                target = self.trainer.label_smoothing(target)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate loss
            if self.use_focal_loss:
                loss = self.trainer.focal_loss(output, target.argmax(dim=1) if target.dim() > 1 else target)
            else:
                loss = self.criterion(output, target.argmax(dim=1) if target.dim() > 1 else target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            if target.dim() > 1:
                target_argmax = target.argmax(dim=1, keepdim=True)
            else:
                target_argmax = target.unsqueeze(1)
            correct += pred.eq(target_argmax).sum().item()
            total += target.size(0)
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if self.use_focal_loss:
                    loss = self.trainer.focal_loss(output, target.argmax(dim=1) if target.dim() > 1 else target)
                else:
                    loss = self.criterion(output, target.argmax(dim=1) if target.dim() > 1 else target)
                
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                if target.dim() > 1:
                    target_argmax = target.argmax(dim=1, keepdim=True)
                else:
                    target_argmax = target.unsqueeze(1)
                correct += pred.eq(target_argmax).sum().item()
                total += target.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def train(self, epochs):
        """Complete training loop."""
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  LR: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print("Early stopping triggered!")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

class CurriculumDataset(Dataset):
    """Dataset with curriculum learning support."""
    
    def __init__(self, data, labels, difficulty_function=None):
        self.data = data
        self.labels = labels
        self.difficulty_function = difficulty_function or self.default_difficulty
        
        # Sort by difficulty
        self.sorted_indices = self.sort_by_difficulty()
    
    def default_difficulty(self, sample):
        """Default difficulty function based on image complexity."""
        # Simple heuristic: higher standard deviation = more complex
        return np.std(sample)
    
    def sort_by_difficulty(self):
        """Sort samples by difficulty."""
        difficulties = [self.difficulty_function(sample) for sample in self.data]
        return np.argsort(difficulties)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Use sorted indices for curriculum learning
        sorted_idx = self.sorted_indices[idx]
        return self.data[sorted_idx], self.labels[sorted_idx]
    
    def get_easy_samples(self, num_samples):
        """Get the easiest samples."""
        return [self.data[self.sorted_indices[i]] for i in range(min(num_samples, len(self.data)))]
    
    def get_hard_samples(self, num_samples):
        """Get the hardest samples."""
        start_idx = max(0, len(self.data) - num_samples)
        return [self.data[self.sorted_indices[i]] for i in range(start_idx, len(self.data))]

def main():
    """Test the advanced training techniques."""
    print("🧪 Testing Advanced Training Techniques...")
    
    # Test Advanced Trainer
    print("Testing Advanced Trainer...")
    trainer = AdvancedTrainer()
    
    # Test mixup
    x1 = torch.randn(2, 1, 64, 64)
    x2 = torch.randn(2, 1, 64, 64)
    y1 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    y2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    
    mixed_x, mixed_y = trainer.mixup_training(x1, x2, y1, y2)
    print(f"Mixup output shapes: {mixed_x.shape}, {mixed_y.shape}")
    
    # Test cutmix
    mixed_x, mixed_y = trainer.cutmix_training(x1, x2, y1, y2)
    print(f"CutMix output shapes: {mixed_x.shape}, {mixed_y.shape}")
    
    # Test label smoothing
    y_smooth = trainer.label_smoothing(y1)
    print(f"Label smoothing output shape: {y_smooth.shape}")
    
    # Test focal loss
    outputs = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    focal_loss = trainer.focal_loss(outputs, targets)
    print(f"Focal loss: {focal_loss.item():.4f}")
    
    # Test Advanced Data Augmentation
    print("Testing Advanced Data Augmentation...")
    aug = AdvancedDataAugmentation()
    basic_transforms = aug.basic_augmentations()
    advanced_transforms = aug.advanced_augmentations()
    astronomical_transforms = aug.astronomical_augmentations()
    print("✅ All augmentation pipelines created successfully")
    
    # Test Learning Rate Scheduler
    print("Testing Learning Rate Scheduler...")
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = LearningRateScheduler(optimizer, 'cosine', T_max=100)
    print(f"Initial LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Test Early Stopping
    print("Testing Early Stopping...")
    early_stopping = EarlyStopping(patience=5)
    should_stop = early_stopping(0.5, model)
    print(f"Early stopping triggered: {should_stop}")
    
    # Test Curriculum Dataset
    print("Testing Curriculum Dataset...")
    data = [np.random.rand(64, 64) for _ in range(10)]
    labels = [np.random.randint(0, 2) for _ in range(10)]
    curriculum_dataset = CurriculumDataset(data, labels)
    print(f"Curriculum dataset size: {len(curriculum_dataset)}")
    
    print("✅ All advanced training techniques tested successfully!")

if __name__ == "__main__":
    main()
