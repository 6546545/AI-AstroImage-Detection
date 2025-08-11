#!/usr/bin/env python3
"""
Advanced Neural Network Models for Space Anomaly Detection
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
import json
from typing import List, Tuple, Dict, Optional, Union
import logging
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionTransformer(nn.Module):
    """Vision Transformer for astronomical object classification."""
    
    def __init__(self, image_size=512, patch_size=16, num_classes=10, dim=768, 
                 depth=12, heads=12, mlp_dim=3072, channels=1, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * channels, dim)
        )
        
        # Position embedding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add classification token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x += self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer in self.transformer:
            x = transformer(x)
        
        # Classification
        x = x[:, 0]  # Use classification token
        x = self.classifier(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention."""
    
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class EfficientNetAstronomical(nn.Module):
    """EfficientNet-based model for astronomical classification."""
    
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        
        # EfficientNet backbone (MobileNetV3-like)
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Inverted residual blocks
            self._make_layer(16, 16, 1),
            self._make_layer(16, 24, 2),
            self._make_layer(24, 24, 1),
            self._make_layer(24, 40, 2),
            self._make_layer(40, 40, 1),
            self._make_layer(40, 80, 2),
            self._make_layer(80, 80, 1),
            self._make_layer(80, 80, 1),
            self._make_layer(80, 112, 1),
            self._make_layer(112, 112, 1),
            self._make_layer(112, 192, 2),
            self._make_layer(192, 192, 1),
            self._make_layer(192, 192, 1),
            self._make_layer(192, 320, 1),
            self._make_layer(320, 320, 1),
            
            # Final convolution
            nn.Conv2d(320, 1280, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        """Create inverted residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for improved anomaly detection."""
    
    def __init__(self, input_channels=1, latent_dim=128, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ])
        
        modules.extend([
            nn.ConvTranspose2d(hidden_dims[-1], input_channels,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = z.view(z.size(0), -1, 2, 2)
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var):
        """VAE loss function."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss

class SelfSupervisedAnomalyDetector(nn.Module):
    """Self-supervised learning approach for anomaly detection."""
    
    def __init__(self, input_channels=1, feature_dim=512):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projection = self.projection(features)
        anomaly_score = self.anomaly_head(features)
        return features, projection, anomaly_score
    
    def contrastive_loss(self, proj1, proj2, temperature=0.1):
        """Contrastive learning loss."""
        # Normalize projections
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(proj1, proj2.t()) / temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        return F.cross_entropy(logits, labels)

class EnsembleAstronomicalClassifier(nn.Module):
    """Ensemble of multiple models for improved classification."""
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0] * len(models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x):
        """Forward pass through ensemble."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        weighted_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            weighted_output += weight * output
        
        return weighted_output
    
    def predict(self, x: torch.Tensor) -> Dict:
        """Get ensemble prediction with confidence."""
        self.eval()
        with torch.no_grad():
            outputs = []
            for model in self.models:
                model.eval()
                output = model(x)
                outputs.append(output)
            
            # Get individual predictions
            predictions = []
            confidences = []
            
            for output, weight in zip(outputs, self.weights):
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                
                predictions.append(predicted.item())
                confidences.append(confidence.item() * weight)
            
            # Weighted voting
            weighted_predictions = {}
            for pred, conf in zip(predictions, confidences):
                if pred in weighted_predictions:
                    weighted_predictions[pred] += conf
                else:
                    weighted_predictions[pred] = conf
            
            # Get final prediction
            final_prediction = max(weighted_predictions.items(), key=lambda x: x[1])
            
            return {
                'prediction': final_prediction[0],
                'confidence': final_prediction[1],
                'ensemble_agreement': len(set(predictions)) == 1,  # All models agree
                'individual_predictions': predictions,
                'individual_confidences': confidences
            }

def create_advanced_classifier(model_type: str = 'vit', num_classes: int = 10, 
                             input_channels: int = 1, **kwargs) -> nn.Module:
    """Factory function to create advanced classifiers."""
    
    if model_type == 'vit':
        return VisionTransformer(
            num_classes=num_classes,
            channels=input_channels,
            **kwargs
        )
    elif model_type == 'efficientnet':
        return EfficientNetAstronomical(
            num_classes=num_classes,
            input_channels=input_channels
        )
    elif model_type == 'vae':
        return VariationalAutoencoder(
            input_channels=input_channels,
            **kwargs
        )
    elif model_type == 'self_supervised':
        return SelfSupervisedAnomalyDetector(
            input_channels=input_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    """Test the advanced models."""
    print("🧪 Testing Advanced Models...")
    
    # Test Vision Transformer
    print("Testing Vision Transformer...")
    vit = VisionTransformer(image_size=512, patch_size=16, num_classes=10, dim=768, depth=6)
    test_input = torch.randn(1, 1, 512, 512)
    output = vit(test_input)
    print(f"ViT output shape: {output.shape}")
    
    # Test EfficientNet
    print("Testing EfficientNet...")
    efficientnet = EfficientNetAstronomical(num_classes=10, input_channels=1)
    output = efficientnet(test_input)
    print(f"EfficientNet output shape: {output.shape}")
    
    # Test VAE
    print("Testing Variational Autoencoder...")
    vae = VariationalAutoencoder(input_channels=1, latent_dim=128)
    recon, mu, log_var = vae(test_input)
    print(f"VAE reconstruction shape: {recon.shape}")
    print(f"VAE latent mu shape: {mu.shape}")
    
    # Test Self-Supervised
    print("Testing Self-Supervised Anomaly Detector...")
    ssad = SelfSupervisedAnomalyDetector(input_channels=1, feature_dim=512)
    features, projection, anomaly_score = ssad(test_input)
    print(f"Self-Supervised features shape: {features.shape}")
    print(f"Anomaly score shape: {anomaly_score.shape}")
    
    print("✅ All advanced models tested successfully!")

if __name__ == "__main__":
    main()
