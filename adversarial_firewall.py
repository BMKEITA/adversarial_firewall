# ================================================================
# adversarial_firewall.py
# Multi-Class Adversarial Firewall with Adaptive Defense Selection
# Lightweight modular defense layer for upstream attack classification
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import json
import datetime
import warnings
import sys
import math
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import defaultdict, OrderedDict
from scipy import fftpack
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import pickle
import time
import hashlib
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adversarial_firewall.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== FIREWALL CONFIGURATION ====================
class AdversarialFirewallConfig:
    """Configuration for Multi-Class Adversarial Firewall"""
    
    def __init__(self, dataset_name: str = 'CIFAR10'):
        self.dataset_name = dataset_name
        
        # Input specifications
        if dataset_name == 'MNIST':
            self.image_size = 32
            self.in_channels = 3
            self.num_classes = 10
            self.main_model_classes = 10
        elif dataset_name == 'CIFAR10':
            self.image_size = 32
            self.in_channels = 3
            self.num_classes = 10
            self.main_model_classes = 10
        elif dataset_name == 'CIFAR100':
            self.image_size = 32
            self.in_channels = 3
            self.num_classes = 100
            self.main_model_classes = 100
        else:
            self.image_size = 224
            self.in_channels = 3
            self.num_classes = 1000
            self.main_model_classes = 1000
        
        # Firewall architecture - ENHANCED
        self.firewall_channels = [16, 32, 64]
        self.hidden_dim = 256  # Increased for better capacity
        self.dropout_rate = 0.3
        
        # Feature extraction
        self.dct_block_size = 8  # Increased for better frequency analysis
        self.num_dct_coefficients = 12  # Increased
        self.lhe_window = 7
        
        # Attack classes - Enhanced with more attack types
        self.attack_classes = {
            0: 'Clean',
            1: 'FGSM',
            2: 'PGD',
            3: 'Noise',
            4: 'MIA',
            5: 'CW',  # Carlini-Wagner attack
            6: 'DeepFool',
            7: 'SPSA',
            8: 'Boundary',
            9: 'JSMA',
        }
        
        # Defense strategies for each attack
        self.defense_strategies = {
            'Clean': 'forward',
            'FGSM': 'denoise_fgsm',
            'PGD': 'adversarial_training',
            'Noise': 'filter_noise',
            'MIA': 'feature_squeezing',
            'CW': 'certified_defense',
            'DeepFool': 'gradient_masking',
            'SPSA': 'ensemble_defense',
            'Boundary': 'randomization',
            'JSMA': 'feature_squeezing',
        }
        
        # Defense parameters - Enhanced with more options
        self.defense_params = {
            'denoise_fgsm': {'filter_size': 3, 'sigma': 1.0, 'method': 'gaussian'},
            'filter_noise': {'threshold': 0.1, 'kernel_size': 3, 'method': 'median'},
            'feature_squeezing': {'bit_depth': 4, 'smooth': True},
            'adversarial_training': {'eps': 0.3, 'alpha': 0.01, 'steps': 7},
            'certified_defense': {'radius': 0.5, 'method': 'randomized_smoothing'},
            'gradient_masking': {'mask_strength': 0.5, 'noise_level': 0.1},
            'ensemble_defense': {'n_models': 3, 'method': 'majority_voting'},
            'randomization': {'scale_range': (0.9, 1.1), 'rotate_range': (-5, 5)}
        }
        
        # Performance thresholds
        self.confidence_threshold = 0.85
        self.block_threshold = 0.85  # Lowered to catch more attacks
        self.defense_threshold = 0.70  # Threshold for applying defense
        
        # Training
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.num_epochs = 25  # Increased for better training
        self.patience = 8
        self.weight_decay = 1e-4
        
        # Real-time requirements
        self.max_inference_time = 20  # Increased slightly for better accuracy
        self.max_memory_mb = 100  # Increased for enhanced model
        
        # Data augmentation
        self.augmentation = {
            'random_crop': True,
            'horizontal_flip': True,
            'color_jitter': 0.1,
            'rotation': 5,
            'scale': (0.9, 1.1)
        }
        
        # Monitoring
        self.save_checkpoints = True
        self.checkpoint_interval = 5
        self.log_interval = 10
        
    def __str__(self):
        return f"AdversarialFirewallConfig(dataset={self.dataset_name}, attacks={len(self.attack_classes)})"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return self.__dict__.copy()
    
    def save(self, path: str):
        """Save config to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls(data.get('dataset_name', 'CIFAR10'))
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_attack_classes_for_num_classes(self, num_classes: int) -> Dict[int, str]:
        """Get attack classes for a specific number of classes"""
        if num_classes > len(self.attack_classes):
            raise ValueError(f"Requested {num_classes} classes but only {len(self.attack_classes)} are available")
        
        return {i: self.attack_classes[i] for i in range(num_classes)}

# ==================== ENHANCED FEATURE EXTRACTORS ====================
class EnhancedSpatialExtractor:
    """Enhanced spatial feature extractor with better discriminative power"""
    
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.feature_names = []
    
    def _safe_laplacian(self, image: np.ndarray) -> np.ndarray:
        """Safe Laplacian computation with error handling"""
        try:
            # Check if image is single channel or multi-channel
            if len(image.shape) == 3:
                # Apply Laplacian to each channel and average
                laplacians = []
                for c in range(image.shape[2]):
                    laplacian = cv2.Laplacian(image[:, :, c], cv2.CV_32F)
                    laplacians.append(laplacian)
                return np.mean(laplacians, axis=0)
            else:
                return cv2.Laplacian(image, cv2.CV_32F)
        except cv2.error as e:
            logger.warning(f"Laplacian computation failed: {e}. Using Sobel fallback.")
            # Fallback: compute using Sobel derivatives
            if len(image.shape) == 3:
                sobel_result = np.zeros_like(image, dtype=np.float32)
                for c in range(image.shape[2]):
                    sobelx = cv2.Sobel(image[:, :, c], cv2.CV_32F, 2, 0, ksize=3)
                    sobely = cv2.Sobel(image[:, :, c], cv2.CV_32F, 0, 2, ksize=3)
                    sobel_result[:, :, c] = sobelx + sobely
                return np.mean(sobel_result, axis=2)
            else:
                sobelx = cv2.Sobel(image, cv2.CV_32F, 2, 0, ksize=3)
                sobely = cv2.Sobel(image, cv2.CV_32F, 0, 2, ksize=3)
                return sobelx + sobely
    
    def _extract_features_with_names(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract features with names for interpretability"""
        try:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            
            # Handle input shape
            if len(image.shape) == 3:
                if image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale for efficiency if it's RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # If already grayscale or single channel
                gray = image.squeeze()
            
            # Normalize to [0, 1]
            if gray.max() > 1.0:
                gray = gray / 255.0
            
            # Apply histogram equalization for better contrast
            if gray.max() <= 1.0:
                gray_uint8 = (gray * 255).astype(np.uint8)
            else:
                gray_uint8 = gray.astype(np.uint8)
            
            enhanced = cv2.equalizeHist(gray_uint8)
            enhanced_float = enhanced.astype(np.float32) / 255.0
            
            features = []
            feature_names = []
            
            # 1. Basic intensity statistics
            features.append(np.mean(gray))
            features.append(np.std(gray))
            features.append(np.max(gray) - np.min(gray))  # Dynamic range
            features.append(np.percentile(gray, 25))
            features.append(np.percentile(gray, 75))
            feature_names.extend(['intensity_mean', 'intensity_std', 'dynamic_range', 
                                 'percentile_25', 'percentile_75'])
            
            # 2. Gradient features (edge information)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            
            features.append(np.mean(grad_mag))
            features.append(np.std(grad_mag))
            features.append(np.max(grad_mag))
            features.append(np.percentile(grad_mag, 90))  # Strong edges
            feature_names.extend(['gradient_mean', 'gradient_std', 'gradient_max', 'gradient_90p'])
            
            # Gradient orientation statistics
            grad_angle = np.arctan2(sobely, sobelx + 1e-8)
            features.append(np.std(np.cos(grad_angle)))
            features.append(np.std(np.sin(grad_angle)))
            feature_names.extend(['grad_orient_cos_std', 'grad_orient_sin_std'])
            
            # 3. Texture features using Laplacian
            laplacian = self._safe_laplacian(gray)
            features.append(np.mean(np.abs(laplacian)))
            features.append(np.std(laplacian))
            features.append(np.max(np.abs(laplacian)))
            feature_names.extend(['laplacian_mean', 'laplacian_std', 'laplacian_max'])
            
            # 4. Local contrast
            local_mean = cv2.boxFilter(gray, cv2.CV_32F, (5, 5))
            local_std = cv2.boxFilter(gray**2, cv2.CV_32F, (5, 5))
            local_std = np.sqrt(np.maximum(local_std - local_mean**2, 0))
            features.append(np.mean(local_std))
            feature_names.append('local_contrast_mean')
            
            # 5. Histogram features (multiple bins)
            hist_8 = cv2.calcHist([enhanced], [0], None, [8], [0, 256]).flatten()
            hist_8 = hist_8 / (hist_8.sum() + 1e-8)
            features.extend(hist_8[:6].tolist())  # First 6 bins
            feature_names.extend([f'hist_bin_{i}' for i in range(6)])
            
            # 6. Edge density (Canny edges)
            edges = cv2.Canny(enhanced, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            feature_names.append('edge_density')
            
            # 7. Noise estimation (high-frequency content)
            kernel_hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
            high_freq = cv2.filter2D(gray, cv2.CV_32F, kernel_hp)
            features.append(np.std(high_freq))
            features.append(np.mean(np.abs(high_freq)))
            feature_names.extend(['high_freq_std', 'high_freq_mean'])
            
            # 8. Entropy of local patches
            def local_entropy(patch):
                hist = cv2.calcHist([patch], [0], None, [16], [0, 256])
                hist = hist / (hist.sum() + 1e-8)
                return -np.sum(hist * np.log(hist + 1e-8))
            
            # Sample patches for entropy
            h, w = gray.shape
            patch_size = 8
            entropies = []
            for i in range(0, h-patch_size, patch_size):
                for j in range(0, w-patch_size, patch_size):
                    patch = enhanced[i:i+patch_size, j:j+patch_size]
                    if patch.size > 0:
                        entropies.append(local_entropy(patch))
            
            if entropies:
                features.append(np.mean(entropies))
                features.append(np.std(entropies))
                feature_names.extend(['local_entropy_mean', 'local_entropy_std'])
            else:
                features.extend([0, 0])
                feature_names.extend(['local_entropy_mean', 'local_entropy_std'])
            
            return np.array(features, dtype=np.float32), feature_names
            
        except Exception as e:
            logger.error(f"Error in spatial feature extraction: {e}")
            # Return zero features with appropriate dimensions
            feature_count = 45  # Based on the features we extract
            return np.zeros(feature_count, dtype=np.float32), [f'error_feature_{i}' for i in range(feature_count)]
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract enhanced spatial features"""
        features, _ = self._extract_features_with_names(image)
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        if not self.feature_names:
            # Extract from dummy image to populate feature names
            dummy = np.random.randn(32, 32).astype(np.float32)
            _, self.feature_names = self._extract_features_with_names(dummy)
        return self.feature_names
    
    def visualize_features(self, image: np.ndarray, save_path: Optional[str] = None):
        """Visualize spatial features"""
        try:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            
            # Handle input shape
            if len(image.shape) == 3:
                if image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.squeeze()
            
            # Normalize to [0, 1]
            if gray.max() > 1.0:
                gray = gray / 255.0
            
            # Apply histogram equalization
            gray_uint8 = (gray * 255).astype(np.uint8)
            enhanced = cv2.equalizeHist(gray_uint8)
            
            # Compute visualizations
            # Gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
            
            # Laplacian
            laplacian = self._safe_laplacian(gray)
            laplacian_abs = np.abs(laplacian)
            laplacian_abs = (laplacian_abs / laplacian_abs.max() * 255).astype(np.uint8)
            
            # Edges (Canny)
            edges = cv2.Canny(enhanced, 50, 150)
            
            # Local contrast
            local_mean = cv2.boxFilter(gray, cv2.CV_32F, (5, 5))
            local_std = cv2.boxFilter(gray**2, cv2.CV_32F, (5, 5))
            local_std = np.sqrt(np.maximum(local_std - local_mean**2, 0))
            local_std = (local_std / local_std.max() * 255).astype(np.uint8)
            
            # High frequency content
            kernel_hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
            high_freq = cv2.filter2D(gray, cv2.CV_32F, kernel_hp)
            high_freq_abs = np.abs(high_freq)
            high_freq_abs = (high_freq_abs / high_freq_abs.max() * 255).astype(np.uint8)
            
            # Create visualization grid
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Original
            axes[0, 0].imshow(gray, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Gradient magnitude
            axes[0, 1].imshow(grad_mag, cmap='gray')
            axes[0, 1].set_title('Gradient Magnitude')
            axes[0, 1].axis('off')
            
            # Laplacian
            axes[0, 2].imshow(laplacian_abs, cmap='gray')
            axes[0, 2].set_title('Laplacian (Edge Detection)')
            axes[0, 2].axis('off')
            
            # Canny edges
            axes[1, 0].imshow(edges, cmap='gray')
            axes[1, 0].set_title('Canny Edges')
            axes[1, 0].axis('off')
            
            # Local contrast
            axes[1, 1].imshow(local_std, cmap='gray')
            axes[1, 1].set_title('Local Contrast (Std)')
            axes[1, 1].axis('off')
            
            # High frequency
            axes[1, 2].imshow(high_freq_abs, cmap='gray')
            axes[1, 2].set_title('High Frequency Content')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Spatial features visualization saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing spatial features: {e}")
            import traceback
            traceback.print_exc()


class EnhancedFrequencyExtractor:
    """Enhanced frequency feature extractor"""
    
    def __init__(self, block_size: int = 8, num_coefficients: int = 12):
        self.block_size = block_size
        self.num_coefficients = num_coefficients
        self.feature_names = []
    
    def _extract_features_with_names(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract features with names for interpretability"""
        try:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            
            if len(image.shape) == 3:
                if image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.squeeze()
            
            # Normalize
            if gray.max() > 1.0:
                gray = gray / 255.0
            
            # Ensure proper data type for DCT
            gray = gray.astype(np.float64)
            
            height, width = gray.shape
            features = []
            feature_names = []
            
            # Extract DCT from multiple blocks with overlap
            dct_coeffs_all = []
            
            stride = max(1, self.block_size // 2)  # 50% overlap, at least 1
            for i in range(0, height - self.block_size + 1, stride):
                for j in range(0, width - self.block_size + 1, stride):
                    block = gray[i:i+self.block_size, j:j+self.block_size]
                    if block.shape == (self.block_size, self.block_size):
                        # Apply DCT
                        try:
                            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                            dct_coeffs_all.append(dct_block.flatten())
                        except Exception as e:
                            logger.warning(f"DCT computation failed: {e}")
                            # Use zeros as fallback
                            dct_coeffs_all.append(np.zeros(self.block_size * self.block_size))
            
            if dct_coeffs_all:
                dct_coeffs_all = np.array(dct_coeffs_all)
                
                # 1. DC component statistics (average intensity)
                dc_values = dct_coeffs_all[:, 0]
                features.append(np.mean(dc_values))
                features.append(np.std(dc_values))
                features.append(np.max(dc_values) - np.min(dc_values))
                feature_names.extend(['dct_dc_mean', 'dct_dc_std', 'dct_dc_range'])
                
                # 2. Low frequency coefficients (first few AC components)
                low_freq = dct_coeffs_all[:, 1:min(self.num_coefficients, dct_coeffs_all.shape[1])]
                features.append(np.mean(np.abs(low_freq)))
                features.append(np.std(low_freq))
                feature_names.extend(['dct_low_freq_mean', 'dct_low_freq_std'])
                
                # 3. Frequency energy distribution
                total_energy = np.sum(dct_coeffs_all**2, axis=1)
                low_freq_energy = np.sum(dct_coeffs_all[:, :4]**2, axis=1)
                mid_freq_energy = np.sum(dct_coeffs_all[:, 4:min(16, dct_coeffs_all.shape[1])]**2, axis=1)
                high_freq_energy = np.sum(dct_coeffs_all[:, min(16, dct_coeffs_all.shape[1]):]**2, axis=1)
                
                energy_ratio_low = np.mean(low_freq_energy / (total_energy + 1e-8))
                energy_ratio_mid = np.mean(mid_freq_energy / (total_energy + 1e-8))
                energy_ratio_high = np.mean(high_freq_energy / (total_energy + 1e-8))
                
                features.append(energy_ratio_low)
                features.append(energy_ratio_mid)
                features.append(energy_ratio_high)
                feature_names.extend(['energy_ratio_low', 'energy_ratio_mid', 'energy_ratio_high'])
                
                # 4. Frequency asymmetry (difference between horizontal and vertical frequencies)
                # Reshape to get 2D frequency indices
                if len(dct_coeffs_all) > 0:
                    dct_block_example = dct_coeffs_all[0].reshape(self.block_size, self.block_size)
                    horiz_freq = np.mean(np.abs(dct_block_example[0, 1:]))
                    vert_freq = np.mean(np.abs(dct_block_example[1:, 0]))
                    features.append(horiz_freq - vert_freq)
                else:
                    features.append(0)
                feature_names.append('freq_asymmetry_hv')
                
                # 5. Frequency concentration (ratio of top coefficients to total)
                if len(dct_coeffs_all) > 0:
                    sorted_coeffs = np.sort(np.abs(dct_coeffs_all.flatten()))[::-1]
                    top_10_percent = max(1, int(0.1 * len(sorted_coeffs)))
                    concentration = np.sum(sorted_coeffs[:top_10_percent]) / (np.sum(sorted_coeffs) + 1e-8)
                    features.append(concentration)
                else:
                    features.append(0)
                feature_names.append('freq_concentration')
                
                # 6. Frequency entropy
                if len(dct_coeffs_all) > 0:
                    freq_magnitudes = np.abs(dct_coeffs_all).flatten()
                    freq_magnitudes = freq_magnitudes / (freq_magnitudes.sum() + 1e-8)
                    freq_entropy = -np.sum(freq_magnitudes * np.log(freq_magnitudes + 1e-8))
                    features.append(freq_entropy)
                else:
                    features.append(0)
                feature_names.append('freq_entropy')
            
            return np.array(features, dtype=np.float32), feature_names
            
        except Exception as e:
            logger.error(f"Error in frequency feature extraction: {e}")
            # Return zero features with appropriate dimensions
            feature_count = 13  # Based on the features we extract
            return np.zeros(feature_count, dtype=np.float32), [f'freq_error_feature_{i}' for i in range(feature_count)]
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract enhanced frequency features"""
        features, _ = self._extract_features_with_names(image)
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        if not self.feature_names:
            # Extract from dummy image to populate feature names
            dummy = np.random.randn(32, 32).astype(np.float32)
            _, self.feature_names = self._extract_features_with_names(dummy)
        return self.feature_names
    
    def visualize_features(self, image: np.ndarray, save_path: Optional[str] = None):
        """Visualize frequency features using DCT"""
        try:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            
            if len(image.shape) == 3:
                if image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.squeeze()
            
            # Normalize
            if gray.max() > 1.0:
                gray = gray / 255.0
            
            # Ensure proper data type
            gray = gray.astype(np.float64)
            
            # Apply DCT to entire image
            dct_image = fftpack.dct(fftpack.dct(gray.T, norm='ortho').T, norm='ortho')
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Original image
            axes[0, 0].imshow(gray, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Full DCT spectrum (log scale for better visualization)
            dct_log = np.log(np.abs(dct_image) + 1)
            axes[0, 1].imshow(dct_log, cmap='hot')
            axes[0, 1].set_title('DCT Spectrum (Log Scale)')
            axes[0, 1].axis('off')
            
            # Low frequencies (first 8x8 coefficients)
            low_freq_mask = np.zeros_like(dct_image)
            low_freq_mask[:8, :8] = 1
            dct_low = dct_image * low_freq_mask
            axes[0, 2].imshow(np.abs(dct_low), cmap='hot')
            axes[0, 2].set_title('Low Frequencies (8x8)')
            axes[0, 2].axis('off')
            
            # Mid frequencies
            mid_freq_mask = np.zeros_like(dct_image)
            mid_freq_mask[8:16, 8:16] = 1
            dct_mid = dct_image * mid_freq_mask
            axes[1, 0].imshow(np.abs(dct_mid), cmap='hot')
            axes[1, 0].set_title('Mid Frequencies (8x8)')
            axes[1, 0].axis('off')
            
            # High frequencies
            h, w = dct_image.shape
            high_freq_mask = np.ones_like(dct_image)
            high_freq_mask[:16, :16] = 0
            dct_high = dct_image * high_freq_mask
            axes[1, 1].imshow(np.abs(dct_high), cmap='hot')
            axes[1, 1].set_title('High Frequencies')
            axes[1, 1].axis('off')
            
            # Energy distribution bar chart
            total_energy = np.sum(dct_image**2)
            low_energy = np.sum(dct_image[:8, :8]**2)
            mid_energy = np.sum(dct_image[8:16, 8:16]**2)
            high_energy = total_energy - low_energy - mid_energy
            
            energies = [low_energy/total_energy, mid_energy/total_energy, high_energy/total_energy]
            labels = ['Low Freq', 'Mid Freq', 'High Freq']
            
            axes[1, 2].bar(labels, energies, color=['blue', 'green', 'red'])
            axes[1, 2].set_title('Frequency Energy Distribution')
            axes[1, 2].set_ylabel('Energy Ratio')
            axes[1, 2].set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Frequency features visualization saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing frequency features: {e}")
            import traceback
            traceback.print_exc()


# ==================== ENHANCED FIREWALL NETWORK ====================
class EnhancedFirewallNetwork(nn.Module):
    """Enhanced neural network for attack classification"""
    
    def __init__(self, config: AdversarialFirewallConfig, num_classes: int = None):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use num_classes if provided, otherwise use from config
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            # Determine based on training mode vs inference mode
            self.num_classes = 5  # Default for training (Clean + 4 attacks)
        
        # Feature dimensions
        self.spatial_extractor = EnhancedSpatialExtractor(config.lhe_window)
        self.freq_extractor = EnhancedFrequencyExtractor(
            config.dct_block_size, config.num_dct_coefficients
        )
        
        # Test feature dimensions
        dummy_input = torch.randn(1, config.in_channels, config.image_size, config.image_size)
        spatial_feat = self.spatial_extractor.extract_features(dummy_input[0])
        freq_feat = self.freq_extractor.extract_features(dummy_input[0])
        
        self.spatial_feat_dim = len(spatial_feat)
        self.freq_feat_dim = len(freq_feat)
        total_feat_dim = self.spatial_feat_dim + self.freq_feat_dim
        
        logger.info(f"\n[EnhancedFirewallNetwork] Feature dimensions:")
        logger.info(f"  Spatial features: {self.spatial_feat_dim}")
        logger.info(f"  Frequency features: {self.freq_feat_dim}")
        logger.info(f"  Total features: {total_feat_dim}")
        logger.info(f"  Output classes: {self.num_classes}")
        
        # Enhanced MLP with batch normalization and residual connections
        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(config.dropout_rate * 0.8),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Linear(64, self.num_classes)
        )
        
        # Confidence calibration with learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Start with higher temp
        
        # Initialize weights
        self._initialize_weights()
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Verify real-time requirements
        if total_params * 4 / 1024 / 1024 > config.max_memory_mb:
            logger.warning(f"  WARNING: Model exceeds memory limit!")
        
        # Move to device
        self.to(self.device)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for a batch of images"""
        batch_size = images.shape[0]
        all_features = []
        
        # Vectorized feature extraction where possible
        chunk_size = min(32, batch_size)  # Adaptive chunk size
        for i in range(0, batch_size, chunk_size):
            chunk_end = min(i + chunk_size, batch_size)
            chunk_features = []
            
            for j in range(i, chunk_end):
                img_np = images[j].cpu().numpy()
                
                # Extract features
                spatial_feat = self.spatial_extractor.extract_features(img_np)
                freq_feat = self.freq_extractor.extract_features(img_np)
                
                # Ensure feature dimensions match
                if len(spatial_feat) != self.spatial_feat_dim:
                    spatial_feat = np.zeros(self.spatial_feat_dim, dtype=np.float32)
                if len(freq_feat) != self.freq_feat_dim:
                    freq_feat = np.zeros(self.freq_feat_dim, dtype=np.float32)
                
                # Combine
                combined = np.concatenate([spatial_feat, freq_feat])
                chunk_features.append(combined)
            
            all_features.append(np.array(chunk_features))
        
        if all_features:
            features_array = np.vstack(all_features)
            return torch.FloatTensor(features_array).to(images.device)
        else:
            # Return zeros if no features extracted
            return torch.zeros(batch_size, self.spatial_feat_dim + self.freq_feat_dim).to(images.device)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with confidence calibration"""
        # Extract features
        features = self.extract_features(images)
        
        # Classification
        logits = self.classifier(features)
        
        # Temperature scaling for confidence calibration
        calibrated_logits = logits / (self.temperature + 1e-8)
        
        # Get probabilities and predictions
        probs = F.softmax(calibrated_logits, dim=1)
        confidence, preds = torch.max(probs, dim=1)
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': preds,
            'confidence': confidence,
            'features': features
        }


# ==================== DEFENSE MODULES ====================
class DefenseModule:
    """Modular defense strategies"""
    
    def __init__(self, config: AdversarialFirewallConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def denoise_fgsm(self, images: torch.Tensor) -> torch.Tensor:
        """Denoising defense for FGSM attacks"""
        params = self.config.defense_params['denoise_fgsm']
        
        # Apply Gaussian filter
        kernel_size = params['filter_size']
        sigma = params['sigma']
        method = params.get('method', 'gaussian')
        
        # Convert to numpy for OpenCV processing
        batch_size = images.shape[0]
        denoised = []
        
        for i in range(batch_size):
            try:
                img_np = images[i].cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # CHW to HWC
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Apply denoising based on method
                if method == 'gaussian':
                    # Ensure odd kernel size
                    kernel_size_adj = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                    denoised_np = cv2.GaussianBlur(img_np, (kernel_size_adj, kernel_size_adj), sigma)
                elif method == 'bilateral':
                    denoised_np = cv2.bilateralFilter(img_np, kernel_size, sigma*75, sigma*75)
                elif method == 'median':
                    # Ensure odd kernel size
                    kernel_size_adj = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                    if len(img_np.shape) == 3:
                        denoised_np = np.zeros_like(img_np)
                        for c in range(img_np.shape[2]):
                            denoised_np[:, :, c] = cv2.medianBlur(img_np[:, :, c], kernel_size_adj)
                    else:
                        denoised_np = cv2.medianBlur(img_np, kernel_size_adj)
                else:
                    kernel_size_adj = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                    denoised_np = cv2.GaussianBlur(img_np, (kernel_size_adj, kernel_size_adj), sigma)
                
                if len(denoised_np.shape) == 3 and denoised_np.shape[2] in [1, 3]:
                    denoised_np = np.transpose(denoised_np, (2, 0, 1))
                
                denoised.append(torch.from_numpy(denoised_np).float())
            except Exception as e:
                logger.warning(f"Error in denoising image {i}: {e}")
                denoised.append(images[i].cpu())  # Return original as fallback
        
        return torch.stack(denoised).to(images.device)
    
    def filter_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Filtering defense for noise attacks"""
        params = self.config.defense_params['filter_noise']
        threshold = params['threshold']
        method = params.get('method', 'median')
        
        batch_size = images.shape[0]
        filtered = []
        
        for i in range(batch_size):
            try:
                img_np = images[i].cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # CHW to HWC
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Apply filter based on method
                kernel_size = params['kernel_size']
                kernel_size_adj = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                
                if method == 'median':
                    if len(img_np.shape) == 3:
                        filtered_np = np.zeros_like(img_np)
                        for c in range(img_np.shape[2]):
                            filtered_np[:, :, c] = cv2.medianBlur(img_np[:, :, c], kernel_size_adj)
                    else:
                        filtered_np = cv2.medianBlur(img_np, kernel_size_adj)
                elif method == 'gaussian':
                    filtered_np = cv2.GaussianBlur(img_np, (kernel_size_adj, kernel_size_adj), 1.0)
                elif method == 'nlm':
                    # Non-local means denoising
                    if len(img_np.shape) == 3:
                        filtered_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
                    else:
                        filtered_np = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
                else:
                    if len(img_np.shape) == 3:
                        filtered_np = np.zeros_like(img_np)
                        for c in range(img_np.shape[2]):
                            filtered_np[:, :, c] = cv2.medianBlur(img_np[:, :, c], kernel_size_adj)
                    else:
                        filtered_np = cv2.medianBlur(img_np, kernel_size_adj)
                
                if len(filtered_np.shape) == 3 and filtered_np.shape[2] in [1, 3]:
                    filtered_np = np.transpose(filtered_np, (2, 0, 1))
                
                filtered.append(torch.from_numpy(filtered_np).float())
            except Exception as e:
                logger.warning(f"Error in filtering image {i}: {e}")
                filtered.append(images[i].cpu())  # Return original as fallback
        
        return torch.stack(filtered).to(images.device)
    
    def feature_squeezing(self, images: torch.Tensor) -> torch.Tensor:
        """Feature squeezing defense for MIA attacks"""
        params = self.config.defense_params['feature_squeezing']
        bit_depth = params['bit_depth']
        smooth = params.get('smooth', True)
        
        # Reduce bit depth with smoothing
        levels = 2 ** bit_depth
        
        # Apply smoothing before quantization
        squeezed = images.clone()
        for i in range(images.shape[1]):  # Channel dimension
            channel = images[:, i:i+1, :, :]
            
            if smooth:
                # Small Gaussian blur before quantization
                try:
                    channel_np = channel.cpu().numpy()
                    smoothed = np.zeros_like(channel_np)
                    for b in range(channel_np.shape[0]):
                        smoothed[b, 0] = cv2.GaussianBlur(channel_np[b, 0], (3, 3), 0.5)
                    channel = torch.from_numpy(smoothed).to(images.device)
                except Exception as e:
                    logger.warning(f"Error in smoothing channel {i}: {e}")
            
            # Quantize
            squeezed[:, i:i+1, :, :] = torch.round(channel * (levels - 1)) / (levels - 1)
        
        return squeezed
    
    def adversarial_training(self, images: torch.Tensor, model: nn.Module, 
                           labels: torch.Tensor) -> torch.Tensor:
        """Adversarial training defense for PGD attacks"""
        params = self.config.defense_params['adversarial_training']
        eps = params['eps']
        alpha = params['alpha']
        steps = params.get('steps', 7)
        
        # Generate adversarial examples for training
        images_adv = images.clone().detach()
        model.eval()
        
        for _ in range(steps):
            images_adv.requires_grad = True
            outputs = model(images_adv)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, images_adv)[0]
            
            images_adv = images_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(images_adv - images, -eps, eps)
            images_adv = torch.clamp(images + delta, 0, 1).detach()
        
        return images_adv
    
    def certified_defense(self, images: torch.Tensor) -> torch.Tensor:
        """Certified defense for CW attacks"""
        params = self.config.defense_params['certified_defense']
        radius = params['radius']
        method = params.get('method', 'randomized_smoothing')
        
        if method == 'randomized_smoothing':
            # Add Gaussian noise
            noise = torch.randn_like(images) * radius
            return torch.clamp(images + noise, 0, 1)
        else:
            # Default: just add noise
            noise = torch.randn_like(images) * radius * 0.5
            return torch.clamp(images + noise, 0, 1)
    
    def gradient_masking(self, images: torch.Tensor) -> torch.Tensor:
        """Gradient masking defense for DeepFool attacks"""
        params = self.config.defense_params['gradient_masking']
        mask_strength = params['mask_strength']
        noise_level = params.get('noise_level', 0.1)
        
        # Add randomized transformations
        masked = images.clone()
        
        # Add small noise
        noise = torch.randn_like(masked) * noise_level * mask_strength
        masked = masked + noise
        
        # Random scaling
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1 * mask_strength
        masked = F.interpolate(masked, scale_factor=scale, mode='bilinear', align_corners=False)
        
        # Ensure same size
        if masked.shape != images.shape:
            masked = F.interpolate(masked, size=images.shape[2:], mode='bilinear', align_corners=False)
        
        return torch.clamp(masked, 0, 1)
    
    def ensemble_defense(self, images: torch.Tensor, models: List[nn.Module]) -> torch.Tensor:
        """Ensemble defense for SPSA attacks"""
        params = self.config.defense_params['ensemble_defense']
        n_models = params['n_models']
        method = params.get('method', 'majority_voting')
        
        if not models or len(models) < n_models:
            logger.warning(f"Need at least {n_models} models for ensemble defense, using randomization")
            return self.randomization_defense(images)
        
        # Implement actual ensemble voting
        batch_size = images.shape[0]
        
        if method == 'majority_voting':
            # Get predictions from all models
            all_preds = []
            for model in models[:n_models]:
                model.eval()
                with torch.no_grad():
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds)
            
            # Stack predictions
            all_preds = torch.stack(all_preds, dim=0)
            
            # Get majority vote for each sample
            majority_votes = []
            for i in range(batch_size):
                sample_preds = all_preds[:, i]
                vote = torch.mode(sample_preds).values.item()
                majority_votes.append(vote)
            
            # Apply different defenses based on consensus
            # For simplicity, we'll apply randomization based on variance
            pred_variance = torch.var(all_preds.float(), dim=0).mean().item()
            if pred_variance > 0.5:  # High disagreement
                # Apply stronger defense
                return self.randomization_defense(images)
            else:
                # Apply mild defense
                return images
        else:
            return images
    
    def randomization_defense(self, images: torch.Tensor) -> torch.Tensor:
        """Randomization defense"""
        params = self.config.defense_params.get('randomization', {'scale_range': (0.9, 1.1), 'rotate_range': (-5, 5)})
        scale_range = params.get('scale_range', (0.9, 1.1))
        rotate_range = params.get('rotate_range', (-5, 5))
        
        batch_size = images.shape[0]
        randomized = []
        
        for i in range(batch_size):
            try:
                img_np = images[i].cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # CHW to HWC
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                h, w = img_np.shape[:2]
                
                # Random scaling
                scale = np.random.uniform(scale_range[0], scale_range[1])
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(img_np, (new_w, new_h))
                
                # Random rotation
                angle = np.random.uniform(rotate_range[0], rotate_range[1])
                center = (new_w // 2, new_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(scaled, M, (new_w, new_h))
                
                # Resize back to original
                resized = cv2.resize(rotated, (w, h))
                
                if len(resized.shape) == 3 and resized.shape[2] in [1, 3]:
                    resized = np.transpose(resized, (2, 0, 1))
                
                randomized.append(torch.from_numpy(resized).float())
            except Exception as e:
                logger.warning(f"Error in randomizing image {i}: {e}")
                randomized.append(images[i].cpu())  # Return original as fallback
        
        return torch.stack(randomized).to(images.device)
    
    def get_defense(self, attack_type: str) -> callable:
        """Get appropriate defense function"""
        defense_name = self.config.defense_strategies.get(attack_type, 'forward')
        
        if defense_name == 'denoise_fgsm':
            return self.denoise_fgsm
        elif defense_name == 'filter_noise':
            return self.filter_noise
        elif defense_name == 'feature_squeezing':
            return self.feature_squeezing
        elif defense_name == 'adversarial_training':
            # Note: adversarial_training requires model and labels
            return lambda x, model=None, labels=None: self.adversarial_training(x, model, labels) if model is not None else x
        elif defense_name == 'certified_defense':
            return self.certified_defense
        elif defense_name == 'gradient_masking':
            return self.gradient_masking
        elif defense_name == 'ensemble_defense':
            # Return a lambda that takes models parameter
            return lambda x, models=None: self.ensemble_defense(x, models if models else [])
        elif defense_name == 'randomization':
            return self.randomization_defense
        else:
            return lambda x: x  # Identity (forward)
    
    def visualize_defense_effects(self, original_images: torch.Tensor, attack_type: str, save_path: Optional[str] = None):
        """Visualize defense effects on adversarial images"""
        try:
            import matplotlib.pyplot as plt
            
            # Get defense function
            defense_func = self.get_defense(attack_type)
            
            # Apply defense
            if defense_func.__name__ == '<lambda>' and 'adversarial_training' in str(defense_func):
                # For adversarial training, we need model and labels
                logger.warning("Cannot visualize adversarial_training defense without model and labels")
                return
            else:
                defended_images = defense_func(original_images)
            
            # Convert to numpy for visualization
            batch_size = min(3, original_images.shape[0])
            
            fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
            
            if batch_size == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(batch_size):
                # Original image
                orig_np = original_images[i].cpu().numpy()
                if orig_np.shape[0] in [1, 3]:
                    orig_np = np.transpose(orig_np, (1, 2, 0))
                
                # Defended image
                def_np = defended_images[i].cpu().numpy()
                if def_np.shape[0] in [1, 3]:
                    def_np = np.transpose(def_np, (1, 2, 0))
                
                # Difference
                diff = np.abs(def_np - orig_np)
                diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff
                
                # Plot original
                axes[i, 0].imshow(orig_np if orig_np.shape[2] == 3 else orig_np.squeeze(), 
                                cmap='gray' if len(orig_np.shape) == 2 else None)
                axes[i, 0].set_title(f'Original (Attack: {attack_type})')
                axes[i, 0].axis('off')
                
                # Plot defended
                axes[i, 1].imshow(def_np if def_np.shape[2] == 3 else def_np.squeeze(), 
                                cmap='gray' if len(def_np.shape) == 2 else None)
                axes[i, 1].set_title('After Defense')
                axes[i, 1].axis('off')
                
                # Plot difference
                axes[i, 2].imshow(diff, cmap='hot')
                axes[i, 2].set_title('Difference Map')
                axes[i, 2].axis('off')
            
            plt.suptitle(f'Defense Effects for {attack_type} Attack', fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Defense effects visualization saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing defense effects: {e}")
            import traceback
            traceback.print_exc()


# ==================== ADVERSARIAL FIREWALL ====================
class AdversarialFirewall:
    """Multi-Class Adversarial Firewall with Adaptive Defense"""
    
    def __init__(self, dataset_name: str = 'CIFAR10', main_model: Optional[nn.Module] = None, 
                 num_classes: Optional[int] = None):
        self.config = AdversarialFirewallConfig(dataset_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine number of classes - use provided value or default
        if num_classes is not None:
            self.num_training_classes = num_classes
        else:
            # For training, use the actual attack classes we'll train on
            self.num_training_classes = 5  # Clean + 4 attacks
        
        # Validate num_classes
        if self.num_training_classes > len(self.config.attack_classes):
            logger.warning(f"Requested {self.num_training_classes} classes but only {len(self.config.attack_classes)} are available")
            logger.warning(f"Using maximum available: {len(self.config.attack_classes)} classes")
            self.num_training_classes = len(self.config.attack_classes)
        
        # Update the config's attack_classes based on num_training_classes
        self.config.attack_classes = self.config.get_attack_classes_for_num_classes(self.num_training_classes)
        
        logger.info(f"\n{'='*80}")
        logger.info("ENHANCED MULTI-CLASS ADVERSARIAL FIREWALL")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training classes: {self.num_training_classes}")
        logger.info(f"Attack classes: {list(self.config.attack_classes.values())}")
        logger.info(f"Defense strategies: {self.config.defense_strategies}")
        
        # Enhanced firewall network - now uses the correct number of classes
        self.firewall = EnhancedFirewallNetwork(
            self.config, 
            num_classes=self.num_training_classes  # Pass the actual number
        ).to(self.device)
        
        # Main classifier (protected model)
        self.main_model = main_model
        if self.main_model is None:
            # Create a dummy model for attack generation if none provided
            self.main_model = self._create_dummy_model()
            logger.info("Created dummy main model for attack generation")
        else:
            self.main_model = self.main_model.to(self.device)
            self.main_model.eval()
        
        # Defense module
        self.defense_module = DefenseModule(self.config)
        
        # Attack generator for training
        self.attack_generator = self._create_attack_generator()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'clean_forwarded': 0,
            'defended_forwarded': 0,
            'blocked': 0,
            'attack_distribution': defaultdict(int),
            'inference_times': [],
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'decision_distribution': defaultdict(int)
        }
        
        # Cache for performance
        self.cache = OrderedDict()
        self.cache_size = 1000
        
        # Optimizer - AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.firewall.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler - Cosine annealing with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epochs': [],
            'learning_rates': [],
            'train_f1': [], 'val_f1': []
        }
        
        # Create output directory
        self.output_dir = Path('firewall_output')
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"\nEnhanced Firewall initialized successfully!")
    
    def _create_attack_generator(self):
        """Create attack generator for training"""
        class AttackGenerator:
            def __init__(self, device):
                self.device = device
            
            def fgsm(self, images, model, labels, eps=0.3):
                images_adv = images.clone().detach().requires_grad_(True)
                model.eval()
                
                with torch.enable_grad():
                    outputs = model(images_adv)
                    loss = F.cross_entropy(outputs, labels)
                    grad = torch.autograd.grad(loss, images_adv)[0]
                    images_adv = images_adv + eps * grad.sign()
                    images_adv = torch.clamp(images_adv, 0, 1).detach()
                
                return images_adv
            
            def pgd(self, images, model, labels, eps=0.3, alpha=0.01, steps=10):
                images_adv = images.clone().detach()
                model.eval()
                
                # Random start for better attack
                images_adv = images_adv + torch.randn_like(images_adv) * 0.1
                images_adv = torch.clamp(images_adv, 0, 1)
                
                for _ in range(steps):
                    images_adv.requires_grad = True
                    outputs = model(images_adv)
                    loss = F.cross_entropy(outputs, labels)
                    grad = torch.autograd.grad(loss, images_adv)[0]
                    
                    images_adv = images_adv.detach() + alpha * grad.sign()
                    delta = torch.clamp(images_adv - images, -eps, eps)
                    images_adv = torch.clamp(images + delta, 0, 1).detach()
                
                return images_adv
            
            def noise(self, images, level=0.1):
                # Different types of noise
                noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle', 'uniform'])
                
                if noise_type == 'gaussian':
                    noise = torch.randn_like(images) * level
                elif noise_type == 'salt_pepper':
                    noise = torch.zeros_like(images)
                    salt_mask = torch.rand_like(images) < level/2
                    pepper_mask = torch.rand_like(images) < level/2
                    noise[salt_mask] = 1.0
                    noise[pepper_mask] = -1.0
                elif noise_type == 'speckle':
                    noise = images * torch.randn_like(images) * level
                else:  # uniform
                    noise = (torch.rand_like(images) - 0.5) * 2 * level
                
                return torch.clamp(images + noise, 0, 1)
            
            def mia(self, images, model, labels, eps=0.3, alpha=0.01, steps=10, momentum=0.9):
                images_adv = images.clone().detach()
                accumulated_grad = torch.zeros_like(images)
                model.eval()
                
                for _ in range(steps):
                    images_adv.requires_grad = True
                    outputs = model(images_adv)
                    loss = F.cross_entropy(outputs, labels)
                    grad = torch.autograd.grad(loss, images_adv)[0]
                    accumulated_grad = momentum * accumulated_grad + grad / (grad.norm(p=1) + 1e-8)
                    images_adv = images_adv.detach() + alpha * accumulated_grad.sign()
                    delta = torch.clamp(images_adv - images, -eps, eps)
                    images_adv = torch.clamp(images + delta, 0, 1).detach()
                
                return images_adv
            
            def cw(self, images, model, labels, c=1.0, kappa=0.0, steps=100, lr=0.01):
                """Carlini-Wagner L2 attack"""
                images_adv = images.clone().detach()
                model.eval()
                
                # Initialize perturbation
                w = torch.zeros_like(images, requires_grad=True)
                optimizer = torch.optim.Adam([w], lr=lr)
                
                for step in range(steps):
                    optimizer.zero_grad()
                    
                    # Compute adversarial image
                    adv_images = torch.tanh(w) * 0.5 + 0.5  # Map to [0, 1]
                    
                    # Compute loss
                    outputs = model(adv_images)
                    f_loss = F.cross_entropy(outputs, labels)
                    
                    # Distance loss
                    dist_loss = torch.norm(adv_images - images, p=2)
                    
                    # Total loss
                    loss = dist_loss + c * f_loss
                    
                    loss.backward()
                    optimizer.step()
                
                adv_images = torch.tanh(w) * 0.5 + 0.5
                return adv_images.detach()
            
            def deepfool(self, images, model, labels, max_iter=50, overshoot=0.02):
                """DeepFool attack"""
                images_adv = images.clone().detach()
                model.eval()
                
                batch_size = images.shape[0]
                for i in range(batch_size):
                    image = images_adv[i:i+1]
                    label = labels[i:i+1]
                    
                    pert = torch.zeros_like(image, requires_grad=True)
                    image_pert = image + pert
                    
                    for _ in range(max_iter):
                        image_pert.requires_grad = True
                        outputs = model(image_pert)
                        orig_output = outputs[0, label].item()
                        
                        # Find closest decision boundary
                        grads = []
                        for c in range(outputs.shape[1]):
                            if c != label:
                                grad = torch.autograd.grad(outputs[0, c], image_pert)[0]
                                grads.append(grad)
                        
                        if not grads:
                            break
                        
                        # Calculate perturbation
                        grads = torch.stack(grads)
                        w = grads - torch.autograd.grad(outputs[0, label], image_pert)[0]
                        f = outputs[0, :] - outputs[0, label]
                        f = f.view(-1, 1)
                        
                        # Avoid division by zero
                        denom = torch.norm(w.view(w.shape[0], -1), dim=1) ** 2
                        denom = denom.view(-1, 1) + 1e-8
                        pert_i = torch.abs(f) * w / denom
                        pert_i = pert_i.sum(dim=0)
                        
                        # Update perturbation
                        pert = pert + pert_i * (1 + overshoot)
                        image_pert = torch.clamp(image + pert, 0, 1).detach()
                
                return torch.clamp(images + pert, 0, 1)
            
            def spsa(self, images, model, labels, epsilon=0.1, delta=0.01, lr=0.01, iterations=10):
                """SPSA (Simultaneous Perturbation Stochastic Approximation) attack"""
                images_adv = images.clone().detach()
                model.eval()
                
                for _ in range(iterations):
                    # Generate random perturbation
                    pert = torch.randn_like(images_adv) * delta
                    pert_plus = images_adv + pert
                    pert_minus = images_adv - pert
                    
                    # Clip to valid range
                    pert_plus = torch.clamp(pert_plus, 0, 1)
                    pert_minus = torch.clamp(pert_minus, 0, 1)
                    
                    # Compute loss gradient approximation
                    with torch.no_grad():
                        loss_plus = F.cross_entropy(model(pert_plus), labels)
                        loss_minus = F.cross_entropy(model(pert_minus), labels)
                    
                    # Approximate gradient
                    grad_approx = (loss_plus - loss_minus) / (2 * delta)
                    
                    # Update adversarial images
                    images_adv = images_adv - lr * grad_approx * pert.sign()
                    images_adv = torch.clamp(images_adv, 0, 1).detach()
                
                return images_adv
            
            def boundary(self, images, model, labels, steps=1000, spherical_step=0.01, source_step=0.01):
                """Boundary attack"""
                images_adv = images.clone().detach()
                model.eval()
                
                batch_size = images.shape[0]
                
                for b in range(batch_size):
                    image = images[b:b+1]
                    label = labels[b:b+1]
                    
                    # Start from random point
                    adv = torch.rand_like(image) * 0.5 + 0.25
                    
                    for step in range(steps):
                        # Random perturbation on sphere
                        perturbation = torch.randn_like(adv)
                        perturbation = perturbation / (perturbation.norm() + 1e-8) * spherical_step
                        
                        # Project back to sphere
                        candidate = adv + perturbation
                        candidate = candidate / (candidate.norm() + 1e-8) * image.norm()
                        
                        # Step towards source
                        candidate = candidate + (image - candidate) * source_step
                        
                        # Check if adversarial
                        with torch.no_grad():
                            pred = model(candidate).argmax()
                        
                        if pred != label:
                            adv = candidate
                
                return adv
            
            def jsma(self, images, model, labels, theta=1.0, gamma=0.1, max_iter=20):
                """JSMA (Jacobian-based Saliency Map Attack)"""
                images_adv = images.clone().detach()
                model.eval()
                
                batch_size = images.shape[0]
                
                for b in range(batch_size):
                    image = images_adv[b:b+1]
                    label = labels[b:b+1]
                    
                    for _ in range(max_iter):
                        image.requires_grad = True
                        outputs = model(image)
                        
                        # Create target class (not the original)
                        num_classes = outputs.shape[1]
                        target = (label + 1) % num_classes
                        
                        # Compute saliency map
                        saliency = torch.zeros_like(image)
                        for c in range(num_classes):
                            if c != label:
                                grad = torch.autograd.grad(outputs[0, c], image)[0]
                                saliency += grad.abs()
                        
                        # Find pixel to modify
                        saliency_flat = saliency.view(-1)
                        idx = saliency_flat.argmax()
                        
                        # Modify pixel
                        flat_image = image.view(-1)
                        flat_image[idx] += theta * gamma
                        
                        # Clip to valid range
                        image = torch.clamp(image, 0, 1).detach()
                        
                        # Check if successful
                        with torch.no_grad():
                            pred = model(image).argmax()
                        
                        if pred != label:
                            break
                
                return images_adv
        
        return AttackGenerator(self.device)
    
    def _load_dataset(self, train: bool = True, num_samples: Optional[int] = None):
        """Load dataset with data augmentation for training"""
        if self.config.dataset_name == 'MNIST':
            if train:
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomRotation(5),
                    transforms.RandomAffine(0, translate=(0.05, 0.05)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ])
            dataset = torchvision.datasets.MNIST(
                root='./data', train=train, download=True, transform=transform
            )
        elif self.config.dataset_name == 'CIFAR100':
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            dataset = torchvision.datasets.CIFAR100(
                root='./data', train=train, download=True, transform=transform
            )
        else:  # CIFAR10
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            dataset = torchvision.datasets.CIFAR10(
                root='./data', train=train, download=True, transform=transform
            )
        
        if num_samples:
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        return dataset
    
    def create_training_dataset(self, samples_per_class: int = 400, include_all_attacks: bool = False):
        """Create enhanced dataset for firewall training"""
        logger.info(f"\n[1/4] Creating enhanced training dataset ({samples_per_class} per class)...")
        
        all_images = []
        all_labels = []
        
        # Clean samples with augmentation
        logger.info("\nProcessing clean samples...")
        clean_dataset = self._load_dataset(train=True, num_samples=samples_per_class * 2)
        clean_loader = torch.utils.data.DataLoader(
            clean_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        clean_count = 0
        for images, _ in tqdm(clean_loader, desc="Clean"):
            if clean_count + len(images) > samples_per_class:
                images = images[:samples_per_class - clean_count]
            
            all_images.append(images)
            all_labels.extend([0] * len(images))
            clean_count += len(images)
            
            if clean_count >= samples_per_class:
                break
        
        # Determine which attacks to include based on num_training_classes
        attack_methods = []
        
        # Define all available attacks with their IDs and methods
        available_attacks = [
            ('FGSM', 1, self.attack_generator.fgsm),
            ('PGD', 2, self.attack_generator.pgd),
            ('Noise', 3, self.attack_generator.noise),
            ('MIA', 4, self.attack_generator.mia),
            ('CW', 5, self.attack_generator.cw),
            ('DeepFool', 6, self.attack_generator.deepfool),
            ('SPSA', 7, self.attack_generator.spsa),
            ('Boundary', 8, self.attack_generator.boundary),
            ('JSMA', 9, self.attack_generator.jsma),
        ]
        
        # Add attacks up to num_training_classes - 1 (since class 0 is Clean)
        for i in range(1, self.num_training_classes):
            if i - 1 < len(available_attacks):
                attack_name, class_id, attack_method = available_attacks[i - 1]
                attack_methods.append((attack_name, class_id, attack_method))
        
        logger.info(f"Will generate data for: Clean + {[name for name, _, _ in attack_methods]}")
        
        # Adversarial samples
        self.main_model.eval()
        
        for attack_name, class_id, attack_method in attack_methods:
            logger.info(f"\nProcessing {attack_name} samples...")
            
            samples_collected = 0
            while samples_collected < samples_per_class:
                # Load batch with augmentation
                batch_needed = min(self.config.batch_size * 2, samples_per_class - samples_collected)
                adv_dataset = self._load_dataset(train=True, num_samples=batch_needed)
                adv_loader = torch.utils.data.DataLoader(
                    adv_dataset, batch_size=min(batch_needed, self.config.batch_size), shuffle=False
                )
                
                images, labels = next(iter(adv_loader))
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Generate adversarial examples with varying strengths
                with torch.enable_grad():
                    if attack_name == 'FGSM':
                        # Vary epsilon for different attack strengths
                        eps = random.choice([0.1, 0.2, 0.3, 0.4])
                        adv_images = attack_method(
                            images, self.main_model, labels, eps=eps
                        )
                    elif attack_name == 'PGD':
                        # Vary steps for different attack strengths
                        steps = random.choice([5, 10, 15])
                        adv_images = attack_method(
                            images, self.main_model, labels, eps=0.3, alpha=0.01, steps=steps
                        )
                    elif attack_name == 'Noise':
                        # Vary noise level
                        level = random.choice([0.05, 0.1, 0.15, 0.2])
                        adv_images = attack_method(images, level=level)
                    elif attack_name == 'MIA':
                        adv_images = attack_method(
                            images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10
                        )
                    elif attack_name == 'CW':
                        adv_images = attack_method(
                            images, self.main_model, labels, c=1.0, steps=50, lr=0.01
                        )
                    elif attack_name == 'DeepFool':
                        adv_images = attack_method(
                            images, self.main_model, labels, max_iter=50, overshoot=0.02
                        )
                    elif attack_name == 'SPSA':
                        adv_images = attack_method(
                            images, self.main_model, labels, epsilon=0.1, delta=0.01, lr=0.01, iterations=10
                        )
                    elif attack_name == 'Boundary':
                        adv_images = attack_method(
                            images, self.main_model, labels, steps=100, spherical_step=0.01, source_step=0.01
                        )
                    elif attack_name == 'JSMA':
                        adv_images = attack_method(
                            images, self.main_model, labels, theta=1.0, gamma=0.1, max_iter=20
                        )
                    else:
                        # Default: use FGSM as fallback
                        adv_images = self.attack_generator.fgsm(
                            images, self.main_model, labels, eps=0.3
                        )
                
                all_images.append(adv_images.cpu())
                all_labels.extend([class_id] * len(images))
                samples_collected += len(images)
                
                if samples_collected % 100 == 0:
                    logger.info(f"  Collected {samples_collected}/{samples_per_class} samples")
        
        # Combine
        images_all = torch.cat(all_images, dim=0)
        labels_all = torch.tensor(all_labels, dtype=torch.long)
        
        # Shuffle the dataset
        indices = torch.randperm(len(images_all))
        images_all = images_all[indices]
        labels_all = labels_all[indices]
        
        logger.info(f"\nEnhanced training dataset created:")
        logger.info(f"  Total samples: {len(images_all)}")
        class_counts = np.bincount(labels_all.numpy())
        for i, count in enumerate(class_counts):
            attack_name = self.config.attack_classes.get(i, f'Class_{i}')
            logger.info(f"  {attack_name}: {count} samples")
        
        return images_all, labels_all
    
    def _create_dummy_model(self) -> nn.Module:
        """Create dummy model for attack generation"""
        if self.config.dataset_name == 'MNIST':
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(512, self.config.main_model_classes)
        elif self.config.dataset_name == 'CIFAR100':
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(512, 100)
        else:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(512, 10)
        
        return model.to(self.device)
    
    def train_firewall(self, num_epochs: int = 25):
        """Train the enhanced firewall"""
        logger.info(f"\n[2/4] Training enhanced adversarial firewall ({num_epochs} epochs)...")
        
        # Create enhanced training dataset
        images, labels = self.create_training_dataset(samples_per_class=400)
        
        # Split dataset with stratification
        X = images.numpy()
        y = labels.numpy()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2
        )
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Calculate class weights for imbalanced data
        # Use the actual number of unique classes in the training data
        num_train_classes = len(np.unique(y_train))
        
        logger.info(f"Training classes: {num_train_classes}")
        logger.info(f"Training class distribution: {np.bincount(y_train, minlength=num_train_classes)}")
        
        # Create weights for actual training classes
        class_counts = np.bincount(y_train, minlength=num_train_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum()
        
        # Create weights tensor with the correct number of classes
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        logger.info(f"Class weights shape: {class_weights_tensor.shape}")
        logger.info(f"Class weights: {class_weights_tensor.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # Training
            self.firewall.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_targets = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_images, batch_labels in pbar:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward
                outputs = self.firewall(batch_images)
                logits = outputs['logits']
                
                loss = criterion(logits, batch_labels)
                
                # Add L2 regularization
                l2_lambda = 1e-4
                l2_norm = sum(p.pow(2.0).sum() for p in self.firewall.parameters())
                loss = loss + l2_lambda * l2_norm
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.firewall.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Stats
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()
                
                train_preds.extend(predicted.cpu().numpy())
                train_targets.extend(batch_labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * predicted.eq(batch_labels).sum().item() / batch_labels.size(0)
                })
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Validation
            self.firewall.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.firewall(batch_images)
                    logits = outputs['logits']
                    
                    loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_labels.cpu().numpy())
            
            # Metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Calculate F1 scores
            try:
                train_f1 = precision_recall_fscore_support(train_targets, train_preds, average='weighted')[2]
                val_f1 = precision_recall_fscore_support(val_targets, val_preds, average='weighted')[2]
            except:
                train_f1 = 0
                val_f1 = 0
            
            # Calculate per-class accuracy
            try:
                cm = confusion_matrix(val_targets, val_preds)
                per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
            except:
                per_class_acc = np.zeros(num_train_classes)
            
            # Store history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss / max(1, len(train_loader)))
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss / max(1, len(val_loader)))
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            logger.info(f"\nEpoch {epoch+1}:")
            logger.info(f"  Train - Loss: {train_loss/max(1, len(train_loader)):.4f}, Acc: {train_acc:.1f}%, F1: {train_f1:.4f}")
            logger.info(f"  Val   - Loss: {val_loss/max(1, len(val_loader)):.4f}, Acc: {val_acc:.1f}%, F1: {val_f1:.4f}")
            logger.info(f"  LR: {current_lr:.6f}")
            
            logger.info(f"  Per-class accuracy:")
            for i, acc in enumerate(per_class_acc):
                attack_name = self.config.attack_classes.get(i, f'Class_{i}')
                logger.info(f"    {attack_name}: {acc*100:.1f}%")
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.firewall.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config.__dict__,
                    'history': self.history,
                    'num_training_classes': self.num_training_classes
                }, checkpoint_path)
                logger.info(f"  [Checkpoint saved: {checkpoint_path}]")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_path = self.output_dir / 'best_firewall_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.firewall.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config.__dict__,
                    'history': self.history,
                    'num_training_classes': self.num_training_classes
                }, best_model_path)
                logger.info(f"  [Best model saved: Val Acc = {val_acc:.1f}%]")
            
            # Early stopping check
            if epoch > self.config.patience:
                recent_acc = self.history['val_acc'][-self.config.patience:]
                if max(recent_acc) < best_val_acc - 2.0:  # 2% drop
                    logger.info(f"  Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        best_model_path = self.output_dir / 'best_firewall_model.pth'
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.firewall.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint['history']
            logger.info(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
        
        logger.info(f"\nEnhanced Firewall training completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")
        
        # Plot training history
        self._plot_training_history()
    
    def _plot_training_history(self):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            epochs = self.history['epochs']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss plot
            ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
            ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # F1 Score plot
            ax3.plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
            ax3.plot(epochs, self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('Training and Validation F1 Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Learning rate plot
            ax4.plot(epochs, self.history['learning_rates'], 'g-', label='Learning Rate', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / 'training_history.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved as '{plot_path}'")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping training history plot")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
    
    def _get_image_hash(self, image: torch.Tensor) -> str:
        """Generate hash for image caching with better performance"""
        with torch.no_grad():
            # Use reduced precision for faster hashing
            img_np = image.cpu().numpy()
            
            # Normalize and resize to fixed size for consistent hashing
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            
            # Use max pooling to reduce size
            if len(img_np.shape) == 3:
                # For color images, convert to grayscale
                if img_np.shape[0] == 3:
                    img_np = np.mean(img_np, axis=0)
                elif img_np.shape[0] == 1:
                    img_np = img_np[0]
            
            # Resize to 8x8 for faster hashing
            if img_np.shape[0] > 8:
                img_np = cv2.resize(img_np, (8, 8), interpolation=cv2.INTER_AREA)
            
            # Quantize to 16 levels
            img_np = (img_np * 16).astype(np.uint8)
            
            # Create hash
            return hashlib.md5(img_np.tobytes()).hexdigest()[:12]
    
    def _update_cache(self, img_hash: str, result: Dict[str, Any]):
        """Update cache with LRU policy"""
        if img_hash in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(img_hash)
        else:
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[img_hash] = result
    
    def process_input(self, image: torch.Tensor, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """
        Process input through the adversarial firewall
        
        Returns:
            decision: 'forward', 'defend', or 'block'
            attack_type: predicted attack class
            confidence: prediction confidence
            processed_image: image after defense (if applied)
            processing_time: ms taken for decision
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # Check cache
        img_hash = self._get_image_hash(image)
        if img_hash in self.cache:
            return self.cache[img_hash]
        
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Firewall inference
        self.firewall.eval()
        with torch.no_grad():
            outputs = self.firewall(image)
        
        # Get prediction
        pred_class = outputs['predictions'][0].item()
        confidence = outputs['confidence'][0].item()
        
        # Map prediction to attack type
        if pred_class < len(self.config.attack_classes):
            attack_name = self.config.attack_classes.get(pred_class, 'Unknown')
        else:
            # Handle predictions beyond trained classes
            if pred_class == 0:
                attack_name = 'Clean'
            elif pred_class < self.num_training_classes:
                attack_name = f'Attack_{pred_class}'
            else:
                attack_name = 'Unknown'
        
        # Decision logic with adaptive threshold
        decision_threshold = self.config.block_threshold
        defense_threshold = self.config.defense_threshold
        
        if attack_name == 'Clean' and confidence > self.config.confidence_threshold:
            decision = 'forward'
            processed_image = image
            self.stats['clean_forwarded'] += 1
            self.stats['decision_distribution']['forward'] += 1
        elif confidence > decision_threshold:
            decision = 'block'
            processed_image = None
            self.stats['blocked'] += 1
            self.stats['decision_distribution']['block'] += 1
        elif confidence > defense_threshold:
            decision = 'defend'
            # Apply appropriate defense
            defense_func = self.defense_module.get_defense(attack_name)
            processed_image = defense_func(image)
            self.stats['defended_forwarded'] += 1
            self.stats['decision_distribution']['defend'] += 1
        else:
            # Low confidence, forward with warning
            decision = 'forward'
            processed_image = image
            self.stats['clean_forwarded'] += 1
            self.stats['decision_distribution']['forward_low_conf'] += 1
            attack_name = 'Clean (low confidence)'
        
        # Update statistics
        self.stats['attack_distribution'][attack_name] += 1
        
        # Record inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.stats['inference_times'].append(inference_time)
        
        # Track errors if ground truth provided
        if ground_truth is not None:
            if ground_truth == 0 and pred_class != 0:  # False positive
                self.stats['false_positives'] += 1
            elif ground_truth != 0 and pred_class == 0:  # False negative
                self.stats['false_negatives'] += 1
            elif ground_truth == 0 and pred_class == 0:  # True negative
                self.stats['true_negatives'] += 1
            elif ground_truth != 0 and pred_class != 0:  # True positive
                self.stats['true_positives'] += 1
        
        # Prepare result
        result = {
            'decision': decision,
            'attack_type': attack_name,
            'confidence': confidence,
            'processed_image': processed_image,
            'processing_time_ms': inference_time,
            'firewall_prediction': pred_class,
            'timestamp': time.time()
        }
        
        # Cache result
        self._update_cache(img_hash, result)
        
        return result
    
    def process_batch(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """Process a batch of images efficiently"""
        batch_size = images.shape[0]
        results = []
        
        # Process in chunks to avoid memory issues
        chunk_size = min(16, batch_size)
        for i in range(0, batch_size, chunk_size):
            chunk_end = min(i + chunk_size, batch_size)
            chunk = images[i:chunk_end]
            
            # Extract features for entire chunk
            with torch.no_grad():
                features = self.firewall.extract_features(chunk)
                logits = self.firewall.classifier(features)
                probs = F.softmax(logits, dim=1)
                confidence, preds = torch.max(probs, dim=1)
            
            # Process each image in chunk
            for j in range(chunk.shape[0]):
                idx = i + j
                pred_class = preds[j].item()
                conf = confidence[j].item()
                
                # Get attack name
                if pred_class < len(self.config.attack_classes):
                    attack_name = self.config.attack_classes.get(pred_class, 'Unknown')
                else:
                    # Handle predictions beyond trained classes
                    if pred_class == 0:
                        attack_name = 'Clean'
                    elif pred_class < self.num_training_classes:
                        attack_name = f'Attack_{pred_class}'
                    else:
                        attack_name = 'Unknown'
                
                # Make decision
                if attack_name == 'Clean' and conf > self.config.confidence_threshold:
                    decision = 'forward'
                    processed_image = chunk[j:j+1]
                elif conf > self.config.block_threshold:
                    decision = 'block'
                    processed_image = None
                elif conf > self.config.defense_threshold:
                    decision = 'defend'
                    defense_func = self.defense_module.get_defense(attack_name)
                    processed_image = defense_func(chunk[j:j+1])
                else:
                    decision = 'forward'
                    processed_image = chunk[j:j+1]
                    attack_name = 'Clean (low confidence)'
                
                results.append({
                    'decision': decision,
                    'attack_type': attack_name,
                    'confidence': conf,
                    'processed_image': processed_image,
                    'prediction': pred_class,
                    'index': idx
                })
        
        return results
    
    def visualize_attacks_comparison(self, save_path: Optional[str] = None):
        """Visualize comparison between clean and adversarial images for different attack types"""
        try:
            import matplotlib.pyplot as plt
            
            # Load test images
            test_dataset = self._load_dataset(train=False, num_samples=5)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            images, labels = next(iter(test_loader))
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Select attack types to visualize (up to 5)
            attack_types = list(self.config.attack_classes.values())[:5]  # First 5 attack types
            
            n_attacks = len(attack_types)
            fig, axes = plt.subplots(n_attacks, 2, figsize=(10, 3 * n_attacks))
            
            if n_attacks == 1:
                axes = axes.reshape(1, -1)
            
            self.main_model.eval()
            
            for i, attack_name in enumerate(attack_types):
                # Clean image (first column)
                clean_np = images[0].cpu().numpy()
                if clean_np.shape[0] in [1, 3]:
                    clean_np = np.transpose(clean_np, (1, 2, 0))
                
                axes[i, 0].imshow(clean_np if clean_np.shape[2] == 3 else clean_np.squeeze(), 
                                cmap='gray' if len(clean_np.shape) == 2 else None)
                axes[i, 0].set_title('Clean Image')
                axes[i, 0].axis('off')
                
                # Adversarial image (second column)
                if attack_name == 'Clean':
                    adv_np = clean_np
                    attack_title = 'Clean (No Attack)'
                else:
                    with torch.enable_grad():
                        if attack_name == 'FGSM':
                            adv_images = self.attack_generator.fgsm(images, self.main_model, labels, eps=0.3)
                        elif attack_name == 'PGD':
                            adv_images = self.attack_generator.pgd(images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10)
                        elif attack_name == 'Noise':
                            adv_images = self.attack_generator.noise(images, level=0.1)
                        elif attack_name == 'MIA':
                            adv_images = self.attack_generator.mia(images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10)
                        elif attack_name == 'CW':
                            adv_images = self.attack_generator.cw(images, self.main_model, labels, c=1.0, steps=50, lr=0.01)
                        else:
                            adv_images = self.attack_generator.fgsm(images, self.main_model, labels, eps=0.3)
                    
                    adv_np = adv_images[0].cpu().numpy()
                    if adv_np.shape[0] in [1, 3]:
                        adv_np = np.transpose(adv_np, (1, 2, 0))
                    attack_title = attack_name
                
                axes[i, 1].imshow(adv_np if adv_np.shape[2] == 3 else adv_np.squeeze(), 
                                cmap='gray' if len(adv_np.shape) == 2 else None)
                axes[i, 1].set_title(f'{attack_title} Attack')
                axes[i, 1].axis('off')
            
            plt.suptitle('Clean vs. Adversarial Images Comparison', fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Attack comparison visualization saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing attack comparison: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_feature_extraction(self, save_dir: Optional[str] = None):
        """Visualize spatial and frequency feature extraction"""
        try:
            import matplotlib.pyplot as plt
            
            # Load a sample image
            test_dataset = self._load_dataset(train=False, num_samples=1)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            image, _ = next(iter(test_loader))
            image = image.to(self.device)
            
            # Convert to numpy for visualization
            img_np = image[0].cpu().numpy()
            
            # Create output directory if needed
            if save_dir:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(exist_ok=True, parents=True)
            else:
                save_dir_path = self.output_dir / 'feature_visualizations'
                save_dir_path.mkdir(exist_ok=True, parents=True)
            
            # Visualize spatial features
            spatial_save_path = save_dir_path / 'spatial_features.png'
            self.firewall.spatial_extractor.visualize_features(img_np, str(spatial_save_path))
            
            # Visualize frequency features
            freq_save_path = save_dir_path / 'frequency_features.png'
            self.firewall.freq_extractor.visualize_features(img_np, str(freq_save_path))
            
            logger.info(f"Feature visualizations saved to: {save_dir_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing feature extraction: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_defense_effects_all(self, save_dir: Optional[str] = None):
        """Visualize defense effects for all attack types"""
        try:
            import matplotlib.pyplot as plt
            
            # Load test images
            test_dataset = self._load_dataset(train=False, num_samples=3)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)
            
            images, labels = next(iter(test_loader))
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Create output directory if needed
            if save_dir:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(exist_ok=True, parents=True)
            else:
                save_dir_path = self.output_dir / 'defense_visualizations'
                save_dir_path.mkdir(exist_ok=True, parents=True)
            
            # Test each attack type
            attack_types = list(self.config.attack_classes.values())[1:5]  # Skip 'Clean'
            
            for attack_name in attack_types:
                # Generate adversarial examples
                self.main_model.eval()
                with torch.enable_grad():
                    if attack_name == 'FGSM':
                        adv_images = self.attack_generator.fgsm(images, self.main_model, labels, eps=0.3)
                    elif attack_name == 'PGD':
                        adv_images = self.attack_generator.pgd(images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10)
                    elif attack_name == 'Noise':
                        adv_images = self.attack_generator.noise(images, level=0.1)
                    elif attack_name == 'MIA':
                        adv_images = self.attack_generator.mia(images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10)
                    elif attack_name == 'CW':
                        adv_images = self.attack_generator.cw(images, self.main_model, labels, c=1.0, steps=50, lr=0.01)
                    else:
                        continue
                
                # Visualize defense effects
                defense_save_path = save_dir_path / f'defense_{attack_name.lower()}.png'
                self.defense_module.visualize_defense_effects(adv_images, attack_name, str(defense_save_path))
            
            logger.info(f"Defense visualizations saved to: {save_dir_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing defense effects: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_firewall(self):
        """Comprehensive evaluation of the firewall"""
        logger.info(f"\n[3/4] Evaluating enhanced adversarial firewall...")
        
        # Load test data
        test_dataset = self._load_dataset(train=False, num_samples=200)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )
        
        # Ensure main model is available for evaluation
        if self.main_model is None:
            logger.warning("No main model available, creating dummy model for evaluation")
            self.main_model = self._create_dummy_model()
        
        self.main_model.eval()
        
        # Test each attack type
        attack_types = list(self.config.attack_classes.values())
        results = {}
        
        for attack_idx, attack_name in enumerate(attack_types):
            logger.info(f"\n{'='*40}")
            logger.info(f"Evaluating {attack_name}")
            logger.info(f"{'='*40}")
            
            all_preds = []
            all_labels = []
            all_confidences = []
            decisions = []
            
            for images, labels in tqdm(test_loader, desc=f"Testing {attack_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Generate adversarial example if needed
                if attack_name != 'Clean':
                    with torch.enable_grad():
                        if attack_name == 'FGSM':
                            images = self.attack_generator.fgsm(
                                images, self.main_model, labels, eps=0.3
                            )
                        elif attack_name == 'PGD':
                            images = self.attack_generator.pgd(
                                images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10
                            )
                        elif attack_name == 'Noise':
                            images = self.attack_generator.noise(images, level=0.1)
                        elif attack_name == 'MIA':
                            images = self.attack_generator.mia(
                                images, self.main_model, labels, eps=0.3, alpha=0.01, steps=10
                            )
                        elif attack_name == 'CW':
                            images = self.attack_generator.cw(
                                images, self.main_model, labels, c=1.0, steps=50, lr=0.01
                            )
                        elif attack_name == 'DeepFool':
                            images = self.attack_generator.deepfool(
                                images, self.main_model, labels, max_iter=50, overshoot=0.02
                            )
                        elif attack_name == 'SPSA':
                            images = self.attack_generator.spsa(
                                images, self.main_model, labels, epsilon=0.1, delta=0.01, lr=0.01, iterations=10
                            )
                        elif attack_name == 'Boundary':
                            images = self.attack_generator.boundary(
                                images, self.main_model, labels, steps=100, spherical_step=0.01, source_step=0.01
                            )
                        elif attack_name == 'JSMA':
                            images = self.attack_generator.jsma(
                                images, self.main_model, labels, theta=1.0, gamma=0.1, max_iter=20
                            )
                        else:
                            # Default to FGSM for unknown attacks
                            images = self.attack_generator.fgsm(
                                images, self.main_model, labels, eps=0.3
                            )
                
                # Process through firewall
                result = self.process_input(
                    images[0], 
                    ground_truth=attack_idx if attack_name != 'Clean' else 0
                )
                
                all_preds.append(result['firewall_prediction'])
                all_labels.append(attack_idx if attack_name != 'Clean' else 0)
                all_confidences.append(result['confidence'])
                decisions.append(result['decision'])
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            
            # Attack-specific metrics
            if attack_name == 'Clean':
                # False positive rate
                fp_rate = np.mean([1 if p != 0 else 0 for p, l in zip(all_preds, all_labels)])
                fn_rate = 0  # No false negatives for clean
                
                results[attack_name] = {
                    'accuracy': float(accuracy),
                    'false_positive_rate': float(fp_rate),
                    'false_negative_rate': float(fn_rate),
                    'avg_confidence': float(np.mean(all_confidences)),
                    'std_confidence': float(np.std(all_confidences)),
                    'decisions': self._convert_decisions_to_serializable(decisions)
                }
                logger.info(f"  Accuracy: {accuracy*100:.1f}%")
                logger.info(f"  False Positive Rate: {fp_rate*100:.1f}%")
            else:
                # Detection rate for attacks
                detection_rate = np.mean([1 if p != 0 else 0 for p in all_preds])
                fp_rate = np.mean([1 if p != attack_idx and p != 0 else 0 for p in all_preds])
                
                results[attack_name] = {
                    'accuracy': float(accuracy),
                    'detection_rate': float(detection_rate),
                    'false_positive_rate': float(fp_rate),
                    'avg_confidence': float(np.mean(all_confidences)),
                    'std_confidence': float(np.std(all_confidences)),
                    'decisions': self._convert_decisions_to_serializable(decisions)
                }
                logger.info(f"  Accuracy: {accuracy*100:.1f}%")
                logger.info(f"  Detection Rate: {detection_rate*100:.1f}%")
                logger.info(f"  False Positive Rate (other attacks): {fp_rate*100:.1f}%")
            
            logger.info(f"  Avg Confidence: {np.mean(all_confidences):.3f} +|- {np.std(all_confidences):.3f}")
            
            # Convert decisions dictionary for logging
            unique_decisions, counts = np.unique(decisions, return_counts=True)
            decisions_dict = {str(decision): int(count) for decision, count in zip(unique_decisions, counts)}
            logger.info(f"  Decisions: {decisions_dict}")
        
        # Overall statistics
        logger.info(f"\n{'='*60}")
        logger.info("ENHANCED FIREWALL PERFORMANCE SUMMARY")
        logger.info(f"{'='*60}")
        
        # Performance metrics
        if self.stats['total_requests'] > 0:
            total_tp = self.stats.get('true_positives', 0)
            total_tn = self.stats.get('true_negatives', 0)
            total_fp = self.stats.get('false_positives', 0)
            total_fn = self.stats.get('false_negatives', 0)
            
            total = total_tp + total_tn + total_fp + total_fn
            if total > 0:
                accuracy = (total_tp + total_tn) / total
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                logger.info(f"\nOverall Metrics:")
                logger.info(f"  Accuracy: {accuracy*100:.2f}%")
                logger.info(f"  Precision: {precision*100:.2f}%")
                logger.info(f"  Recall (Detection Rate): {recall*100:.2f}%")
                logger.info(f"  F1-Score: {f1_score*100:.2f}%")
        
        if self.stats['inference_times']:
            avg_inference_time = np.mean(self.stats['inference_times'])
            std_inference_time = np.std(self.stats['inference_times'])
            logger.info(f"\nReal-time Performance:")
            logger.info(f"  Avg Inference Time: {avg_inference_time:.2f} +|- {std_inference_time:.2f} ms")
            logger.info(f"  Min Inference Time: {min(self.stats['inference_times']):.2f} ms")
            logger.info(f"  Max Inference Time: {max(self.stats['inference_times']):.2f} ms")
            logger.info(f"  Memory Usage: {sum(p.numel() for p in self.firewall.parameters()) * 4 / 1024 / 1024:.2f} MB")
        
        logger.info(f"\nTraffic Statistics:")
        logger.info(f"  Total Requests: {self.stats['total_requests']}")
        logger.info(f"  Clean Forwarded: {self.stats['clean_forwarded']}")
        logger.info(f"  Defended Forwarded: {self.stats['defended_forwarded']}")
        logger.info(f"  Blocked: {self.stats['blocked']}")
        
        logger.info(f"\nDecision Distribution:")
        for decision, count in self.stats['decision_distribution'].items():
            percentage = (count / self.stats['total_requests']) * 100 if self.stats['total_requests'] > 0 else 0
            logger.info(f"  {decision}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\nAttack Distribution:")
        for attack, count in self.stats['attack_distribution'].items():
            percentage = (count / self.stats['total_requests']) * 100 if self.stats['total_requests'] > 0 else 0
            logger.info(f"  {attack}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\nError Rates:")
        logger.info(f"  True Positives: {self.stats.get('true_positives', 0)}")
        logger.info(f"  True Negatives: {self.stats.get('true_negatives', 0)}")
        logger.info(f"  False Positives: {self.stats.get('false_positives', 0)}")
        logger.info(f"  False Negatives: {self.stats.get('false_negatives', 0)}")
        
        if self.stats['total_requests'] > 0:
            total_fp = self.stats.get('false_positives', 0)
            total_fn = self.stats.get('false_negatives', 0)
            fp_rate = total_fp / self.stats['total_requests']
            fn_rate = total_fn / self.stats['total_requests']
            logger.info(f"  False Positive Rate: {fp_rate*100:.2f}%")
            logger.info(f"  False Negative Rate: {fn_rate*100:.2f}%")
        
        # Save evaluation results
        eval_results_path = self.output_dir / 'evaluation_results.json'
        
        # Convert results to JSON serializable format
        serializable_results = {}
        for attack_name, metrics in results.items():
            serializable_results[attack_name] = {}
            for key, value in metrics.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_results[attack_name][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_results[attack_name][key] = int(value)
                elif isinstance(value, dict):
                    serializable_results[attack_name][key] = {
                        str(k): int(v) if isinstance(v, (np.int32, np.int64)) else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_results[attack_name][key] = value
        
        with open(eval_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"\nEvaluation results saved to: {eval_results_path}")
        
        return results
    
    def _convert_decisions_to_serializable(self, decisions):
        """Convert decisions list to JSON serializable dictionary"""
        unique_decisions, counts = np.unique(decisions, return_counts=True)
        decisions_dict = {}
        for decision, count in zip(unique_decisions, counts):
            decisions_dict[str(decision)] = int(count)
        return decisions_dict
    
    def deploy_as_layer(self, main_model: nn.Module) -> nn.Module:
        """Deploy firewall as a wrapper layer around main model"""
        class FirewallProtectedModel(nn.Module):
            def __init__(self, firewall_instance, main_model):
                super().__init__()
                self.firewall_instance = firewall_instance
                self.main_model = main_model
                # Call eval on the neural network, not the firewall instance
                self.firewall_instance.firewall.eval()
                self.main_model.eval()
            
            def forward(self, x):
                # Process through firewall
                result = self.firewall_instance.process_input(x)
                
                if result['decision'] == 'block':
                    # Return high-entropy output for blocked inputs
                    batch_size = x.shape[0]
                    num_classes = self.main_model.fc.out_features if hasattr(self.main_model, 'fc') else 10
                    return torch.ones(batch_size, num_classes).to(x.device) / num_classes
                
                # Get image to forward to main model
                if result['decision'] == 'defend':
                    image_to_forward = result['processed_image']
                else:  # 'forward'
                    image_to_forward = x
                
                # Ensure proper shape
                if len(image_to_forward.shape) == 3:
                    image_to_forward = image_to_forward.unsqueeze(0)
                
                # Forward to main model
                return self.main_model(image_to_forward)
            
            def get_firewall_info(self):
                """Get firewall information"""
                return {
                    'firewall_stats': self.firewall_instance.stats,
                    'config': self.firewall_instance.config.__dict__
                }
        
        return FirewallProtectedModel(self, main_model)
    
    def compress_firewall(self, compression_ratio: float = 0.5):
        """Compress firewall model for deployment"""
        logger.info(f"\nCompressing firewall model with ratio {compression_ratio}...")
        
        # Get original size
        original_params = sum(p.numel() for p in self.firewall.parameters())
        original_size = original_params * 4 / 1024 / 1024  # MB
        
        # Apply pruning
        for name, module in self.firewall.named_modules():
            if isinstance(module, nn.Linear):
                # Prune weights
                weights = module.weight.data.abs()
                threshold = torch.quantile(weights, compression_ratio)
                mask = weights > threshold
                module.weight.data = module.weight.data * mask.float()
        
        # Quantize to half precision
        self.firewall.half()
        
        # Calculate compressed size
        compressed_params = sum(p.numel() for p in self.firewall.parameters())
        compressed_size = compressed_params * 2 / 1024 / 1024  # Half precision = 2 bytes
        
        logger.info(f"  Original: {original_params:,} params, {original_size:.2f} MB")
        logger.info(f"  Compressed: {compressed_params:,} params, {compressed_size:.2f} MB")
        logger.info(f"  Compression: {original_size/compressed_size:.1f}x")
        
        return self.firewall
    
    def save_firewall(self, path: Optional[str] = None):
        """Save firewall state"""
        if path is None:
            path = self.output_dir / 'enhanced_adversarial_firewall.pth'
        
        torch.save({
            'config': self.config.__dict__,
            'firewall_state_dict': self.firewall.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stats': self.stats,
            'history': self.history,
            'num_training_classes': self.num_training_classes
        }, path)
        logger.info(f"\nEnhanced Firewall saved to: {path}")
    
    def load_firewall(self, path: str):
        """Load firewall state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update num_training_classes from checkpoint
        self.num_training_classes = checkpoint.get('num_training_classes', 5)
        
        # Re-initialize firewall with correct number of classes
        self.firewall = EnhancedFirewallNetwork(
            self.config, 
            num_classes=self.num_training_classes
        ).to(self.device)
        
        # Load state dict
        self.firewall.load_state_dict(checkpoint['firewall_state_dict'])
        
        # Load optimizer and scheduler if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load stats and history
        self.stats = checkpoint.get('stats', self.stats)
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"\nEnhanced Firewall loaded from: {path}")
        logger.info(f"  Training classes: {self.num_training_classes}")
    
    def get_feature_importance(self, n_samples: int = 100):
        """Get feature importance for interpretability"""
        logger.info(f"\nCalculating feature importance...")
        
        # Load some samples
        dataset = self._load_dataset(train=False, num_samples=n_samples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_features = []
        all_predictions = []
        
        self.firewall.eval()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.firewall(images)
                
                all_features.append(outputs['features'].cpu().numpy())
                all_predictions.append(outputs['predictions'].cpu().numpy())
        
        features = np.vstack(all_features)
        predictions = np.concatenate(all_predictions)
        
        # Calculate correlation with predictions
        correlations = []
        for i in range(features.shape[1]):
            try:
                corr = np.corrcoef(features[:, i], predictions)[0, 1]
                correlations.append(abs(corr))
            except:
                correlations.append(0)
        
        # Get feature names
        spatial_names = self.firewall.spatial_extractor.get_feature_names()
        freq_names = self.firewall.freq_extractor.get_feature_names()
        all_names = spatial_names + freq_names
        
        # Ensure we have enough names
        if len(all_names) < len(correlations):
            all_names.extend([f'feature_{i}' for i in range(len(all_names), len(correlations))])
        
        # Sort by importance
        importance = list(zip(all_names[:len(correlations)], correlations))
        importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"\nTop 10 most important features:")
        for name, score in importance[:10]:
            logger.info(f"  {name}: {score:.4f}")
        
        return importance

# ==================== MAIN EXECUTION ====================
def run_adversarial_firewall_demo(dataset_name='CIFAR10', num_training_classes=5):
    """Run complete adversarial firewall demonstration"""
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED ADVERSARIAL FIREWALL DEMONSTRATION")
    logger.info(f"Dataset: {dataset_name}, Training Classes: {num_training_classes}")
    logger.info(f"{'='*80}")
    
    try:
        # Step 1: Create or load main model
        logger.info(f"\n[1/4] Setting up main classifier...")
        if dataset_name == 'MNIST':
            main_model = torchvision.models.resnet18(pretrained=False)
            # FIX: Use 3 input channels since MNIST images are converted to 3 channels
            main_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            main_model.fc = nn.Linear(512, 10)
        elif dataset_name == 'CIFAR100':
            main_model = torchvision.models.resnet18(pretrained=False)
            main_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            main_model.fc = nn.Linear(512, 100)
        else:
            main_model = torchvision.models.resnet18(pretrained=False)
            main_model.fc = nn.Linear(512, 10)
        
        main_model = main_model.eval()
        logger.info("Main classifier ready")
        
        # Step 2: Initialize adversarial firewall
        logger.info(f"\n[2/4] Initializing enhanced adversarial firewall...")
        firewall = AdversarialFirewall(dataset_name, main_model, num_classes=num_training_classes)
        
        # Step 3: Train firewall
        firewall.train_firewall(num_epochs=25)
        
        # Step 4: Visualize examples
        logger.info(f"\n[4/4] Generating visual examples...")
        
        # Create visualization directory
        vis_dir = firewall.output_dir / 'visual_examples'
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Clean vs. adversarial images comparison
        logger.info("\n1. Generating clean vs. adversarial images comparison...")
        attack_comparison_path = vis_dir / 'attack_comparison.png'
        firewall.visualize_attacks_comparison(str(attack_comparison_path))
        
        # 2. Feature visualizations
        logger.info("\n2. Generating feature extraction visualizations...")
        firewall.visualize_feature_extraction(str(vis_dir))
        
        # 3. Defense effects visualizations
        logger.info("\n3. Generating defense effects visualizations...")
        firewall.visualize_defense_effects_all(str(vis_dir))
        
        # Step 5: Evaluate firewall
        logger.info(f"\n[5/5] Comprehensive evaluation...")
        results = firewall.evaluate_firewall()
        
        # Step 6: Save firewall
        firewall.save_firewall()
        
        # Step 7: Demonstrate deployment
        logger.info(f"\n{'='*80}")
        logger.info("DEPLOYMENT DEMONSTRATION")
        logger.info(f"{'='*80}")
        
        # Create protected model
        protected_model = firewall.deploy_as_layer(main_model)
        
        # Test with sample inputs
        test_dataset = firewall._load_dataset(train=False, num_samples=5)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        logger.info("\nSample Processing:")
        for i, (image, label) in enumerate(test_loader):
            if i >= 3:  # Show first 3 samples
                break
            
            # Original prediction
            with torch.no_grad():
                orig_pred = main_model(image.to(firewall.device))
                orig_class = orig_pred.argmax().item()
            
            # Firewall processing
            result = firewall.process_input(image[0])
            
            # Protected prediction - use the protected model
            with torch.no_grad():
                protected_pred = protected_model(image.to(firewall.device))
                protected_class = protected_pred.argmax().item()
            
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Original prediction: Class {orig_class}")
            logger.info(f"  Firewall decision: {result['decision']}")
            logger.info(f"  Attack type: {result['attack_type']}")
            logger.info(f"  Confidence: {result['confidence']:.3f}")
            logger.info(f"  Protected prediction: Class {protected_class}")
            logger.info(f"  Processing time: {result['processing_time_ms']:.2f} ms")
        
        # Feature importance analysis
        firewall.get_feature_importance(n_samples=50)
        
        logger.info(f"\n{'='*80}")
        logger.info("ENHANCED ADVERSARIAL FIREWALL DEMO COMPLETED!")
        logger.info(f"{'='*80}")
        
        # Print final statistics
        logger.info(f"\nFINAL STATISTICS:")
        logger.info(f"  Model Parameters: {sum(p.numel() for p in firewall.firewall.parameters()):,}")
        logger.info(f"  Model Size: {sum(p.numel() for p in firewall.firewall.parameters()) * 4 / 1024 / 1024:.2f} MB")
        
        if firewall.stats['inference_times']:
            logger.info(f"  Avg Inference Time: {np.mean(firewall.stats['inference_times']):.2f} ms")
        
        # Performance summary
        if firewall.stats['total_requests'] > 0:
            defense_rate = (firewall.stats['defended_forwarded'] + firewall.stats['blocked']) / firewall.stats['total_requests']
            logger.info(f"  Defense Activation Rate: {defense_rate*100:.1f}%")
        
        # Visual examples summary
        logger.info(f"\nVISUAL EXAMPLES GENERATED:")
        logger.info(f"  1. Attack Comparison: {attack_comparison_path}")
        logger.info(f"  2. Feature Visualizations: {vis_dir / 'spatial_features.png'}")
        logger.info(f"  3. Feature Visualizations: {vis_dir / 'frequency_features.png'}")
        logger.info(f"  4. Defense Effects: Check {vis_dir} for defense_*.png files")
        
        return firewall, protected_model
        
    except Exception as e:
        logger.error(f"\nError during firewall demo: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== COMMAND LINE INTERFACE ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Multi-Class Adversarial Firewall with Adaptive Defense'
    )
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'train', 'evaluate', 'deploy', 'importance', 'compress', 'visualize'],
                       help='Execution mode')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='Dataset to use')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load pre-trained firewall')
    parser.add_argument('--samples', type=int, default=400,
                       help='Number of samples per class for training')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--output', type=str, default='firewall_output',
                       help='Output directory for results')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of output classes (default: 5 for training)')
    parser.add_argument('--compress_ratio', type=float, default=0.5,
                       help='Compression ratio for model compression (0.0-1.0)')
    parser.add_argument('--include_all_attacks', action='store_true',
                       help='Include all attack types in training')
    parser.add_argument('--visualize_only', action='store_true',
                       help='Only generate visualizations without training')
    
    args = parser.parse_args()
    
    # Update output directory
    if args.output:
        import os
        os.environ['FIREWALL_OUTPUT_DIR'] = args.output
        # Create output directory
        Path(args.output).mkdir(exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED MULTI-CLASS ADVERSARIAL FIREWALL")
    logger.info(f"Mode: {args.mode}, Dataset: {args.dataset}, Classes: {args.num_classes}")
    logger.info(f"{'='*80}")
    
    try:
        if args.mode == 'demo':
            firewall, protected_model = run_adversarial_firewall_demo(
                args.dataset, 
                num_training_classes=args.num_classes
            )
        elif args.mode == 'train':
            logger.info("\nTraining mode...")
            firewall = AdversarialFirewall(args.dataset, num_classes=args.num_classes)
            firewall.create_training_dataset(
                samples_per_class=args.samples,
                include_all_attacks=args.include_all_attacks
            )
            firewall.train_firewall(num_epochs=args.epochs)
            firewall.save_firewall()
        elif args.mode == 'evaluate':
            logger.info("\nEvaluation mode...")
            firewall = AdversarialFirewall(args.dataset, num_classes=args.num_classes)
            if args.load:
                firewall.load_firewall(args.load)
            firewall.evaluate_firewall()
        elif args.mode == 'deploy':
            logger.info("\nDeployment mode...")
            # Load main model
            if args.dataset == 'MNIST':
                main_model = torchvision.models.resnet18(pretrained=False)
                main_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                main_model.fc = nn.Linear(512, 10)
            elif args.dataset == 'CIFAR100':
                main_model = torchvision.models.resnet18(pretrained=False)
                main_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                main_model.fc = nn.Linear(512, 100)
            else:
                main_model = torchvision.models.resnet18(pretrained=True)
                main_model.fc = nn.Linear(512, 10)
            
            # Load firewall
            firewall = AdversarialFirewall(args.dataset, main_model, num_classes=args.num_classes)
            if args.load:
                firewall.load_firewall(args.load)
            
            # Create protected model
            protected_model = firewall.deploy_as_layer(main_model)
            logger.info(f"Protected model created with firewall layer")
        elif args.mode == 'importance':
            logger.info("\nFeature importance analysis...")
            firewall = AdversarialFirewall(args.dataset, num_classes=args.num_classes)
            if args.load:
                firewall.load_firewall(args.load)
            firewall.get_feature_importance()
        elif args.mode == 'compress':
            logger.info("\nModel compression mode...")
            firewall = AdversarialFirewall(args.dataset, num_classes=args.num_classes)
            if args.load:
                firewall.load_firewall(args.load)
            firewall.compress_firewall(compression_ratio=args.compress_ratio)
            firewall.save_firewall('compressed_firewall.pth')
        elif args.mode == 'visualize':
            logger.info("\nVisualization mode...")
            firewall = AdversarialFirewall(args.dataset, num_classes=args.num_classes)
            if args.load:
                firewall.load_firewall(args.load)
            
            # Create visualization directory
            vis_dir = Path(args.output) / 'visual_examples'
            vis_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate all visualizations
            firewall.visualize_attacks_comparison(str(vis_dir / 'attack_comparison.png'))
            firewall.visualize_feature_extraction(str(vis_dir))
            firewall.visualize_defense_effects_all(str(vis_dir))
            
            logger.info(f"\nAll visualizations saved to: {vis_dir}")
        
        logger.info("\nOperation completed!")
        
    except KeyboardInterrupt:
        logger.info("\n\nOperation interrupted by user!")
    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
