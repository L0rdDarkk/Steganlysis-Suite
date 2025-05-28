#!/usr/bin/env python3
"""
StegAnalysis Suite - Model Training
Advanced machine learning training for steganography detection
"""

import os
import json
import logging
import pickle
import warnings
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import argparse
from datetime import datetime
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be disabled.")

# Image processing imports
import cv2
from PIL import Image
from skimage import feature, filters, measure, segmentation

# FIXED: Import GLCM functions with proper error handling
try:
    from skimage.feature import graycomatrix, graycoprops
    GLCM_AVAILABLE = True
except ImportError:
    try:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
        GLCM_AVAILABLE = True
    except ImportError:
        # Create dummy functions if not available
        def graycomatrix(*args, **kwargs):
            return None
        def graycoprops(*args, **kwargs):
            return [0.0, 0.0, 0.0, 0.0]
        GLCM_AVAILABLE = False

import scipy.stats as stats
from scipy.fft import fft2, fftshift

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FeatureExtractor:
    """Advanced feature extraction for steganography detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction settings
        self.feature_types = config.get('feature_types', [
            'statistical', 'texture', 'frequency', 'gradient', 'histogram'
        ])
        self.image_size = config.get('image_size', (256, 256))
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract comprehensive features from an image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return np.array([])
            
            # Resize image
            img_resized = cv2.resize(img, self.image_size)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # Extract different types of features
            if 'statistical' in self.feature_types:
                features.extend(self._extract_statistical_features(gray))
            
            if 'texture' in self.feature_types:
                features.extend(self._extract_texture_features(gray))
            
            if 'frequency' in self.feature_types:
                features.extend(self._extract_frequency_features(gray))
            
            if 'gradient' in self.feature_types:
                features.extend(self._extract_gradient_features(gray))
            
            if 'histogram' in self.feature_types:
                features.extend(self._extract_histogram_features(gray))
            
            if 'lbp' in self.feature_types:
                features.extend(self._extract_lbp_features(gray))
            
            if 'glcm' in self.feature_types:
                features.extend(self._extract_glcm_features(gray))
            
            if 'wavelet' in self.feature_types:
                features.extend(self._extract_wavelet_features(gray))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {str(e)}")
            return np.array([])
    
    def _extract_statistical_features(self, image: np.ndarray) -> List[float]:
        """Extract basic statistical features"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.var(image),
            np.median(image),
            stats.skew(image.flatten()),
            stats.kurtosis(image.flatten()),
            np.min(image),
            np.max(image),
            np.ptp(image)  # Peak-to-peak
        ])
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features.append(np.percentile(image, p))
        
        # Entropy
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(entropy)
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture-based features"""
        features = []
        
        # Sobel filters
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(sobel_mag),
            np.std(sobel_mag),
            np.max(sobel_mag)
        ])
        
        # Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            np.var(laplacian)
        ])
        
        # Canny edge density
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return features
    
    def _extract_frequency_features(self, image: np.ndarray) -> List[float]:
        """Extract frequency domain features"""
        features = []
        
        # FFT
        f_transform = fft2(image)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Frequency domain statistics
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.max(magnitude_spectrum)
        ])
        
        # DCT features
        dct = cv2.dct(image.astype(np.float32))
        features.extend([
            np.mean(dct),
            np.std(dct),
            np.mean(np.abs(dct))
        ])
        
        # Energy in different frequency bands
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency energy (center 25%)
        low_freq = magnitude_spectrum[
            center_h - h//8:center_h + h//8,
            center_w - w//8:center_w + w//8
        ]
        features.append(np.sum(low_freq**2))
        
        # High frequency energy (corners)
        high_freq_mask = np.ones_like(magnitude_spectrum)
        high_freq_mask[center_h - h//4:center_h + h//4, center_w - w//4:center_w + w//4] = 0
        high_freq_energy = np.sum((magnitude_spectrum * high_freq_mask)**2)
        features.append(high_freq_energy)
        
        return features
    
    def _extract_gradient_features(self, image: np.ndarray) -> List[float]:
        """Extract gradient-based features"""
        features = []
        
        # Compute gradients
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        # Gradient statistics
        features.extend([
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            np.max(grad_magnitude),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])
        
        # Gradient direction histogram
        hist, _ = np.histogram(grad_direction, bins=8, range=(-np.pi, np.pi))
        hist = hist / hist.sum()
        features.extend(hist.tolist())
        
        return features
    
    def _extract_histogram_features(self, image: np.ndarray) -> List[float]:
        """Extract histogram-based features"""
        features = []
        
        # Intensity histogram
        hist, _ = np.histogram(image, bins=32, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        features.extend(hist.tolist())
        
        # Histogram statistics
        features.extend([
            np.mean(hist),
            np.std(hist),
            np.max(hist),
            np.min(hist)
        ])
        
        return features
    
    def _extract_lbp_features(self, image: np.ndarray) -> List[float]:
        """Extract Local Binary Pattern features"""
        features = []
        
        try:
            # LBP with different parameters
            radius = 3
            n_points = 8 * radius
            
            lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
            
            # LBP histogram
            n_bins = n_points + 2  # uniform patterns + non-uniform
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
            hist = hist / hist.sum()
            features.extend(hist.tolist())
            
        except Exception as e:
            self.logger.warning(f"LBP extraction failed: {str(e)}")
            # Return zeros if LBP fails
            features.extend([0.0] * 26)  # Default size for 8*3 radius
        
        return features
    
    def _extract_glcm_features(self, image: np.ndarray) -> List[float]:
        """Extract Gray-Level Co-occurrence Matrix features"""
        features = []
        
        if not GLCM_AVAILABLE:
            # Return dummy features if GLCM not available
            features.extend([0.0] * 10)  # 5 features × 2 distances
            return features
        
        try:
            # GLCM with different angles and distances
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            # Reduce image levels for GLCM computation
            image_glcm = (image // 32).astype(np.uint8)  # 8 levels
            
            for distance in distances:
                glcm = graycomatrix(image_glcm, [distance], angles, 
                                 levels=8, symmetric=True, normed=True)
                
                # Extract Haralick features
                features.extend([
                    np.mean(graycoprops(glcm, 'contrast')),
                    np.mean(graycoprops(glcm, 'dissimilarity')),
                    np.mean(graycoprops(glcm, 'homogeneity')),
                    np.mean(graycoprops(glcm, 'energy')),
                    np.mean(graycoprops(glcm, 'correlation'))
                ])
                
        except Exception as e:
            self.logger.warning(f"GLCM extraction failed: {str(e)}")
            # Return zeros if GLCM fails
            features.extend([0.0] * 10)  # 5 features × 2 distances
        
        return features
    
    def _extract_wavelet_features(self, image: np.ndarray) -> List[float]:
        """Extract wavelet-based features"""
        features = []
        
        try:
            import pywt
            
            # Wavelet decomposition
            coeffs = pywt.wavedec2(image, 'db4', level=3)
            
            # Extract statistics from each subband
            for i, coeff in enumerate(coeffs):
                if i == 0:  # Approximation coefficients
                    features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        np.max(coeff)
                    ])
                else:  # Detail coefficients (horizontal, vertical, diagonal)
                    for subband in coeff:
                        features.extend([
                            np.mean(np.abs(subband)),
                            np.std(subband),
                            np.max(np.abs(subband))
                        ])
                        
        except ImportError:
            self.logger.warning("PyWavelets not available, skipping wavelet features")
            features.extend([0.0] * 30)  # Placeholder
        except Exception as e:
            self.logger.warning(f"Wavelet extraction failed: {str(e)}")
            features.extend([0.0] * 30)
        
        return features


class ModelTrainer:
    """Simplified model trainer for steganography detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Paths
        self.dataset_dir = Path("datasets/images")
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training settings
        self.random_state = 42
        
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data from our dataset structure."""
        images = []
        labels = []
        
        # Load training split information
        splits_file = Path("datasets/metadata/splits.json")
        if splits_file.exists():
            import json
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            # Load clean training images
            clean_dir = self.dataset_dir / "clean"
            for filename in splits["train"]["clean"]:
                img_path = clean_dir / filename
                try:
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(0)  # Clean = 0
                except Exception as e:
                    self.logger.warning(f"Failed to load clean image {img_path}: {e}")
                    
            # Load steganographic training images
            stego_dir = self.dataset_dir / "steganographic"
            for filename in splits["train"]["steganographic"]:
                img_path = stego_dir / filename
                try:
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Steganographic = 1
                except Exception as e:
                    self.logger.warning(f"Failed to load stego image {img_path}: {e}")
        else:
            # Fallback: load all images if no splits file
            self.logger.warning("No splits file found, loading all images")
            
            # Load clean images
            clean_dir = self.dataset_dir / "clean"
            for img_path in clean_dir.glob("*.jpg"):
                try:
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(0)  # Clean = 0
                except Exception as e:
                    self.logger.warning(f"Failed to load clean image {img_path}: {e}")
                    
            # Load steganographic images
            stego_dir = self.dataset_dir / "steganographic"
            for img_path in stego_dir.glob("*.jpg"):
                try:
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Steganographic = 1
                except Exception as e:
                    self.logger.warning(f"Failed to load stego image {img_path}: {e}")
                
        if not images:
            raise ValueError("No training images found. Please run setup_dataset.py first!")
            
        self.logger.info(f"Loaded {len(images)} training images ({sum(labels)} steganographic, {len(labels)-sum(labels)} clean)")
        return np.array(images), np.array(labels)
    
    def _load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and preprocess a single image."""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Resize to standard size
            img_resized = cv2.resize(img, (224, 224))
            
            # Convert to RGB and normalize
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            return img_normalized
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def create_cnn_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Create CNN model for steganographic detection."""
        model = tf.keras.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            
            # Dense layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def train_cnn_model(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """Train CNN model."""
        self.logger.info("🤖 Training CNN Model...")
        
        # Create model
        model = self.create_cnn_model(X_train.shape[1:])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return model
    
    def train_svm_model(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """Train SVM model using simple features."""
        self.logger.info("🔧 Training SVM Model...")
        
        # Extract simple features for SVM
        X_train_features = self._extract_simple_features(X_train)
        
        # Create and train SVM
        svm_model = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        svm_model.fit(X_train_features, y_train)
        
        return svm_model
    
    def _extract_simple_features(self, images: np.ndarray) -> np.ndarray:
        """Extract simple statistical features from images."""
        features = []
        
        for img in images:
            # Convert to grayscale
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Basic statistical features
            img_features = [
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray),
                np.ptp(gray)  # Peak-to-peak
            ]
            
            # Histogram features
            hist, _ = np.histogram(gray, bins=16, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            img_features.extend(hist.tolist())
            
            features.append(img_features)
        
        return np.array(features)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_type: str = "CNN") -> Dict[str, float]:
        """Evaluate model performance."""
        if model_type == "CNN":
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:  # SVM
            X_test_features = self._extract_simple_features(X_test)
            y_pred = model.predict(X_test_features)
            y_pred_proba = model.predict_proba(X_test_features)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def save_models(self, cnn_model: tf.keras.Model, svm_model: SVC):
        """Save trained models."""
        self.logger.info("💾 Saving models...")
        
        # Save CNN model
        cnn_path = self.models_dir / "cnn_stego_detector.h5"
        cnn_model.save(str(cnn_path))
        self.logger.info(f"CNN model saved to {cnn_path}")
        
        # Save SVM model
        svm_path = self.models_dir / "svm_stego_detector.pkl"
        import joblib
        joblib.dump(svm_model, str(svm_path))
        self.logger.info(f"SVM model saved to {svm_path}")
    
    def train_models(self):
        """Main training function."""
        try:
            self.logger.info("🚀 StegAnalysis Suite - Model Training")
            self.logger.info("=" * 50)
            
            # Load data
            self.logger.info("📊 Loading training data...")
            X, y = self.load_training_data()
            
            # Split data
            self.logger.info("🔀 Splitting data...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
            )
            
            self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train CNN model
            if TF_AVAILABLE:
                cnn_model = self.train_cnn_model(X_train, y_train, X_val, y_val)
                
                # Evaluate CNN
                cnn_metrics = self.evaluate_model(cnn_model, X_test, y_test, "CNN")
                self.logger.info(f"✅ CNN Results: Accuracy={cnn_metrics['accuracy']:.4f}, "
                               f"F1={cnn_metrics['f1_score']:.4f}")
            else:
                self.logger.warning("TensorFlow not available, skipping CNN training")
                cnn_model = None
            
            # Train SVM model
            svm_model = self.train_svm_model(X_train, y_train)
            
            # Evaluate SVM
            svm_metrics = self.evaluate_model(svm_model, X_test, y_test, "SVM")
            self.logger.info(f"✅ SVM Results: Accuracy={svm_metrics['accuracy']:.4f}, "
                           f"F1={svm_metrics['f1_score']:.4f}")
            
            # Save models
            if cnn_model is not None:
                self.save_models(cnn_model, svm_model)
            else:
                # Save only SVM
                svm_path = self.models_dir / "svm_stego_detector.pkl"
                import joblib
                joblib.dump(svm_model, str(svm_path))
                self.logger.info(f"SVM model saved to {svm_path}")
            
            self.logger.info("🎉 Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main function for model training."""
    trainer = ModelTrainer()
    trainer.train_models()


if __name__ == "__main__":
    main()