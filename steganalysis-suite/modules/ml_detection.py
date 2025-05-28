#!/usr/bin/env python3
"""
StegAnalysis Suite - Machine Learning Detection Module
CNN and SVM-based steganography detection with GPU acceleration
"""

import os
import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pickle
import joblib

# Set TensorFlow log level BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Machine Learning imports with proper error handling
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    # SUCCESS MESSAGE - TensorFlow loaded
    logging.info(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    # Only log as info, not warning
    logging.info(f"TensorFlow not installed: {e}")
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    logging.error(f"TensorFlow error: {e}")

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
    logging.info("✅ scikit-learn loaded successfully")
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.info("scikit-learn not installed")

from scipy import stats
from scipy.fftpack import dct
import matplotlib.pyplot as plt


class MLDetectionEngine:
    """Machine Learning-based steganography detection engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.ml_config = self.config.get('ml_models', {})
        self.gpu_config = self.config.get('gpu', {})
        
        # Initialize GPU if available
        self._setup_gpu()
        
        # Load models
        self.cnn_model = None
        self.svm_model = None
        self.scaler = None
        
        self.cnn_available = False
        self.svm_available = False
        
        self._load_models()
    
    def _setup_gpu(self):
        """Setup GPU configuration for TensorFlow"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        if self.gpu_config.get('enabled', False):
            try:
                # Configure GPU memory growth
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit if specified
                    memory_limit = self.gpu_config.get('memory_limit_mb')
                    if memory_limit:
                        tf.config.experimental.set_memory_limit(gpus[0], memory_limit)
                    
                    self.logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
                else:
                    self.logger.info("GPU enabled in config but no GPUs found")
            except Exception as e:
                self.logger.error(f"Failed to setup GPU: {str(e)}")
        else:
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.logger.info("Using CPU for ML inference")
    
    def _load_models(self):
        """Load pre-trained CNN and SVM models"""
        # Load CNN model
        if TENSORFLOW_AVAILABLE:
            cnn_path = self.ml_config.get('cnn', {}).get('model_path')
            if cnn_path and Path(cnn_path).exists():
                try:
                    self.cnn_model = keras.models.load_model(cnn_path)
                    self.cnn_available = True
                    self.logger.info(f"CNN model loaded from {cnn_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load CNN model: {str(e)}")
            else:
                self.logger.info("CNN model not found. Will create default model if needed.")
        
        # Load SVM model
        if SKLEARN_AVAILABLE:
            svm_path = self.ml_config.get('svm', {}).get('model_path')
            scaler_path = self.ml_config.get('svm', {}).get('scaler_path')
            
            if svm_path and Path(svm_path).exists():
                try:
                    self.svm_model = joblib.load(svm_path)
                    self.svm_available = True
                    self.logger.info(f"SVM model loaded from {svm_path}")
                    
                    if scaler_path and Path(scaler_path).exists():
                        self.scaler = joblib.load(scaler_path)
                        self.logger.info(f"Scaler loaded from {scaler_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load SVM model: {str(e)}")
            else:
                self.logger.info("SVM model not found. Will create default model if needed.")
    
    def create_cnn_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Optional[Any]:
        """
        Create a CNN model for steganography detection
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Compiled Keras model or None if TensorFlow not available
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available for CNN model creation")
            return None
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def preprocess_image_for_cnn(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image for CNN input
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Get target size from config
            input_shape = self.ml_config.get('cnn', {}).get('input_shape', [224, 224, 3])
            target_size = (input_shape[1], input_shape[0])  # (width, height)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed for {image_path}: {str(e)}")
            return None
    
    def cnn_predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict steganography using CNN model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing CNN prediction results
        """
        if not self.cnn_available:
            if TENSORFLOW_AVAILABLE:
                # Create default model
                self.cnn_model = self.create_cnn_model()
                if self.cnn_model is not None:
                    self.cnn_available = True
                    self.logger.info("Using untrained CNN model - results may be unreliable")
                else:
                    return {
                        'detected': False,
                        'confidence': 0.0,
                        'error': 'Failed to create CNN model',
                        'method': 'cnn_prediction'
                    }
            else:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'error': 'TensorFlow not available',
                    'method': 'cnn_prediction'
                }
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image_for_cnn(image_path)
            if processed_image is None:
                raise ValueError("Image preprocessing failed")
            
            # Make prediction
            prediction = self.cnn_model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            
            # Apply threshold
            threshold = self.config.get('detection', {}).get('thresholds', {}).get('cnn_confidence', 0.8)
            detected = confidence > threshold
            
            result = {
                'detected': detected,
                'confidence': confidence,
                'prediction_raw': float(prediction[0][0]),
                'threshold_used': threshold,
                'method': 'cnn_prediction',
                'model_info': {
                    'input_shape': self.cnn_model.input_shape[1:],
                    'parameters': self.cnn_model.count_params()
                }
            }
            
            self.logger.info(f"CNN prediction completed for {image_path}: "
                           f"Detected={detected}, Confidence={confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"CNN prediction failed for {image_path}: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'error': str(e),
                'method': 'cnn_prediction'
            }
    
    def extract_features_for_svm(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract features from image for SVM classification
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector or None if failed
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            features = []
            
            # Process each color channel
            for channel in range(3):
                channel_data = image[:, :, channel]
                
                # 1. Statistical features
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.var(channel_data),
                    stats.skew(channel_data.flatten()),
                    stats.kurtosis(channel_data.flatten())
                ])
                
                # 2. Entropy
                hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
                hist = hist + 1e-10  # Avoid log(0)
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob))
                features.append(entropy)
                
                # 3. LSB analysis
                lsb_plane = channel_data & 1
                lsb_mean = np.mean(lsb_plane)
                lsb_var = np.var(lsb_plane)
                features.extend([lsb_mean, lsb_var])
                
                # 4. Gradient features
                grad_x = cv2.Sobel(channel_data, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(channel_data, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                features.extend([
                    np.mean(gradient_magnitude),
                    np.std(gradient_magnitude)
                ])
                
                # 5. Texture features (GLCM-inspired)
                texture_features = self._extract_texture_features(channel_data)
                features.extend(texture_features)
            
            # 6. DCT features (for JPEG images)
            try:
                dct_features = self._extract_dct_features(image)
                features.extend(dct_features)
            except:
                # If DCT extraction fails, pad with zeros
                features.extend([0.0] * 10)
            
            # 7. Noise analysis
            noise_features = self._extract_noise_features(image)
            features.extend(noise_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {str(e)}")
            return None
    
    def _extract_texture_features(self, channel: np.ndarray) -> list:
        """Extract texture features from image channel"""
        # Local Binary Pattern-inspired features
        features = []
        
        # Calculate local variance
        kernel = np.ones((3, 3), np.float32) / 9
        mean_filtered = cv2.filter2D(channel.astype(np.float32), -1, kernel)
        variance = (channel.astype(np.float32) - mean_filtered) ** 2
        local_variance = cv2.filter2D(variance, -1, kernel)
        
        features.extend([
            np.mean(local_variance),
            np.std(local_variance),
            np.max(local_variance),
            np.min(local_variance)
        ])
        
        return features
    
    def _extract_dct_features(self, image: np.ndarray) -> list:
        """Extract DCT-based features"""
        # Convert to grayscale for DCT analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Apply DCT to 8x8 blocks
        height, width = gray.shape
        dct_coeffs = []
        
        for i in range(0, height - 7, 8):
            for j in range(0, width - 7, 8):
                block = gray[i:i+8, j:j+8]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten()[1:])  # Skip DC coefficient
        
        if dct_coeffs:
            dct_array = np.array(dct_coeffs)
            return [
                np.mean(dct_array),
                np.std(dct_array),
                np.var(dct_array),
                stats.skew(dct_array),
                stats.kurtosis(dct_array),
                np.sum(dct_array == 0) / len(dct_array),  # Zero coefficient ratio
                np.percentile(dct_array, 25),
                np.percentile(dct_array, 75),
                np.median(dct_array),
                np.max(np.abs(dct_array))
            ]
        else:
            return [0.0] * 10
    
    def _extract_noise_features(self, image: np.ndarray) -> list:
        """Extract noise-related features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Laplacian for noise estimation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = np.var(laplacian)
        
        # High-frequency content analysis
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        high_freq = cv2.filter2D(gray, -1, kernel)
        high_freq_energy = np.sum(high_freq ** 2)
        
        # Edge density
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return [
            noise_variance,
            high_freq_energy,
            edge_density,
            np.std(laplacian),
            np.mean(np.abs(laplacian))
        ]
    
    def svm_predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict steganography using SVM model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing SVM prediction results
        """
        if not self.svm_available:
            if SKLEARN_AVAILABLE:
                self.logger.info("SVM model not available - creating default model")
                # Create a simple default SVM (will be untrained)
                self.svm_model = SVC(probability=True, kernel='rbf')
                # Note: This model needs training data to be useful
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'error': 'SVM model not trained',
                    'method': 'svm_prediction'
                }
            else:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'error': 'scikit-learn not available',
                    'method': 'svm_prediction'
                }
        
        try:
            # Extract features
            features = self.extract_features_for_svm(image_path)
            if features is None:
                raise ValueError("Feature extraction failed")
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.svm_model.predict(features)[0]
            
            # Get prediction probabilities if available
            try:
                probabilities = self.svm_model.predict_proba(features)[0]
                confidence = float(probabilities[1])  # Probability of steganographic class
            except:
                # If probabilities not available, use decision function
                try:
                    decision = self.svm_model.decision_function(features)[0]
                    confidence = float(1.0 / (1.0 + np.exp(-decision)))  # Sigmoid transformation
                except:
                    confidence = 0.5  # Default confidence
            
            # Apply threshold
            threshold = self.config.get('detection', {}).get('thresholds', {}).get('svm_confidence', 0.75)
            detected = confidence > threshold
            
            result = {
                'detected': detected,
                'confidence': confidence,
                'prediction': int(prediction),
                'threshold_used': threshold,
                'method': 'svm_prediction',
                'feature_count': len(features[0])
            }
            
            self.logger.info(f"SVM prediction completed for {image_path}: "
                           f"Detected={detected}, Confidence={confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"SVM prediction failed for {image_path}: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'error': str(e),
                'method': 'svm_prediction'
            }
    
    def ensemble_predict(self, image_path: str) -> Dict[str, Any]:
        """
        Ensemble prediction using both CNN and SVM
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing ensemble prediction results
        """
        results = {
            'cnn': None,
            'svm': None,
            'ensemble': {
                'detected': False,
                'confidence': 0.0,
                'method': 'ensemble_prediction'
            }
        }
        
        # Get CNN prediction
        if self.cnn_available:
            results['cnn'] = self.cnn_predict(image_path)
        
        # Get SVM prediction
        if self.svm_available:
            results['svm'] = self.svm_predict(image_path)
        
        # Combine predictions
        predictions = []
        confidences = []
        
        if results['cnn'] and 'error' not in results['cnn']:
            predictions.append(results['cnn']['detected'])
            confidences.append(results['cnn']['confidence'])
        
        if results['svm'] and 'error' not in results['svm']:
            predictions.append(results['svm']['detected'])
            confidences.append(results['svm']['confidence'])
        
        if predictions:
            # Ensemble strategy: weighted average
            cnn_weight = 0.6  # CNN typically more accurate for image analysis
            svm_weight = 0.4
            
            if len(predictions) == 2:
                ensemble_confidence = (confidences[0] * cnn_weight + 
                                     confidences[1] * svm_weight)
                ensemble_detected = ensemble_confidence > 0.5
            else:
                # Only one model available
                ensemble_confidence = confidences[0]
                ensemble_detected = predictions[0]
            
            results['ensemble'] = {
                'detected': ensemble_detected,
                'confidence': float(ensemble_confidence),
                'method': 'ensemble_prediction',
                'models_used': len(predictions),
                'individual_predictions': predictions,
                'individual_confidences': confidences
            }
        
        self.logger.info(f"Ensemble prediction completed for {image_path}: "
                        f"Detected={results['ensemble']['detected']}, "
                        f"Confidence={results['ensemble']['confidence']:.3f}")
        
        return results
    
    def save_models(self, output_dir: str):
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CNN model
        if self.cnn_model is not None:
            cnn_path = output_path / "cnn_stego_detector.h5"
            self.cnn_model.save(str(cnn_path))
            self.logger.info(f"CNN model saved to {cnn_path}")
        
        # Save SVM model
        if self.svm_model is not None:
            svm_path = output_path / "svm_stego_detector.pkl"
            joblib.dump(self.svm_model, str(svm_path))
            self.logger.info(f"SVM model saved to {svm_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = output_path / "scaler.pkl"
            joblib.dump(self.scaler, str(scaler_path))
            self.logger.info(f"Scaler saved to {scaler_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'cnn_available': self.cnn_available,
            'svm_available': self.svm_available,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'gpu_enabled': self.gpu_config.get('enabled', False)
        }
        
        if self.cnn_model is not None:
            info['cnn_info'] = {
                'input_shape': self.cnn_model.input_shape,
                'output_shape': self.cnn_model.output_shape,
                'parameters': self.cnn_model.count_params(),
                'layers': len(self.cnn_model.layers)
            }
        
        if self.svm_model is not None:
            info['svm_info'] = {
                'kernel': getattr(self.svm_model, 'kernel', 'unknown'),
                'support_vectors': getattr(self.svm_model, 'n_support_', 'unknown')
            }
        
        return info


# Create alias for backward compatibility
MLDetection = MLDetectionEngine