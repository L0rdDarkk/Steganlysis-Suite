#!/usr/bin/env python3
"""
StegAnalysis Suite - SVM Model
Optimized Support Vector Machine implementation for steganography detection
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import json
import time
from datetime import datetime

# Machine learning imports
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Feature extraction
from skimage import feature
import cv2


class StegSVM:
    """Advanced SVM implementation optimized for steganography detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.kernel = config.get('kernel', 'rbf')  # linear, poly, rbf, sigmoid
        self.C = config.get('C', 1.0)
        self.gamma = config.get('gamma', 'scale')
        self.degree = config.get('degree', 3)  # For polynomial kernel
        self.probability = config.get('probability', True)
        
        # Feature processing
        self.feature_selection_method = config.get('feature_selection', 'selectkbest')  # selectkbest, rfe, lasso
        self.n_features = config.get('n_features', 100)
        self.scaling_method = config.get('scaling', 'standard')  # standard, robust, none
        self.use_pca = config.get('use_pca', False)
        self.pca_components = config.get('pca_components', 50)
        
        # Training parameters
        self.cv_folds = config.get('cv_folds', 5)
        self.hyperparameter_tuning = config.get('hyperparameter_tuning', True)
        self.random_state = config.get('random_state', 42)
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.label_encoder = None
        self.pipeline = None
        
        # Training results
        self.training_results = {}
        self.feature_importance = {}
        
    def create_feature_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline for features"""
        
        pipeline_steps = []
        
        # Feature scaling
        if self.scaling_method == 'standard':
            pipeline_steps.append(('scaler', StandardScaler()))
        elif self.scaling_method == 'robust':
            pipeline_steps.append(('scaler', RobustScaler()))
        
        # Feature selection
        if self.feature_selection_method == 'selectkbest':
            pipeline_steps.append(('feature_selection', 
                                 SelectKBest(f_classif, k=self.n_features)))
        elif self.feature_selection_method == 'rfe':
            # Will be configured after SVM is created
            pass
        elif self.feature_selection_method == 'lasso':
            pipeline_steps.append(('feature_selection',
                                 SelectFromModel(SVC(kernel='linear', C=0.01))))
        
        # PCA (optional)
        if self.use_pca:
            pipeline_steps.append(('pca', PCA(n_components=self.pca_components)))
        
        # SVM classifier
        svm_classifier = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            random_state=self.random_state
        )
        
        pipeline_steps.append(('svm', svm_classifier))
        
        # Handle RFE separately (needs estimator)
        if self.feature_selection_method == 'rfe':
            # Insert RFE before SVM
            pipeline_steps.insert(-1, ('feature_selection',
                                     RFE(svm_classifier, n_features_to_select=self.n_features)))
        
        self.pipeline = Pipeline(pipeline_steps)
        return self.pipeline
    
    def extract_statistical_features(self, image_path: str) -> np.ndarray:
        """Extract statistical features from image for SVM training"""
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return np.array([])
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # Basic statistical features
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray),
                np.ptp(gray)  # Peak-to-peak
            ])
            
            # Histogram features
            hist, _ = np.histogram(gray, bins=32, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            features.extend(hist.tolist())
            
            # Entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features.append(entropy)
            
            # Gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(grad_mag),
                np.std(grad_mag),
                np.max(grad_mag)
            ])
            
            # Texture features (LBP)
            try:
                lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
                lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
                lbp_hist = lbp_hist / lbp_hist.sum()
                features.extend(lbp_hist.tolist())
            except:
                features.extend([0.0] * 10)  # Fallback
            
            # Frequency domain features
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            features.extend([
                np.mean(magnitude_spectrum),
                np.std(magnitude_spectrum),
                np.max(magnitude_spectrum)
            ])
            
            # DCT features
            dct = cv2.dct(gray.astype(np.float32))
            features.extend([
                np.mean(dct),
                np.std(dct),
                np.mean(np.abs(dct))
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {str(e)}")
            return np.array([])
    
    def load_features_from_dataset(self, dataset_path: str, 
                                 labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels from dataset"""
        
        self.logger.info(f"Loading features from dataset: {dataset_path}")
        
        # Load labels
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        features_list = []
        labels_list = []
        
        total_images = len(labels_data)
        processed = 0
        
        for image_path, label_info in labels_data.items():
            try:
                features = self.extract_statistical_features(image_path)
                
                if len(features) > 0:
                    features_list.append(features)
                    labels_list.append(label_info['label'])
                
                processed += 1
                if processed % 100 == 0:
                    self.logger.info(f"Processed {processed}/{total_images} images")
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No features extracted from dataset")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        self.logger.info(f"Features loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter optimization using Grid Search"""
        
        self.logger.info("Starting hyperparameter optimization...")
        
        # Define parameter grids for different kernels
        param_grids = {
            'linear': {
                'svm__C': [0.1, 1, 10, 100],
                'svm__kernel': ['linear']
            },
            'rbf': {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svm__kernel': ['rbf']
            },
            'poly': {
                'svm__C': [0.1, 1, 10],
                'svm__degree': [2, 3, 4],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1],
                'svm__kernel': ['poly']
            }
        }
        
        best_models = {}
        
        for kernel_name, param_grid in param_grids.items():
            self.logger.info(f"Optimizing {kernel_name} kernel...")
            
            # Create pipeline for this kernel
            pipeline = self.create_feature_pipeline()
            
            # Grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                 random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            best_models[kernel_name] = {
                'model': grid_search.best_estimator_,
                'params': grid_search.best_params_,
                'score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"{kernel_name} - Best score: {grid_search.best_score_:.4f}")
            self.logger.info(f"{kernel_name} - Best params: {grid_search.best_params_}")
        
        # Select overall best model
        best_kernel = max(best_models.keys(), key=lambda k: best_models[k]['score'])
        self.pipeline = best_models[best_kernel]['model']
        
        self.logger.info(f"Best overall kernel: {best_kernel}")
        
        return {
            'best_kernel': best_kernel,
            'all_results': best_models,
            'best_score': best_models[best_kernel]['score'],
            'best_params': best_models[best_kernel]['params']
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the SVM model"""
        
        self.logger.info("Starting SVM training...")
        start_time = time.time()
        
        # Hyperparameter optimization
        if self.hyperparameter_tuning:
            hp_results = self.hyperparameter_optimization(X, y)
        else:
            # Use default pipeline
            self.pipeline = self.create_feature_pipeline()
            self.pipeline.fit(X, y)
            hp_results = {}
        
        # Cross-validation evaluation
        cv_scores = cross_val_score(
            self.pipeline, X, y,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                             random_state=self.random_state),
            scoring='f1'
        )
        
        # Training metrics
        y_pred_train = self.pipeline.predict(X)
        train_accuracy = accuracy_score(y, y_pred_train)
        train_f1 = f1_score(y, y_pred_train)
        
        # Validation metrics (if provided)
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_pred_val = self.pipeline.predict(X_val)
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_pred_val),
                'precision': precision_score(y_val, y_pred_val, zero_division=0),
                'recall': recall_score(y_val, y_pred_val, zero_division=0),
                'f1': f1_score(y_val, y_pred_val, zero_division=0)
            }
            
            # ROC AUC if probabilities available
            if hasattr(self.pipeline, 'predict_proba'):
                try:
                    y_proba_val = self.pipeline.predict_proba(X_val)[:, 1]
                    val_metrics['auc_roc'] = roc_auc_score(y_val, y_proba_val)
                except:
                    pass
        
        training_time = time.time() - start_time
        
        # Store results
        self.training_results = {
            'hyperparameter_optimization': hp_results,
            'cross_validation': {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            },
            'train_metrics': {
                'accuracy': train_accuracy,
                'f1': train_f1
            },
            'validation_metrics': val_metrics,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Feature importance (for linear kernel)
        self._extract_feature_importance()
        
        self.logger.info(f"Training completed in {training_time:.2f}s")
        self.logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.training_results
    
    def _extract_feature_importance(self):
        """Extract feature importance for interpretability"""
        
        try:
            # Get the SVM step from pipeline
            svm_step = self.pipeline.named_steps['svm']
            
            if svm_step.kernel == 'linear':
                # For linear kernel, coefficients indicate feature importance
                coef = svm_step.coef_[0]
                
                # Transform back through pipeline to get original feature importance
                if 'feature_selection' in self.pipeline.named_steps:
                    selector = self.pipeline.named_steps['feature_selection']
                    if hasattr(selector, 'get_support'):
                        # Create full-size coefficient array
                        full_coef = np.zeros(selector.get_support().shape[0])
                        full_coef[selector.get_support()] = coef
                        coef = full_coef
                
                self.feature_importance = {
                    'coefficients': coef.tolist(),
                    'abs_coefficients': np.abs(coef).tolist(),
                    'top_features': np.argsort(np.abs(coef))[::-1][:20].tolist()
                }
                
                self.logger.info("Feature importance extracted for linear SVM")
            else:
                self.logger.info(f"Feature importance not available for {svm_step.kernel} kernel")
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {str(e)}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model on test set"""
        
        if self.pipeline is None:
            raise ValueError("Model not trained")
        
        self.logger.info("Evaluating model on test set...")
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Probabilities (if available)
        y_proba = None
        if hasattr(self.pipeline, 'predict_proba'):
            try:
                y_proba = self.pipeline.predict_proba(X_test)[:, 1]
            except:
                pass
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_proba.tolist() if y_proba is not None else None
        }
        
        self.logger.info(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, "
                        f"F1: {metrics['f1']:.4f}")
        
        return evaluation_results
    
    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Predict steganography for a single image"""
        
        if self.pipeline is None:
            raise ValueError("Model not trained")
        
        try:
            # Extract features
            features = self.extract_statistical_features(image_path)
            
            if len(features) == 0:
                raise ValueError("Could not extract features from image")
            
            # Reshape for prediction
            features = features.reshape(1, -1)
            
            # Predict
            prediction = self.pipeline.predict(features)[0]
            
            # Get probability if available
            confidence = 0.5
            probability_clean = 0.5
            probability_steganographic = 0.5
            
            if hasattr(self.pipeline, 'predict_proba'):
                try:
                    proba = self.pipeline.predict_proba(features)[0]
                    probability_clean = float(proba[0])
                    probability_steganographic = float(proba[1])
                    confidence = float(max(proba))
                except:
                    pass
            
            result = {
                'prediction': 'steganographic' if prediction == 1 else 'clean',
                'confidence': confidence,
                'probability_clean': probability_clean,
                'probability_steganographic': probability_steganographic,
                'image_path': image_path
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting image {image_path}: {str(e)}")
            raise
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict steganography for multiple images"""
        
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append({
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
    
    def save_model(self, save_path: str):
        """Save the trained model and preprocessing components"""
        
        if self.pipeline is None:
            raise ValueError("No model to save")
        
        # Create save directory
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        joblib.dump(self.pipeline, save_path)
        
        # Save training results and metadata
        metadata = {
            'config': self.config,
            'training_results': self.training_results,
            'feature_importance': self.feature_importance,
            'model_type': 'SVM',
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = save_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        
        self.pipeline = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_results = metadata.get('training_results', {})
                self.feature_importance = metadata.get('feature_importance', {})
                self.config.update(metadata.get('config', {}))
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def plot_feature_importance(self, save_path: str = None, top_n: int = 20):
        """Plot feature importance (for linear SVM)"""
        
        if not self.feature_importance or 'coefficients' not in self.feature_importance:
            self.logger.warning("No feature importance data available")
            return
        
        coefficients = np.array(self.feature_importance['coefficients'])
        abs_coefficients = np.abs(coefficients)
        
        # Get top features
        top_indices = np.argsort(abs_coefficients)[::-1][:top_n]
        top_coeffs = coefficients[top_indices]
        
        # Create feature names (generic)
        feature_names = [f"Feature_{i}" for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['red' if coef < 0 else 'blue' for coef in top_coeffs]
        bars = plt.barh(range(len(top_coeffs)), top_coeffs, color=colors)
        
        plt.yticks(range(len(top_coeffs)), feature_names)
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Importance (Linear SVM)')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, coef) in enumerate(zip(bars, top_coeffs)):
            plt.text(coef + 0.01 * np.sign(coef), i, f'{coef:.3f}', 
                    va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Clean', 'Steganographic'],
                   yticklabels=['Clean', 'Steganographic'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        
        if self.pipeline is None:
            return {"status": "No model trained or loaded"}
        
        svm_step = self.pipeline.named_steps['svm']
        
        info = {
            'model_type': 'Support Vector Machine',
            'kernel': svm_step.kernel,
            'C': svm_step.C,
            'gamma': svm_step.gamma,
            'n_support_vectors': svm_step.n_support_.tolist() if hasattr(svm_step, 'n_support_') else None,
            'support_vectors_count': int(np.sum(svm_step.n_support_)) if hasattr(svm_step, 'n_support_') else None,
            'pipeline_steps': list(self.pipeline.named_steps.keys()),
            'training_results': self.training_results,
            'feature_importance_available': bool(self.feature_importance)
        }
        
        return info


# Example usage and testing
if __name__ == "__main__":
    # Configuration for testing
    config = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'feature_selection': 'selectkbest',
        'n_features': 50,
        'scaling': 'standard',
        'use_pca': False,
        'cv_folds': 5,
        'hyperparameter_tuning': True,
        'random_state': 42
    }
    
    # Initialize SVM
    svm = StegSVM(config)
    
    # Create pipeline
    pipeline = svm.create_feature_pipeline()
    
    print("SVM Pipeline created successfully:")
    print(pipeline)
    
    # Print model info
    info = svm.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"{key}: {value}")