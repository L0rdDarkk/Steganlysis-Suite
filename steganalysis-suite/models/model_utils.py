#!/usr/bin/env python3
"""
StegAnalysis Suite - Model Utilities
Comprehensive utilities for model management, evaluation, and deployment
"""

import os
import logging
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import time
import warnings
from collections import defaultdict

# Machine learning imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Image processing
import cv2
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelManager:
    """Centralized model management and deployment system"""
    
    def __init__(self, models_dir: str = "models/trained"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.registered_models = {}
        self.model_metadata = {}
        
        # Load existing model registry
        self._load_model_registry()
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_path = self.models_dir / "model_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    self.registered_models = data.get('models', {})
                    self.model_metadata = data.get('metadata', {})
                    
                self.logger.info(f"Loaded {len(self.registered_models)} models from registry")
            except Exception as e:
                self.logger.error(f"Error loading model registry: {str(e)}")
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_path = self.models_dir / "model_registry.json"
        
        try:
            registry_data = {
                'models': self.registered_models,
                'metadata': self.model_metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving model registry: {str(e)}")
    
    def register_model(self, model_name: str, model_path: str, 
                      model_type: str, metadata: Dict[str, Any] = None):
        """Register a trained model"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.registered_models[model_name] = {
            'path': str(model_path),
            'type': model_type,
            'registered_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(model_path)
        }
        
        if metadata:
            self.model_metadata[model_name] = metadata
        
        self._save_model_registry()
        self.logger.info(f"Model '{model_name}' registered successfully")
    
    def load_model(self, model_name: str):
        """Load a registered model"""
        
        if model_name not in self.registered_models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.registered_models[model_name]
        model_path = model_info['path']
        model_type = model_info['type']
        
        try:
            if model_type.lower() == 'tensorflow' or model_path.endswith('.h5'):
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available")
                model = keras.models.load_model(model_path)
            elif model_type.lower() == 'sklearn' or model_path.endswith('.pkl'):
                model = joblib.load(model_path)
            else:
                # Try generic pickle loading
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            self.logger.info(f"Model '{model_name}' loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {str(e)}")
            raise
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.registered_models.copy()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        
        if model_name not in self.registered_models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.registered_models[model_name].copy()
        
        # Add metadata if available
        if model_name in self.model_metadata:
            model_info['metadata'] = self.model_metadata[model_name]
        
        return model_info
    
    def delete_model(self, model_name: str, delete_file: bool = False):
        """Remove a model from registry"""
        
        if model_name not in self.registered_models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.registered_models[model_name]
        
        if delete_file:
            try:
                os.remove(model_info['path'])
                self.logger.info(f"Model file deleted: {model_info['path']}")
            except Exception as e:
                self.logger.warning(f"Could not delete model file: {str(e)}")
        
        # Remove from registry
        del self.registered_models[model_name]
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]
        
        self._save_model_registry()
        self.logger.info(f"Model '{model_name}' removed from registry")


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            
            # Handle different prediction formats
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Multi-class predictions (e.g., from softmax)
                y_pred_class = np.argmax(y_pred, axis=1)
                y_proba = y_pred[:, 1] if y_pred.shape[1] == 2 else None
            else:
                # Binary predictions
                y_pred_class = y_pred
                y_proba = None
                
            # Get probabilities if available
            if hasattr(model, 'predict_proba') and y_proba is None:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                except:
                    pass
                    
        else:
            raise ValueError("Model does not have a predict method")
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(y_test, y_pred_class, y_proba)
        
        # Additional analysis
        analysis = {
            'confusion_matrix': confusion_matrix(y_test, y_pred_class).tolist(),
            'classification_report': classification_report(y_test, y_pred_class, output_dict=True),
            'class_distribution': {
                'actual': np.bincount(y_test).tolist(),
                'predicted': np.bincount(y_pred_class).tolist()
            }
        }
        
        # ROC analysis
        if y_proba is not None:
            roc_analysis = self._roc_analysis(y_test, y_proba)
            analysis['roc_analysis'] = roc_analysis
        
        # Store results
        evaluation_result = {
            'model_name': model_name,
            'metrics': metrics,
            'analysis': analysis,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_size': len(y_test)
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        self.logger.info(f"Evaluation completed for {model_name}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return evaluation_result
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'balanced_accuracy': self._calculate_balanced_accuracy(y_true, y_pred)
        }
        
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_proba)
            except ValueError:
                # Handle case where only one class is present
                metrics['auc_roc'] = 0.0
                metrics['average_precision'] = 0.0
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy"""
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (recall + specificity) / 2.0
    
    def _roc_analysis(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Perform ROC curve analysis"""
        
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            # Find optimal threshold (Youden's index)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            return {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(auc),
                'optimal_threshold': float(optimal_threshold),
                'optimal_tpr': float(tpr[optimal_idx]),
                'optimal_fpr': float(fpr[optimal_idx])
            }
        except Exception as e:
            self.logger.warning(f"ROC analysis failed: {str(e)}")
            return {}
    
    def compare_models(self, models_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple model evaluation results"""
        
        if len(models_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Extract metrics for comparison
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        comparison_data = {}
        
        for metric in comparison_metrics:
            comparison_data[metric] = {}
            for model_name, results in models_results.items():
                if metric in results['metrics']:
                    comparison_data[metric][model_name] = results['metrics'][metric]
        
        # Find best model for each metric
        best_models = {}
        for metric, values in comparison_data.items():
            if values:
                best_model = max(values.keys(), key=lambda k: values[k])
                best_models[metric] = {
                    'model': best_model,
                    'value': values[best_model]
                }
        
        # Overall ranking (based on F1 score)
        f1_scores = comparison_data.get('f1', {})
        if f1_scores:
            ranking = sorted(f1_scores.keys(), key=lambda k: f1_scores[k], reverse=True)
        else:
            ranking = list(models_results.keys())
        
        comparison_result = {
            'comparison_metrics': comparison_data,
            'best_models_by_metric': best_models,
            'overall_ranking': ranking,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        return comparison_result
    
    def cross_model_analysis(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-model analysis and ensemble recommendations"""
        
        # Analyze prediction agreements
        predictions = {}
        probabilities = {}
        
        for result in model_results:
            model_name = result['model_name']
            # Note: This would need actual predictions, not just metrics
            # Placeholder for demonstration
            predictions[model_name] = []
            probabilities[model_name] = []
        
        # Agreement analysis (placeholder)
        agreement_analysis = {
            'pairwise_agreements': {},
            'consensus_predictions': [],
            'disagreement_cases': []
        }
        
        # Ensemble recommendations
        ensemble_recommendations = {
            'voting_ensemble': list(predictions.keys()),
            'weighted_ensemble': {
                model: 1.0 / len(predictions) for model in predictions.keys()
            },
            'stacking_candidates': list(predictions.keys())[:3]  # Top 3 models
        }
        
        return {
            'agreement_analysis': agreement_analysis,
            'ensemble_recommendations': ensemble_recommendations
        }


class ModelBenchmark:
    """Performance benchmarking and profiling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.benchmark_results = {}
    
    def benchmark_inference_speed(self, model, X_test: np.ndarray, 
                                 n_runs: int = 100) -> Dict[str, float]:
        """Benchmark model inference speed"""
        
        self.logger.info(f"Benchmarking inference speed with {n_runs} runs...")
        
        # Warm-up run
        _ = model.predict(X_test[:1])
        
        # Benchmark runs
        times = []
        for _ in range(n_runs):
            start_time = time.time()
            _ = model.predict(X_test)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        
        benchmark_result = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'samples_per_second': float(len(X_test) / np.mean(times)),
            'total_samples': len(X_test),
            'n_runs': n_runs
        }
        
        self.logger.info(f"Average inference time: {benchmark_result['mean_time']:.4f}s")
        self.logger.info(f"Throughput: {benchmark_result['samples_per_second']:.1f} samples/sec")
        
        return benchmark_result
    
    def benchmark_memory_usage(self, model, X_test: np.ndarray) -> Dict[str, Any]:
        """Benchmark model memory usage"""
        
        import psutil
        import gc
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection
        gc.collect()
        
        # Load model and make prediction
        memory_before = process.memory_info().rss / 1024 / 1024
        _ = model.predict(X_test)
        memory_after = process.memory_info().rss / 1024 / 1024
        
        memory_result = {
            'baseline_memory_mb': baseline_memory,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_after - memory_before,
            'peak_memory_mb': memory_after
        }
        
        return memory_result
    
    def profile_model_complexity(self, model) -> Dict[str, Any]:
        """Analyze model complexity metrics"""
        
        complexity_metrics = {}
        
        # For scikit-learn models
        if hasattr(model, 'n_features_in_'):
            complexity_metrics['n_features'] = model.n_features_in_
        
        if hasattr(model, 'n_support_'):  # SVM
            complexity_metrics['n_support_vectors'] = int(np.sum(model.n_support_))
        
        if hasattr(model, 'n_estimators'):  # Ensemble methods
            complexity_metrics['n_estimators'] = model.n_estimators
        
        if hasattr(model, 'max_depth'):  # Tree-based methods
            complexity_metrics['max_depth'] = model.max_depth
        
        # For TensorFlow models
        if TF_AVAILABLE and isinstance(model, keras.Model):
            complexity_metrics['total_params'] = model.count_params()
            complexity_metrics['trainable_params'] = sum([
                keras.backend.count_params(layer) for layer in model.trainable_weights
            ])
            complexity_metrics['layers_count'] = len(model.layers)
        
        return complexity_metrics


class ModelDeployment:
    """Model deployment and serving utilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def create_prediction_service(self, model, preprocessing_pipeline=None):
        """Create a prediction service wrapper"""
        
        class PredictionService:
            def __init__(self, model, preprocessor=None):
                self.model = model
                self.preprocessor = preprocessor
                self.logger = logging.getLogger(__name__)
                
            def predict_image(self, image_path: str) -> Dict[str, Any]:
                """Predict from image path"""
                try:
                    # Load and preprocess image
                    if self.preprocessor:
                        features = self.preprocessor(image_path)
                    else:
                        # Basic image loading
                        img = cv2.imread(image_path)
                        if img is None:
                            raise ValueError(f"Could not load image: {image_path}")
                        
                        # Convert and normalize
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (256, 256))
                        features = img.astype('float32') / 255.0
                        
                        if len(features.shape) == 3:
                            features = np.expand_dims(features, axis=0)
                    
                    # Make prediction
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(features)
                        prediction = np.argmax(proba, axis=1)[0]
                        confidence = np.max(proba)
                    else:
                        prediction = self.model.predict(features)
                        if len(prediction.shape) > 1:
                            prediction = np.argmax(prediction, axis=1)[0]
                            confidence = np.max(prediction)
                        else:
                            prediction = prediction[0]
                            confidence = 0.5  # Default confidence
                    
                    result = {
                        'prediction': 'steganographic' if prediction == 1 else 'clean',
                        'confidence': float(confidence),
                        'prediction_value': int(prediction),
                        'image_path': image_path,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Prediction failed: {str(e)}")
                    return {
                        'error': str(e),
                        'image_path': image_path,
                        'timestamp': datetime.now().isoformat()
                    }
            
            def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
                """Batch prediction"""
                return [self.predict_image(path) for path in image_paths]
        
        return PredictionService(model, preprocessing_pipeline)
    
    def export_model_for_serving(self, model, export_path: str, 
                                model_format: str = 'tensorflow_saved_model'):
        """Export model for production serving"""
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if model_format == 'tensorflow_saved_model' and TF_AVAILABLE:
            if isinstance(model, keras.Model):
                tf.saved_model.save(model, str(export_dir))
                self.logger.info(f"Model exported as TensorFlow SavedModel to {export_dir}")
            else:
                raise ValueError("Model is not a TensorFlow model")
                
        elif model_format == 'onnx':
            # ONNX export (would require additional dependencies)
            self.logger.warning("ONNX export not implemented")
            
        elif model_format == 'pickle':
            export_file = export_dir / "model.pkl"
            joblib.dump(model, export_file)
            self.logger.info(f"Model exported as pickle to {export_file}")
            
        else:
            raise ValueError(f"Unsupported export format: {model_format}")


class VisualizationUtils:
    """Utilities for model visualization and interpretation"""
    
    @staticmethod
    def plot_metrics_comparison(comparison_data: Dict[str, Dict], 
                              save_path: str = None):
        """Plot comparison of multiple models across metrics"""
        
        metrics = list(comparison_data.keys())
        models = list(next(iter(comparison_data.values())).keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:6]):  # Plot up to 6 metrics
            if i >= len(axes):
                break
                
            values = [comparison_data[metric].get(model, 0) for model in models]
            
            bars = axes[i].bar(models, values)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_roc_curves(roc_results: Dict[str, Dict], save_path: str = None):
        """Plot ROC curves for multiple models"""
        
        plt.figure(figsize=(10, 8))
        
        for model_name, roc_data in roc_results.items():
            if 'fpr' in roc_data and 'tpr' in roc_data:
                plt.plot(roc_data['fpr'], roc_data['tpr'], 
                        label=f"{model_name} (AUC = {roc_data['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrices(models_results: Dict[str, Dict], 
                              save_path: str = None):
        """Plot confusion matrices for multiple models"""
        
        n_models = len(models_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for i, (model_name, results) in enumerate(models_results.items()):
            if i >= len(axes):
                break
                
            cm = np.array(results['analysis']['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Clean', 'Steganographic'],
                       yticklabels=['Clean', 'Steganographic'])
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Remove empty subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test ModelManager
    print("Testing ModelManager...")
    manager = ModelManager("test_models")
    
    # Test ModelEvaluator
    print("\nTesting ModelEvaluator...")
    evaluator = ModelEvaluator()
    
    # Generate dummy data for testing
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.random(100)
    
    # Test metrics calculation
    metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred, y_proba)
    print("Sample metrics:", metrics)
    
    # Test ModelBenchmark
    print("\nTesting ModelBenchmark...")
    benchmark = ModelBenchmark()
    
    print("Model utilities testing completed!")