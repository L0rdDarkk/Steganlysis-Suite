#!/usr/bin/env python3
"""
StegAnalysis Suite - Models Module
Advanced machine learning models for steganography detection
"""

import logging
from typing import Dict, Any, Optional

# Core model imports
from .cnn_model import StegCNN
from .svm_model import StegSVM
from .model_utils import (
    ModelManager, ModelEvaluator, ModelBenchmark, 
    ModelDeployment, VisualizationUtils
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "StegAnalysis Suite"
__description__ = "Machine learning models for steganography detection"

# Configure logging
logger = logging.getLogger(__name__)

# Model registry for easy access
MODEL_TYPES = {
    'cnn': StegCNN,
    'svm': StegSVM
}

# Default configurations
DEFAULT_CNN_CONFIG = {
    'input_shape': (256, 256, 3),
    'num_classes': 2,
    'model_type': 'custom',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'use_augmentation': True,
    'augmentation_factor': 0.2,
    'use_tensorboard': False
}

DEFAULT_SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True,
    'feature_selection': 'selectkbest',
    'n_features': 100,
    'scaling': 'standard',
    'use_pca': False,
    'cv_folds': 5,
    'hyperparameter_tuning': True,
    'random_state': 42
}


class ModelFactory:
    """Factory class for creating and managing models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._model_manager = None
    
    @property
    def model_manager(self) -> ModelManager:
        """Lazy initialization of model manager"""
        if self._model_manager is None:
            self._model_manager = ModelManager()
        return self._model_manager
    
    def create_model(self, model_type: str, config: Dict[str, Any] = None):
        """Create a model instance"""
        
        if model_type not in MODEL_TYPES:
            available_types = ', '.join(MODEL_TYPES.keys())
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {available_types}")
        
        # Use default config if none provided
        if config is None:
            if model_type == 'cnn':
                config = DEFAULT_CNN_CONFIG.copy()
            elif model_type == 'svm':
                config = DEFAULT_SVM_CONFIG.copy()
            else:
                config = {}
        
        # Create model instance
        model_class = MODEL_TYPES[model_type]
        model = model_class(config)
        
        self.logger.info(f"Created {model_type.upper()} model with config: {config}")
        return model
    
    def load_pretrained_model(self, model_name: str):
        """Load a pre-trained model from the model manager"""
        return self.model_manager.load_model(model_name)
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available pre-trained models"""
        return self.model_manager.list_models()


def create_cnn_model(config: Dict[str, Any] = None) -> StegCNN:
    """Convenience function to create CNN model"""
    factory = ModelFactory()
    return factory.create_model('cnn', config)


def create_svm_model(config: Dict[str, Any] = None) -> StegSVM:
    """Convenience function to create SVM model"""
    factory = ModelFactory()
    return factory.create_model('svm', config)


def create_model_evaluator(config: Dict[str, Any] = None) -> ModelEvaluator:
    """Convenience function to create model evaluator"""
    return ModelEvaluator(config)


def create_model_benchmark() -> ModelBenchmark:
    """Convenience function to create model benchmark"""
    return ModelBenchmark()


def setup_model_environment(models_dir: str = "models/trained") -> ModelManager:
    """Setup model environment and return manager"""
    manager = ModelManager(models_dir)
    logger.info(f"Model environment setup complete. Models directory: {models_dir}")
    return manager


# Quick start example configuration
QUICK_START_CONFIGS = {
    'lightweight_cnn': {
        'input_shape': (128, 128, 3),
        'model_type': 'custom',
        'learning_rate': 0.01,
        'batch_size': 64,
        'epochs': 50,
        'use_augmentation': False
    },
    'robust_cnn': {
        'input_shape': (256, 256, 3),
        'model_type': 'attention',
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 200,
        'use_augmentation': True,
        'augmentation_factor': 0.3
    },
    'fast_svm': {
        'kernel': 'linear',
        'feature_selection': 'selectkbest',
        'n_features': 50,
        'hyperparameter_tuning': False
    },
    'accurate_svm': {
        'kernel': 'rbf',
        'feature_selection': 'rfe',
        'n_features': 200,
        'hyperparameter_tuning': True,
        'cv_folds': 10
    }
}


def create_quick_start_model(preset: str, model_type: str = None):
    """Create a model using quick start presets"""
    
    if preset not in QUICK_START_CONFIGS:
        available_presets = ', '.join(QUICK_START_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Available presets: {available_presets}")
    
    config = QUICK_START_CONFIGS[preset].copy()
    
    # Auto-detect model type from preset name
    if model_type is None:
        if 'cnn' in preset:
            model_type = 'cnn'
        elif 'svm' in preset:
            model_type = 'svm'
        else:
            raise ValueError(f"Cannot auto-detect model type for preset: {preset}. "
                           f"Please specify model_type parameter.")
    
    factory = ModelFactory()
    return factory.create_model(model_type, config)


# Validation functions
def validate_model_config(model_type: str, config: Dict[str, Any]) -> bool:
    """Validate model configuration"""
    
    if model_type == 'cnn':
        required_keys = ['input_shape', 'num_classes']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required CNN config key: {key}")
                return False
    
    elif model_type == 'svm':
        if 'kernel' in config and config['kernel'] not in ['linear', 'poly', 'rbf', 'sigmoid']:
            logger.error(f"Invalid SVM kernel: {config['kernel']}")
            return False
    
    return True


def get_model_info() -> Dict[str, Any]:
    """Get information about available models and configurations"""
    
    info = {
        'version': __version__,
        'available_model_types': list(MODEL_TYPES.keys()),
        'default_configs': {
            'cnn': DEFAULT_CNN_CONFIG,
            'svm': DEFAULT_SVM_CONFIG
        },
        'quick_start_presets': list(QUICK_START_CONFIGS.keys()),
        'description': __description__
    }
    
    return info


# Export main classes and functions
__all__ = [
    # Core model classes
    'StegCNN',
    'StegSVM',
    
    # Utility classes
    'ModelManager',
    'ModelEvaluator',
    'ModelBenchmark',
    'ModelDeployment',
    'VisualizationUtils',
    
    # Factory and convenience functions
    'ModelFactory',
    'create_cnn_model',
    'create_svm_model',
    'create_model_evaluator',
    'create_model_benchmark',
    'setup_model_environment',
    'create_quick_start_model',
    
    # Configuration and validation
    'DEFAULT_CNN_CONFIG',
    'DEFAULT_SVM_CONFIG',
    'QUICK_START_CONFIGS',
    'validate_model_config',
    'get_model_info',
    
    # Constants
    'MODEL_TYPES',
    '__version__',
    '__author__',
    '__description__'
]


# Module initialization
def _initialize_module():
    """Initialize the models module"""
    
    logger.info(f"StegAnalysis Models Module v{__version__} initialized")
    
    # Check for optional dependencies
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow {tf.__version__} available for deep learning models")
    except ImportError:
        logger.warning("TensorFlow not available. CNN models will be disabled.")
    
    try:
        import sklearn
        logger.info(f"Scikit-learn {sklearn.__version__} available for traditional ML models")
    except ImportError:
        logger.error("Scikit-learn not available. SVM models will be disabled.")
    
    # Check for GPU availability
    try:
        import tensorflow as tf
        if tf.test.is_gpu_available():
            gpus = tf.config.experimental.list_physical_devices('GPU')
            logger.info(f"GPU acceleration available: {len(gpus)} GPU(s) detected")
        else:
            logger.info("GPU acceleration not available, using CPU")
    except:
        pass


# Initialize module on import
_initialize_module()


# Example usage documentation
def print_usage_examples():
    """Print usage examples for the models module"""
    
    examples = """
    StegAnalysis Models Module - Usage Examples
    ==========================================
    
    1. Create a basic CNN model:
    ```python
    from models import create_cnn_model
    
    cnn = create_cnn_model()
    model = cnn.build_model()
    ```
    
    2. Create an SVM model with custom config:
    ```python
    from models import create_svm_model
    
    config = {
        'kernel': 'rbf',
        'C': 10.0,
        'feature_selection': 'rfe',
        'n_features': 150
    }
    svm = create_svm_model(config)
    ```
    
    3. Use quick start presets:
    ```python
    from models import create_quick_start_model
    
    # Fast training CNN
    lightweight_cnn = create_quick_start_model('lightweight_cnn')
    
    # Accurate SVM
    accurate_svm = create_quick_start_model('accurate_svm')
    ```
    
    4. Model management:
    ```python
    from models import setup_model_environment
    
    manager = setup_model_environment()
    manager.register_model('my_cnn', 'path/to/model.h5', 'tensorflow')
    loaded_model = manager.load_model('my_cnn')
    ```
    
    5. Model evaluation:
    ```python
    from models import create_model_evaluator
    
    evaluator = create_model_evaluator()
    results = evaluator.evaluate_model(model, X_test, y_test)
    ```
    
    6. Model comparison:
    ```python
    from models import ModelEvaluator, VisualizationUtils
    
    evaluator = ModelEvaluator()
    
    # Evaluate multiple models
    cnn_results = evaluator.evaluate_model(cnn_model, X_test, y_test, 'CNN')
    svm_results = evaluator.evaluate_model(svm_model, X_test, y_test, 'SVM')
    
    # Compare results
    comparison = evaluator.compare_models({
        'CNN': cnn_results,
        'SVM': svm_results
    })
    
    # Visualize comparison
    VisualizationUtils.plot_metrics_comparison(comparison['comparison_metrics'])
    ```
    
    7. Model deployment:
    ```python
    from models import ModelDeployment
    
    deployment = ModelDeployment()
    service = deployment.create_prediction_service(model)
    
    # Single prediction
    result = service.predict_image('path/to/image.jpg')
    
    # Batch prediction
    results = service.predict_batch(['image1.jpg', 'image2.jpg'])
    ```
    
    8. Performance benchmarking:
    ```python
    from models import create_model_benchmark
    
    benchmark = create_model_benchmark()
    speed_results = benchmark.benchmark_inference_speed(model, X_test)
    memory_results = benchmark.benchmark_memory_usage(model, X_test)
    ```
    """
    
    print(examples)


# Configuration validation examples
def validate_configs():
    """Validate all default configurations"""
    
    logger.info("Validating default configurations...")
    
    # Validate CNN config
    cnn_valid = validate_model_config('cnn', DEFAULT_CNN_CONFIG)
    logger.info(f"CNN config validation: {'PASSED' if cnn_valid else 'FAILED'}")
    
    # Validate SVM config
    svm_valid = validate_model_config('svm', DEFAULT_SVM_CONFIG)
    logger.info(f"SVM config validation: {'PASSED' if svm_valid else 'FAILED'}")
    
    # Validate quick start configs
    for preset_name, config in QUICK_START_CONFIGS.items():
        model_type = 'cnn' if 'cnn' in preset_name else 'svm'
        valid = validate_model_config(model_type, config)
        logger.info(f"Quick start '{preset_name}' validation: {'PASSED' if valid else 'FAILED'}")
    
    return cnn_valid and svm_valid


if __name__ == "__main__":
    # Run validation and print examples when module is run directly
    print(f"StegAnalysis Models Module v{__version__}")
    print("=" * 50)
    
    # Validate configurations
    validate_configs()
    
    # Print usage examples
    print_usage_examples()
    
    # Print module info
    info = get_model_info()
    print("\nModule Information:")
    print("-" * 20)
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key in value.keys():
                print(f"  - {sub_key}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(map(str, value))}")
        else:
            print(f"{key}: {value}")