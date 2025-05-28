#!/usr/bin/env python3
"""
StegAnalysis Suite - CNN Model
Advanced Convolutional Neural Network architectures for steganography detection
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    raise ImportError("TensorFlow is required for CNN models. Install with: pip install tensorflow")

import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class StegCNN:
    """Advanced CNN architecture specifically designed for steganography detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.input_shape = config.get('input_shape', (256, 256, 3))
        self.num_classes = config.get('num_classes', 2)  # Clean vs Steganographic
        self.model_type = config.get('model_type', 'custom')  # custom, vgg16, resnet50, efficientnet
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.patience = config.get('patience', 15)
        
        # Augmentation parameters
        self.use_augmentation = config.get('use_augmentation', True)
        self.augmentation_factor = config.get('augmentation_factor', 0.2)
        
        # Model architecture
        self.model = None
        self.history = None
        
        # Callbacks
        self.callbacks_list = []
        
    def build_custom_cnn(self) -> keras.Model:
        """Build custom CNN architecture optimized for steganography detection"""
        
        model = models.Sequential([
            # First Convolutional Block - Focus on high-frequency patterns
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape,
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Block - Detect spatial relationships
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Block - Complex pattern recognition
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Block - Deep feature extraction
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Global pooling and classification
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_residual_cnn(self) -> keras.Model:
        """Build ResNet-inspired architecture for steganography detection"""
        
        def residual_block(x, filters, kernel_size=3, stride=1):
            """Residual block with skip connections"""
            shortcut = x
            
            # First conv layer
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                            kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Second conv layer
            x = layers.Conv2D(filters, kernel_size, padding='same',
                            kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(64, 7, strides=2, padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = layers.Dropout(0.25)(x)
        
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = layers.Dropout(0.25)(x)
        
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        x = layers.Dropout(0.3)(x)
        
        x = residual_block(x, 512, stride=2)
        x = residual_block(x, 512)
        x = layers.Dropout(0.3)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def build_attention_cnn(self) -> keras.Model:
        """Build CNN with attention mechanism for steganography detection"""
        
        def attention_block(x, filters):
            """Attention mechanism block"""
            # Channel attention
            gap = layers.GlobalAveragePooling2D()(x)
            channel_attention = layers.Dense(filters // 8, activation='relu')(gap)
            channel_attention = layers.Dense(filters, activation='sigmoid')(channel_attention)
            channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
            
            # Apply channel attention
            x = layers.Multiply()([x, channel_attention])
            
            # Spatial attention
            spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(x)
            x = layers.Multiply()([x, spatial_attention])
            
            return x
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Feature extraction with attention
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 64)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 128)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 256)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 512)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def build_transfer_learning_model(self, base_model_name: str = 'vgg16') -> keras.Model:
        """Build model using transfer learning from pre-trained networks"""
        
        # Select base model
        if base_model_name.lower() == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=self.input_shape)
        elif base_model_name.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                 input_shape=self.input_shape)
        elif base_model_name.lower() == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                       input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def build_model(self) -> keras.Model:
        """Build the specified CNN model"""
        
        self.logger.info(f"Building {self.model_type} CNN model...")
        
        if self.model_type == 'custom':
            self.model = self.build_custom_cnn()
        elif self.model_type == 'residual':
            self.model = self.build_residual_cnn()
        elif self.model_type == 'attention':
            self.model = self.build_attention_cnn()
        elif self.model_type in ['vgg16', 'resnet50', 'efficientnet']:
            self.model = self.build_transfer_learning_model(self.model_type)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def create_data_generator(self) -> ImageDataGenerator:
        """Create data augmentation generator"""
        
        if self.use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=self.augmentation_factor * 10,
                width_shift_range=self.augmentation_factor,
                height_shift_range=self.augmentation_factor,
                shear_range=self.augmentation_factor,
                zoom_range=self.augmentation_factor,
                horizontal_flip=True,
                fill_mode='nearest',
                rescale=1./255
            )
        else:
            datagen = ImageDataGenerator(rescale=1./255)
        
        return datagen
    
    def setup_callbacks(self, model_save_path: str) -> List[callbacks.Callback]:
        """Setup training callbacks"""
        
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                model_save_path.replace('.h5', '_training_log.csv'),
                append=True
            )
        ]
        
        # TensorBoard logging (optional)
        if self.config.get('use_tensorboard', False):
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=f"logs/{self.model_type}",
                    histogram_freq=1,
                    write_graph=True
                )
            )
        
        self.callbacks_list = callback_list
        return callback_list
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_save_path: str) -> Dict[str, Any]:
        """Train the CNN model"""
        
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting CNN training...")
        
        # Setup callbacks
        self.setup_callbacks(model_save_path)
        
        # Normalize data if not using data generator
        if not self.use_augmentation:
            X_train = X_train.astype('float32') / 255.0
            X_val = X_val.astype('float32') / 255.0
        
        # Train model
        if self.use_augmentation:
            # Create data generator
            train_datagen = self.create_data_generator()
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            # Fit generator
            train_generator = train_datagen.flow(X_train, y_train, 
                                               batch_size=self.batch_size)
            val_generator = val_datagen.flow(X_val, y_val,
                                           batch_size=self.batch_size)
            
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // self.batch_size,
                epochs=self.epochs,
                validation_data=val_generator,
                validation_steps=len(X_val) // self.batch_size,
                callbacks=self.callbacks_list,
                verbose=1
            )
        else:
            # Standard training
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=self.callbacks_list,
                verbose=1
            )
        
        # Training results
        training_results = {
            'final_accuracy': max(self.history.history['val_accuracy']),
            'final_loss': min(self.history.history['val_loss']),
            'epochs_trained': len(self.history.history['loss']),
            'best_epoch': np.argmax(self.history.history['val_accuracy']) + 1
        }
        
        self.logger.info(f"Training completed. Best validation accuracy: "
                        f"{training_results['final_accuracy']:.4f}")
        
        return training_results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Normalize test data
        X_test = X_test.astype('float32') / 255.0
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation_results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        return evaluation_results
    
    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  unfreeze_layers: int = 20) -> Dict[str, Any]:
        """Fine-tune pre-trained model by unfreezing top layers"""
        
        if self.model_type not in ['vgg16', 'resnet50', 'efficientnet']:
            self.logger.warning("Fine-tuning only available for transfer learning models")
            return {}
        
        self.logger.info(f"Fine-tuning model by unfreezing top {unfreeze_layers} layers...")
        
        # Unfreeze top layers
        base_model = self.model.layers[1]  # Get base model
        base_model.trainable = True
        
        # Freeze all layers except the top ones
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate * 0.1),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            X_train / 255.0, y_train,
            batch_size=self.batch_size // 2,  # Smaller batch size for fine-tuning
            epochs=self.epochs // 2,
            validation_data=(X_val / 255.0, y_val),
            callbacks=self.callbacks_list,
            verbose=1
        )
        
        return {
            'fine_tune_history': fine_tune_history.history,
            'unfrozen_layers': unfreeze_layers
        }
    
    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Predict steganography for a single image"""
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_shape[:2])
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            prediction_proba = self.model.predict(img)
            prediction = np.argmax(prediction_proba, axis=1)[0]
            confidence = np.max(prediction_proba)
            
            result = {
                'prediction': 'steganographic' if prediction == 1 else 'clean',
                'confidence': float(confidence),
                'probability_clean': float(prediction_proba[0][0]),
                'probability_steganographic': float(prediction_proba[0][1]),
                'image_path': image_path
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting image {image_path}: {str(e)}")
            raise
    
    def save_model(self, save_path: str, save_weights_only: bool = False):
        """Save the trained model"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        if save_weights_only:
            self.model.save_weights(save_path)
        else:
            self.model.save(save_path)
        
        # Save training history if available
        if self.history is not None:
            history_path = save_path.replace('.h5', '_history.json')
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {}
                for key, value in self.history.history.items():
                    if isinstance(value, list):
                        history_dict[key] = value
                    else:
                        history_dict[key] = [float(v) for v in value]
                json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        
        self.model = keras.models.load_model(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        
        # Load training history if available
        history_path = model_path.replace('.h5', '_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_dict = json.load(f)
                self.logger.info("Training history loaded")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        
        if self.history is None:
            self.logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        
        if self.model is None:
            return "No model built"
        
        # Capture model summary
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Configuration for testing
    config = {
        'input_shape': (256, 256, 3),
        'num_classes': 2,
        'model_type': 'custom',  # custom, residual, attention, vgg16, resnet50, efficientnet
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10,
        'use_augmentation': True,
        'augmentation_factor': 0.2,
        'use_tensorboard': False
    }
    
    # Initialize CNN
    cnn = StegCNN(config)
    
    # Build model
    model = cnn.build_model()
    
    # Print model summary
    print(cnn.get_model_summary())
    
    print(f"Model built successfully with {model.count_params():,} parameters")