#!/usr/bin/env python3
"""
StegAnalysis Suite - Dataset Generator
Tool for generating steganographic test datasets
"""

import os
import json
import logging
import random
import string
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
import cv2

# Import steganography tools
try:
    from stegano import lsb
    from stegano.lsbset import generators
    STEGANO_AVAILABLE = True
except ImportError:
    STEGANO_AVAILABLE = False
    logging.warning("Stegano library not available. Some embedding methods will be disabled.")


class DatasetGenerator:
    """Generate steganographic datasets for testing and training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.input_dir = Path(config.get('input_dir', 'datasets/images/clean'))
        self.output_dir = Path(config.get('output_dir', 'datasets/images/steganographic'))
        self.metadata_dir = Path(config.get('metadata_dir', 'datasets/metadata'))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation settings
        self.embedding_methods = config.get('embedding_methods', ['lsb', 'dct', 'spatial'])
        self.payload_sizes = config.get('payload_sizes', [0.1, 0.25, 0.5, 1.0])  # Percentage of capacity
        self.message_types = config.get('message_types', ['text', 'binary', 'image'])
        
        # Quality settings
        self.jpeg_qualities = config.get('jpeg_qualities', [70, 80, 90, 95])
        self.compression_levels = config.get('compression_levels', [1, 3, 6, 9])
        
        # Dataset metadata
        self.dataset_info = {
            'created': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'total_images': 0,
            'embedding_methods': {},
            'statistics': {}
        }
        
    def generate_dataset(self, num_images: int = 1000, clean_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate a comprehensive steganographic dataset
        
        Args:
            num_images: Total number of images to generate
            clean_ratio: Ratio of clean (non-steganographic) images
            
        Returns:
            Dictionary with generation statistics
        """
        self.logger.info(f"Starting dataset generation: {num_images} images")
        
        # Get source images
        source_images = self._get_source_images()
        if not source_images:
            raise ValueError("No source images found in input directory")
        
        # Calculate split
        num_clean = int(num_images * clean_ratio)
        num_stego = num_images - num_clean
        
        stats = {
            'total_generated': 0,
            'clean_images': 0,
            'steganographic_images': 0,
            'embedding_methods': {},
            'errors': 0
        }
        
        # Generate clean images (copies/variations of originals)
        self.logger.info(f"Generating {num_clean} clean images...")
        for i in range(num_clean):
            try:
                source_img = random.choice(source_images)
                output_path = self.output_dir / f"clean_{i:06d}.png"
                
                if self._create_clean_variant(source_img, output_path):
                    stats['clean_images'] += 1
                    stats['total_generated'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error generating clean image {i}: {str(e)}")
                stats['errors'] += 1
        
        # Generate steganographic images
        self.logger.info(f"Generating {num_stego} steganographic images...")
        for i in range(num_stego):
            try:
                source_img = random.choice(source_images)
                method = random.choice(self.embedding_methods)
                output_path = self.output_dir / f"stego_{method}_{i:06d}.png"
                
                if self._create_steganographic_image(source_img, output_path, method):
                    stats['steganographic_images'] += 1
                    stats['total_generated'] += 1
                    
                    # Track method usage
                    if method not in stats['embedding_methods']:
                        stats['embedding_methods'][method] = 0
                    stats['embedding_methods'][method] += 1
                    
            except Exception as e:
                self.logger.error(f"Error generating steganographic image {i}: {str(e)}")
                stats['errors'] += 1
        
        # Update dataset info
        self.dataset_info['total_images'] = stats['total_generated']
        self.dataset_info['statistics'] = stats
        
        # Save metadata
        self._save_dataset_metadata()
        
        self.logger.info(f"Dataset generation completed. Generated {stats['total_generated']} images")
        return stats
    
    def _get_source_images(self) -> List[Path]:
        """Get list of source images for processing"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        source_images = []
        
        for ext in extensions:
            source_images.extend(self.input_dir.glob(f"*{ext}"))
            source_images.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(source_images)} source images")
        return source_images
    
    def _create_clean_variant(self, source_path: Path, output_path: Path) -> bool:
        """Create a clean (non-steganographic) image variant"""
        try:
            img = Image.open(source_path)
            
            # Apply random transformations to create variants
            transformations = [
                self._apply_compression,
                self._apply_rotation,
                self._apply_noise,
                self._apply_brightness,
                self._apply_contrast
            ]
            
            # Apply 1-2 random transformations
            num_transforms = random.randint(1, 2)
            selected_transforms = random.sample(transformations, num_transforms)
            
            for transform in selected_transforms:
                img = transform(img)
            
            # Save image
            if output_path.suffix.lower() == '.jpg':
                img.save(output_path, 'JPEG', quality=random.choice(self.jpeg_qualities))
            else:
                img.save(output_path, 'PNG')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating clean variant: {str(e)}")
            return False
    
    def _create_steganographic_image(self, source_path: Path, output_path: Path, method: str) -> bool:
        """Create a steganographic image using specified method"""
        try:
            # Generate payload
            payload = self._generate_payload()
            
            # Apply steganographic embedding
            if method == 'lsb':
                success = self._embed_lsb(source_path, output_path, payload)
            elif method == 'dct':
                success = self._embed_dct(source_path, output_path, payload)
            elif method == 'spatial':
                success = self._embed_spatial(source_path, output_path, payload)
            else:
                self.logger.warning(f"Unknown embedding method: {method}")
                return False
            
            if success:
                # Save embedding metadata
                self._save_embedding_metadata(output_path, method, payload)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error creating steganographic image: {str(e)}")
            return False
    
    def _generate_payload(self) -> Dict[str, Any]:
        """Generate random payload for embedding"""
        message_type = random.choice(self.message_types)
        payload_size = random.choice(self.payload_sizes)
        
        if message_type == 'text':
            # Generate random text
            length = random.randint(50, 500)
            message = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
        elif message_type == 'binary':
            # Generate random binary data
            length = random.randint(100, 1000)
            message = bytes(random.getrandbits(8) for _ in range(length))
        else:  # image
            # Generate small random image data
            size = random.randint(32, 128)
            message = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8).tobytes()
        
        return {
            'type': message_type,
            'content': message,
            'size_ratio': payload_size,
            'length': len(message) if isinstance(message, (str, bytes)) else len(message)
        }
    
    def _embed_lsb(self, source_path: Path, output_path: Path, payload: Dict[str, Any]) -> bool:
        """Embed data using LSB steganography"""
        try:
            if not STEGANO_AVAILABLE:
                return self._embed_lsb_manual(source_path, output_path, payload)
            
            # Use stegano library for LSB embedding
            message = payload['content']
            if isinstance(message, bytes):
                message = message.hex()  # Convert binary to hex string
            
            secret_image = lsb.hide(str(source_path), str(message))
            secret_image.save(str(output_path))
            
            return True
            
        except Exception as e:
            self.logger.error(f"LSB embedding failed: {str(e)}")
            return False
    
    def _embed_lsb_manual(self, source_path: Path, output_path: Path, payload: Dict[str, Any]) -> bool:
        """Manual LSB embedding implementation"""
        try:
            # Load image
            img = cv2.imread(str(source_path))
            if img is None:
                return False
            
            # Convert message to binary
            message = payload['content']
            if isinstance(message, str):
                binary_message = ''.join(format(ord(char), '08b') for char in message)
            else:
                binary_message = ''.join(format(byte, '08b') for byte in message)
            
            # Add delimiter
            binary_message += '1111111111111110'  # Delimiter to mark end
            
            # Embed in LSBs
            data_index = 0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(3):  # RGB channels
                        if data_index < len(binary_message):
                            # Modify LSB
                            img[i, j, k] = (img[i, j, k] & 0xFE) | int(binary_message[data_index])
                            data_index += 1
                        else:
                            break
                    if data_index >= len(binary_message):
                        break
                if data_index >= len(binary_message):
                    break
            
            # Save image
            cv2.imwrite(str(output_path), img)
            return True
            
        except Exception as e:
            self.logger.error(f"Manual LSB embedding failed: {str(e)}")
            return False
    
    def _embed_dct(self, source_path: Path, output_path: Path, payload: Dict[str, Any]) -> bool:
        """Embed data using DCT-based steganography"""
        try:
            # Load image
            img = cv2.imread(str(source_path))
            if img is None:
                return False
            
            # Convert to YUV color space
            yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel = yuv_img[:, :, 0].astype(np.float32)
            
            # Convert message to binary
            message = payload['content']
            if isinstance(message, str):
                binary_message = ''.join(format(ord(char), '08b') for char in message)
            else:
                binary_message = ''.join(format(byte, '08b') for byte in message)
            
            # Add delimiter
            binary_message += '1111111111111110'
            
            # Embed in DCT coefficients
            h, w = y_channel.shape
            data_index = 0
            
            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    if data_index >= len(binary_message):
                        break
                    
                    # Extract 8x8 block
                    block = y_channel[i:i+8, j:j+8]
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Modify middle frequency coefficient
                    if data_index < len(binary_message):
                        bit = int(binary_message[data_index])
                        if bit == 1:
                            dct_block[4, 4] = abs(dct_block[4, 4]) + 10
                        else:
                            dct_block[4, 4] = abs(dct_block[4, 4]) - 10
                        data_index += 1
                    
                    # Apply inverse DCT
                    idct_block = cv2.idct(dct_block)
                    y_channel[i:i+8, j:j+8] = idct_block
                
                if data_index >= len(binary_message):
                    break
            
            # Reconstruct image
            yuv_img[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
            result_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
            
            # Save image
            cv2.imwrite(str(output_path), result_img)
            return True
            
        except Exception as e:
            self.logger.error(f"DCT embedding failed: {str(e)}")
            return False
    
    def _embed_spatial(self, source_path: Path, output_path: Path, payload: Dict[str, Any]) -> bool:
        """Embed data using spatial domain techniques"""
        try:
            # Load image
            img = cv2.imread(str(source_path))
            if img is None:
                return False
            
            # Convert message to binary
            message = payload['content']
            if isinstance(message, str):
                binary_message = ''.join(format(ord(char), '08b') for char in message)
            else:
                binary_message = ''.join(format(byte, '08b') for byte in message)
            
            # Add delimiter
            binary_message += '1111111111111110'
            
            # Embed using +/- 1 modification in pixel values
            h, w, c = img.shape
            data_index = 0
            
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if data_index < len(binary_message):
                            bit = int(binary_message[data_index])
                            pixel_val = int(img[i, j, k])
                            
                            if bit == 1:
                                # Increase pixel value by 1 (if possible)
                                if pixel_val < 255:
                                    img[i, j, k] = pixel_val + 1
                            else:
                                # Decrease pixel value by 1 (if possible)
                                if pixel_val > 0:
                                    img[i, j, k] = pixel_val - 1
                            
                            data_index += 1
                        else:
                            break
                    if data_index >= len(binary_message):
                        break
                if data_index >= len(binary_message):
                    break
            
            # Save image
            cv2.imwrite(str(output_path), img)
            return True
            
        except Exception as e:
            self.logger.error(f"Spatial embedding failed: {str(e)}")
            return False
    
    def _save_embedding_metadata(self, image_path: Path, method: str, payload: Dict[str, Any]):
        """Save metadata about the embedded content"""
        metadata = {
            'image_path': str(image_path),
            'embedding_method': method,
            'payload_type': payload['type'],
            'payload_size': payload['length'],
            'size_ratio': payload['size_ratio'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to metadata file
        metadata_file = self.metadata_dir / f"{image_path.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_dataset_metadata(self):
        """Save overall dataset metadata"""
        metadata_file = self.metadata_dir / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    # Image transformation methods
    def _apply_compression(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression"""
        import io
        quality = random.choice(self.jpeg_qualities)
        
        # Save to bytes with compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load back
        return Image.open(buffer)
    
    def _apply_rotation(self, img: Image.Image) -> Image.Image:
        """Apply small rotation"""
        angle = random.uniform(-2, 2)  # Small rotation
        return img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    
    def _apply_noise(self, img: Image.Image) -> Image.Image:
        """Add random noise"""
        np_img = np.array(img)
        noise = np.random.normal(0, 5, np_img.shape).astype(np.int16)
        noisy_img = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def _apply_brightness(self, img: Image.Image) -> Image.Image:
        """Adjust brightness"""
        from PIL import ImageEnhance
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def _apply_contrast(self, img: Image.Image) -> Image.Image:
        """Adjust contrast"""
        from PIL import ImageEnhance
        factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def generate_labels_file(self, output_path: str = None) -> str:
        """Generate labels file for machine learning training"""
        if output_path is None:
            output_path = str(self.metadata_dir / "labels.json")
        
        labels = {}
        
        # Scan output directory for images
        for img_path in self.output_dir.glob("*.png"):
            if img_path.name.startswith("clean_"):
                labels[str(img_path)] = {
                    'label': 0,  # Clean image
                    'type': 'clean',
                    'method': None
                }
            elif img_path.name.startswith("stego_"):
                # Extract method from filename
                parts = img_path.name.split("_")
                method = parts[1] if len(parts) > 1 else "unknown"
                
                labels[str(img_path)] = {
                    'label': 1,  # Steganographic image
                    'type': 'steganographic',
                    'method': method
                }
        
        # Save labels
        with open(output_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        self.logger.info(f"Labels file saved: {output_path}")
        return output_path


def main():
    """Command-line interface for dataset generation"""
    parser = argparse.ArgumentParser(description="Generate steganographic datasets")
    parser.add_argument('--input-dir', default='datasets/images/clean',
                       help='Input directory with clean images')
    parser.add_argument('--output-dir', default='datasets/images/steganographic',
                       help='Output directory for generated images')
    parser.add_argument('--num-images', type=int, default=1000,
                       help='Total number of images to generate')
    parser.add_argument('--clean-ratio', type=float, default=0.3,
                       help='Ratio of clean images (0.0-1.0)')
    parser.add_argument('--methods', nargs='+', default=['lsb', 'dct', 'spatial'],
                       help='Embedding methods to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configuration
    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'metadata_dir': 'datasets/metadata',
        'embedding_methods': args.methods,
        'payload_sizes': [0.1, 0.25, 0.5, 1.0],
        'message_types': ['text', 'binary'],
        'jpeg_qualities': [70, 80, 90, 95]
    }
    
    # Generate dataset
    generator = DatasetGenerator(config)
    stats = generator.generate_dataset(args.num_images, args.clean_ratio)
    
    # Generate labels file
    generator.generate_labels_file()
    
    print(f"\nDataset Generation Complete!")
    print(f"Total images generated: {stats['total_generated']}")
    print(f"Clean images: {stats['clean_images']}")
    print(f"Steganographic images: {stats['steganographic_images']}")
    print(f"Embedding methods used: {stats['embedding_methods']}")
    print(f"Errors: {stats['errors']}")


if __name__ == "__main__":
    main()