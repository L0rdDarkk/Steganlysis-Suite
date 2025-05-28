#!/usr/bin/env python3
"""
StegAnalysis Suite - Data Extraction Module
Extract and recover hidden data from steganographic images
"""

import os
import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import struct
import binascii
from PIL import Image
import io
import base64

try:
    from scipy.fftpack import dct, idct
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. DCT-based extraction will be limited.")


class DataExtractor:
    """Data extraction and recovery engine for steganographic content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.extraction_config = config.get('extraction', {})
        
        # Configuration parameters
        self.methods = self.extraction_config.get('methods', ['lsb_sequential', 'lsb_random', 'dct_coefficients'])
        self.output_formats = self.extraction_config.get('output_formats', ['raw_binary', 'text', 'auto_detect'])
        self.max_payload_size = self.extraction_config.get('max_payload_size', 1048576)  # 1MB
        self.recovery_attempts = self.extraction_config.get('recovery_attempts', 3)
        
    def extract_all_methods(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Attempt extraction using all available methods
        
        Args:
            image_path: Path to the steganographic image
            output_dir: Directory to save extracted data (optional)
            
        Returns:
            Dictionary containing extraction results from all methods
        """
        try:
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path(image_path).parent / "extracted_data"
                output_path.mkdir(exist_ok=True)
            
            results = {
                'image_path': str(image_path),
                'extraction_timestamp': self._get_timestamp(),
                'output_directory': str(output_path),
                'extraction_results': {},
                'successful_extractions': [],
                'failed_extractions': [],
                'summary': {}
            }
            
            # Try each extraction method
            for method in self.methods:
                self.logger.info(f"Attempting extraction with method: {method}")
                
                try:
                    method_result = self._extract_by_method(image_path, method, output_path)
                    results['extraction_results'][method] = method_result
                    
                    if method_result.get('success', False):
                        results['successful_extractions'].append(method)
                        self.logger.info(f"Successful extraction with {method}")
                    else:
                        results['failed_extractions'].append(method)
                        
                except Exception as e:
                    self.logger.error(f"Extraction method {method} failed: {str(e)}")
                    results['extraction_results'][method] = {
                        'success': False,
                        'error': str(e),
                        'method': method
                    }
                    results['failed_extractions'].append(method)
            
            # Generate summary
            results['summary'] = self._generate_extraction_summary(results)
            
            self.logger.info(f"Extraction completed. Success: {len(results['successful_extractions'])}, Failed: {len(results['failed_extractions'])}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {str(e)}")
            return {
                'error': str(e),
                'image_path': str(image_path),
                'extraction_timestamp': self._get_timestamp()
            }
    
    def _extract_by_method(self, image_path: str, method: str, output_path: Path) -> Dict[str, Any]:
        """Extract data using specific method"""
        
        if method == 'lsb_sequential':
            return self._extract_lsb_sequential(image_path, output_path)
        elif method == 'lsb_random':
            return self._extract_lsb_random(image_path, output_path)
        elif method == 'dct_coefficients':
            return self._extract_dct_coefficients(image_path, output_path)
        elif method == 'palette_based':
            return self._extract_palette_based(image_path, output_path)
        else:
            return {
                'success': False,
                'error': f'Unknown extraction method: {method}',
                'method': method
            }
    
    def _extract_lsb_sequential(self, image_path: str, output_path: Path) -> Dict[str, Any]:
        """Extract data using sequential LSB method"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width, channels = image.shape
            
            # Extract LSBs sequentially
            extracted_bits = []
            total_pixels = height * width * channels
            
            for channel in range(channels):
                for row in range(height):
                    for col in range(width):
                        pixel_value = image[row, col, channel]
                        lsb = pixel_value & 1
                        extracted_bits.append(str(lsb))
            
            # Convert bits to bytes
            bit_string = ''.join(extracted_bits)
            extracted_data = self._bits_to_bytes(bit_string)
            
            # Try to identify data format and extract meaningful content
            analysis_result = self._analyze_extracted_data(extracted_data)
            
            # Save raw data
            raw_file = output_path / f"lsb_sequential_raw.bin"
            with open(raw_file, 'wb') as f:
                f.write(extracted_data)
            
            result = {
                'success': True,
                'method': 'lsb_sequential',
                'extracted_size': len(extracted_data),
                'output_file': str(raw_file),
                'data_analysis': analysis_result,
                'extraction_info': {
                    'total_bits_extracted': len(extracted_bits),
                    'image_dimensions': [height, width, channels],
                    'theoretical_capacity': total_pixels // 8  # bytes
                }
            }
            
            # Try to save in different formats based on analysis
            additional_files = self._save_in_detected_format(extracted_data, analysis_result, output_path, 'lsb_sequential')
            result['additional_outputs'] = additional_files
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'lsb_sequential'
            }
    
    def _extract_lsb_random(self, image_path: str, output_path: Path) -> Dict[str, Any]:
        """Extract data using random/key-based LSB method"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width, channels = image.shape
            
            # Try common keys/seeds for randomized extraction
            common_keys = [12345, 54321, 1234567890, 999999999, 0, 1, 42, 2023, 2024]
            best_result = None
            best_score = 0
            
            for key in common_keys:
                try:
                    # Generate pseudo-random sequence
                    np.random.seed(key)
                    total_pixels = height * width * channels
                    
                    # Create random indices
                    indices = list(range(total_pixels))
                    np.random.shuffle(indices)
                    
                    # Extract bits using random order
                    extracted_bits = []
                    flat_image = image.flatten()
                    
                    for idx in indices[:min(total_pixels, self.max_payload_size * 8)]:
                        pixel_value = flat_image[idx]
                        lsb = pixel_value & 1
                        extracted_bits.append(str(lsb))
                    
                    # Convert to bytes
                    bit_string = ''.join(extracted_bits)
                    extracted_data = self._bits_to_bytes(bit_string)
                    
                    # Analyze quality of extracted data
                    analysis_result = self._analyze_extracted_data(extracted_data)
                    quality_score = analysis_result.get('quality_score', 0)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_result = {
                            'key': key,
                            'data': extracted_data,
                            'analysis': analysis_result,
                            'bits_extracted': len(extracted_bits)
                        }
                    
                except Exception as e:
                    self.logger.warning(f"Random extraction with key {key} failed: {str(e)}")
                    continue
            
            if best_result:
                # Save best result
                raw_file = output_path / f"lsb_random_key_{best_result['key']}.bin"
                with open(raw_file, 'wb') as f:
                    f.write(best_result['data'])
                
                result = {
                    'success': True,
                    'method': 'lsb_random',
                    'best_key': best_result['key'],
                    'extracted_size': len(best_result['data']),
                    'output_file': str(raw_file),
                    'data_analysis': best_result['analysis'],
                    'quality_score': best_score,
                    'extraction_info': {
                        'keys_tested': len(common_keys),
                        'bits_extracted': best_result['bits_extracted']
                    }
                }
                
                # Save in detected format
                additional_files = self._save_in_detected_format(
                    best_result['data'], 
                    best_result['analysis'], 
                    output_path, 
                    f'lsb_random_key_{best_result["key"]}'
                )
                result['additional_outputs'] = additional_files
                
                return result
            else:
                return {
                    'success': False,
                    'error': 'No meaningful data found with any tested key',
                    'method': 'lsb_random',
                    'keys_tested': len(common_keys)
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'lsb_random'
            }
    
    def _extract_dct_coefficients(self, image_path: str, output_path: Path) -> Dict[str, Any]:
        """Extract data from DCT coefficients (JPEG steganography)"""
        if not SCIPY_AVAILABLE:
            return {
                'success': False,
                'error': 'SciPy not available for DCT extraction',
                'method': 'dct_coefficients'
            }
        
        try:
            # Load image as grayscale for DCT processing
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape
            
            # Process 8x8 blocks for DCT
            extracted_bits = []
            
            for i in range(0, height - 7, 8):
                for j in range(0, width - 7, 8):
                    # Extract 8x8 block
                    block = image[i:i+8, j:j+8].astype(np.float32)
                    
                    # Apply DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Extract from specific DCT coefficients (commonly used positions)
                    # Avoid DC coefficient (0,0) and use mid-frequency coefficients
                    extraction_positions = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
                    
                    for pos_y, pos_x in extraction_positions:
                        if pos_y < 8 and pos_x < 8:
                            coeff = dct_block[pos_y, pos_x]
                            # Extract LSB of coefficient
                            coeff_int = int(abs(coeff))
                            bit = coeff_int & 1
                            extracted_bits.append(str(bit))
            
            # Convert bits to bytes
            bit_string = ''.join(extracted_bits)
            extracted_data = self._bits_to_bytes(bit_string)
            
            # Analyze extracted data
            analysis_result = self._analyze_extracted_data(extracted_data)
            
            # Save raw data
            raw_file = output_path / "dct_coefficients_raw.bin"
            with open(raw_file, 'wb') as f:
                f.write(extracted_data)
            
            result = {
                'success': True,
                'method': 'dct_coefficients',
                'extracted_size': len(extracted_data),
                'output_file': str(raw_file),
                'data_analysis': analysis_result,
                'extraction_info': {
                    'blocks_processed': ((height // 8) * (width // 8)),
                    'bits_per_block': len(extraction_positions),
                    'total_bits_extracted': len(extracted_bits)
                }
            }
            
            # Save in detected format
            additional_files = self._save_in_detected_format(extracted_data, analysis_result, output_path, 'dct_coefficients')
            result['additional_outputs'] = additional_files
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'dct_coefficients'
            }
    
    def _extract_palette_based(self, image_path: str, output_path: Path) -> Dict[str, Any]:
        """Extract data from palette-based steganography (GIF, PNG with palette)"""
        try:
            # Load image using PIL to access palette
            with Image.open(image_path) as img:
                if img.mode not in ['P', 'L']:
                    return {
                        'success': False,
                        'error': 'Image does not have a palette (not suitable for palette-based extraction)',
                        'method': 'palette_based'
                    }
                
                # Get palette
                if img.mode == 'P':
                    palette = img.getpalette()
                    if not palette:
                        return {
                            'success': False,
                            'error': 'No palette found in image',
                            'method': 'palette_based'
                        }
                    
                    # Extract LSBs from palette entries
                    extracted_bits = []
                    
                    # Palette is typically RGB triplets
                    for i in range(0, len(palette), 3):
                        if i + 2 < len(palette):
                            r, g, b = palette[i], palette[i+1], palette[i+2]
                            # Extract LSB from each color component
                            extracted_bits.append(str(r & 1))
                            extracted_bits.append(str(g & 1))
                            extracted_bits.append(str(b & 1))
                    
                    # Convert bits to bytes
                    bit_string = ''.join(extracted_bits)
                    extracted_data = self._bits_to_bytes(bit_string)
                    
                    # Analyze extracted data
                    analysis_result = self._analyze_extracted_data(extracted_data)
                    
                    # Save raw data
                    raw_file = output_path / "palette_based_raw.bin"
                    with open(raw_file, 'wb') as f:
                        f.write(extracted_data)
                    
                    result = {
                        'success': True,
                        'method': 'palette_based',
                        'extracted_size': len(extracted_data),
                        'output_file': str(raw_file),
                        'data_analysis': analysis_result,
                        'extraction_info': {
                            'palette_entries': len(palette) // 3,
                            'bits_extracted': len(extracted_bits)
                        }
                    }
                    
                    # Save in detected format
                    additional_files = self._save_in_detected_format(extracted_data, analysis_result, output_path, 'palette_based')
                    result['additional_outputs'] = additional_files
                    
                    return result
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'palette_based'
            }
    
    def _bits_to_bytes(self, bit_string: str) -> bytes:
        """Convert bit string to bytes"""
        # Pad to multiple of 8
        while len(bit_string) % 8 != 0:
            bit_string = bit_string[:-1]  # Remove last bit if padding needed
        
        bytes_data = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_bits = bit_string[i:i+8]
            if len(byte_bits) == 8:
                byte_value = int(byte_bits, 2)
                bytes_data.append(byte_value)
        
        return bytes(bytes_data)
    
    def _analyze_extracted_data(self, data: bytes) -> Dict[str, Any]:
        """Analyze extracted data to determine format and quality"""
        analysis = {
            'size': len(data),
            'format_detected': 'unknown',
            'quality_score': 0.0,
            'entropy': 0.0,
            'printable_ratio': 0.0,
            'null_ratio': 0.0,
            'file_signatures': [],
            'text_preview': '',
            'recommendations': []
        }
        
        if len(data) == 0:
            return analysis
        
        # Calculate entropy
        if len(data) > 0:
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            entropy = 0.0
            for count in byte_counts:
                if count > 0:
                    prob = count / len(data)
                    entropy -= prob * np.log2(prob)
            
            analysis['entropy'] = entropy
        
        # Calculate printable character ratio
        printable_count = sum(1 for byte in data if 32 <= byte <= 126)
        analysis['printable_ratio'] = printable_count / len(data)
        
        # Calculate null byte ratio
        null_count = sum(1 for byte in data if byte == 0)
        analysis['null_ratio'] = null_count / len(data)
        
        # Check for file signatures
        analysis['file_signatures'] = self._detect_file_signatures(data)
        
        # Try to decode as text
        analysis['text_preview'] = self._extract_text_preview(data)
        
        # Calculate quality score
        quality_score = 0.0
        
        # High entropy suggests meaningful data
        if entropy > 7.0:
            quality_score += 0.3
        elif entropy > 5.0:
            quality_score += 0.2
        
        # Reasonable printable ratio for text
        if 0.7 <= analysis['printable_ratio'] <= 1.0:
            quality_score += 0.3
        elif 0.3 <= analysis['printable_ratio'] <= 0.7:
            quality_score += 0.1
        
        # Low null ratio is generally good
        if analysis['null_ratio'] < 0.1:
            quality_score += 0.2
        
        # File signatures are very good indicators
        if analysis['file_signatures']:
            quality_score += 0.4
        
        # Meaningful text content
        if len(analysis['text_preview']) > 10:
            quality_score += 0.2
        
        analysis['quality_score'] = min(quality_score, 1.0)
        
        # Generate recommendations
        if analysis['quality_score'] > 0.7:
            analysis['recommendations'].append('High quality extraction - likely contains meaningful data')
        elif analysis['quality_score'] > 0.4:
            analysis['recommendations'].append('Moderate quality extraction - may contain partial data')
        else:
            analysis['recommendations'].append('Low quality extraction - likely noise or no hidden data')
        
        # Format-specific recommendations
        if analysis['file_signatures']:
            analysis['format_detected'] = analysis['file_signatures'][0]['type']
            analysis['recommendations'].append(f'File signature detected: {analysis["format_detected"]}')
        elif analysis['printable_ratio'] > 0.8:
            analysis['format_detected'] = 'text'
            analysis['recommendations'].append('High printable character ratio - likely text data')
        
        return analysis
    
    def _detect_file_signatures(self, data: bytes) -> List[Dict[str, Any]]:
        """Detect file format signatures in extracted data"""
        signatures = []
        
        # Common file signatures
        file_sigs = {
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'GIF87a': 'GIF87a',
            b'GIF89a': 'GIF89a',
            b'BM': 'BMP',
            b'PK\x03\x04': 'ZIP',
            b'Rar!\x1a\x07\x00': 'RAR',
            b'%PDF': 'PDF',
            b'MZ': 'Executable',
            b'\x7fELF': 'ELF',
            b'RIFF': 'RIFF'
        }
        
        # Check beginning of data
        for sig, file_type in file_sigs.items():
            if data.startswith(sig):
                signatures.append({
                    'type': file_type,
                    'offset': 0,
                    'signature': sig.hex()
                })
        
        # Check for signatures at other positions (in case of headers/padding)
        for offset in range(min(100, len(data) - 8)):
            for sig, file_type in file_sigs.items():
                if data[offset:].startswith(sig):
                    signatures.append({
                        'type': file_type,
                        'offset': offset,
                        'signature': sig.hex()
                    })
        
        return signatures
    
    def _extract_text_preview(self, data: bytes) -> str:
        """Extract text preview from data"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'ascii', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = data.decode(encoding, errors='ignore')
                    # Filter for printable characters
                    printable_text = ''.join(c for c in text if c.isprintable())
                    
                    if len(printable_text) > 10:
                        return printable_text[:200]  # First 200 chars
                except:
                    continue
            
            # If no encoding works well, return hex representation of first bytes
            return binascii.hexlify(data[:50]).decode('ascii')
            
        except:
            return 'Unable to extract text preview'
    
    def _save_in_detected_format(self, data: bytes, analysis: Dict[str, Any], output_path: Path, method_name: str) -> List[str]:
        """Save extracted data in appropriate format based on analysis"""
        additional_files = []
        
        try:
            format_detected = analysis.get('format_detected', 'unknown')
            
            # Save as detected format
            if format_detected in ['JPEG', 'PNG', 'GIF87a', 'GIF89a', 'BMP']:
                img_file = output_path / f"{method_name}_extracted.{format_detected.lower()}"
                with open(img_file, 'wb') as f:
                    f.write(data)
                additional_files.append(str(img_file))
            
            elif format_detected == 'text':
                text_file = output_path / f"{method_name}_extracted.txt"
                with open(text_file, 'wb') as f:
                    f.write(data)
                additional_files.append(str(text_file))
            
            elif format_detected in ['ZIP', 'RAR']:
                archive_ext = 'zip' if format_detected == 'ZIP' else 'rar'
                archive_file = output_path / f"{method_name}_extracted.{archive_ext}"
                with open(archive_file, 'wb') as f:
                    f.write(data)
                additional_files.append(str(archive_file))
            
            elif format_detected == 'PDF':
                pdf_file = output_path / f"{method_name}_extracted.pdf"
                with open(pdf_file, 'wb') as f:
                    f.write(data)
                additional_files.append(str(pdf_file))
            
            # Always save as hex dump for analysis
            hex_file = output_path / f"{method_name}_hexdump.txt"
            with open(hex_file, 'w') as f:
                hex_data = binascii.hexlify(data).decode('ascii')
                # Format as hex dump
                for i in range(0, len(hex_data), 32):
                    line = hex_data[i:i+32]
                    # Add spaces every 2 characters
                    formatted_line = ' '.join(line[j:j+2] for j in range(0, len(line), 2))
                    f.write(f"{i//2:08x}: {formatted_line}\n")
            additional_files.append(str(hex_file))
            
            # Save readable text if high printable ratio
            if analysis.get('printable_ratio', 0) > 0.7:
                text_file = output_path / f"{method_name}_text.txt"
                try:
                    text_content = data.decode('utf-8', errors='ignore')
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    additional_files.append(str(text_file))
                except:
                    pass
            
        except Exception as e:
            self.logger.warning(f"Failed to save in detected format: {str(e)}")
        
        return additional_files
    
    def _generate_extraction_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of extraction results"""
        successful = results.get('successful_extractions', [])
        failed = results.get('failed_extractions', [])
        extraction_results = results.get('extraction_results', {})
        
        summary = {
            'total_methods_attempted': len(self.methods),
            'successful_extractions': len(successful),
            'failed_extractions': len(failed),
            'success_rate': len(successful) / len(self.methods) if self.methods else 0,
            'best_extraction': None,
            'recommended_method': None,
            'total_data_extracted': 0,
            'quality_summary': {}
        }
        
        # Find best extraction based on quality score
        best_quality = 0
        best_method = None
        
        for method in successful:
            method_result = extraction_results.get(method, {})
            analysis = method_result.get('data_analysis', {})
            quality = analysis.get('quality_score', 0)
            
            if quality > best_quality:
                best_quality = quality
                best_method = method
                summary['best_extraction'] = {
                    'method': method,
                    'quality_score': quality,
                    'extracted_size': method_result.get('extracted_size', 0),
                    'format_detected': analysis.get('format_detected', 'unknown')
                }
        
        summary['recommended_method'] = best_method
        
        # Calculate total data extracted
        for method_result in extraction_results.values():
            if method_result.get('success', False):
                summary['total_data_extracted'] += method_result.get('extracted_size', 0)
        
        # Quality summary
        if successful:
            qualities = []
            for method in successful:
                method_result = extraction_results.get(method, {})
                analysis = method_result.get('data_analysis', {})
                qualities.append(analysis.get('quality_score', 0))
            
            summary['quality_summary'] = {
                'average_quality': sum(qualities) / len(qualities),
                'max_quality': max(qualities),
                'min_quality': min(qualities)
            }
        
        return summary
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def extract_specific_method(self, image_path: str, method: str, output_dir: str = None, **kwargs) -> Dict[str, Any]:
        """Extract data using a specific method with custom parameters"""
        try:
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path(image_path).parent / "extracted_data"
                output_path.mkdir(exist_ok=True)
            
            # Extract using specific method
            result = self._extract_by_method(image_path, method, output_path)
            
            # Add metadata
            result['extraction_timestamp'] = self._get_timestamp()
            result['image_path'] = str(image_path)
            result['output_directory'] = str(output_path)
            result['parameters'] = kwargs
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': method,
                'extraction_timestamp': self._get_timestamp()
            }
    
    def get_extraction_capabilities(self) -> Dict[str, Any]:
        """Get information about extraction capabilities"""
        return {
            'available_methods': self.methods,
            'supported_formats': ['JPEG', 'PNG', 'BMP', 'TIFF', 'GIF'],
            'output_formats': self.output_formats,
            'max_payload_size': self.max_payload_size,
            'dependencies': {
                'scipy_available': SCIPY_AVAILABLE,
                'opencv_available': True,  # Already imported
                'pil_available': True      # Already imported
            }
        }