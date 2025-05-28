#!/usr/bin/env python3
"""
StegAnalysis Suite - Forensic Analysis Module
Advanced forensic analysis including metadata extraction, integrity verification, and modification detection
"""

import os
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import struct

import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Optional imports for advanced metadata extraction
try:
    import pyexiv2
    PYEXIV2_AVAILABLE = True
except ImportError:
    PYEXIV2_AVAILABLE = False
    logging.warning("pyexiv2 not available. Advanced metadata extraction will be limited.")

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available. File type detection will be limited.")

try:
    from scipy import stats
    from scipy.fftpack import dct
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some analysis features will be limited.")


class ForensicAnalyzer:
    """Forensic analysis engine for steganography investigation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.forensic_config = config.get('forensic_analysis', {})
        
        # Configuration flags
        self.extract_metadata = self.forensic_config.get('extract_metadata', True)
        self.verify_integrity = self.forensic_config.get('verify_integrity', True)
        self.analyze_compression = self.forensic_config.get('analyze_compression', True)
        self.detect_modifications = self.forensic_config.get('detect_modifications', True)
        
        # Supported hash algorithms
        self.hash_algorithms = self.forensic_config.get('hash_algorithms', ['md5', 'sha1', 'sha256'])
        
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive forensic analysis on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing forensic analysis results
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            results = {
                'image_path': str(image_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'file_info': self._analyze_file_properties(image_path),
                'metadata_analysis': {},
                'integrity_analysis': {},
                'compression_analysis': {},
                'modification_analysis': {},
                'signature_analysis': {},
                'forensic_summary': {}
            }
            
            # File metadata extraction
            if self.extract_metadata:
                results['metadata_analysis'] = self._extract_metadata(image_path)
            
            # Integrity verification
            if self.verify_integrity:
                results['integrity_analysis'] = self._verify_integrity(image_path)
            
            # Compression analysis
            if self.analyze_compression:
                results['compression_analysis'] = self._analyze_compression(image_path)
            
            # Modification detection
            if self.detect_modifications:
                results['modification_analysis'] = self._detect_modifications(image_path)
            
            # File signature analysis
            results['signature_analysis'] = self._analyze_file_signature(image_path)
            
            # Generate forensic summary
            results['forensic_summary'] = self._generate_forensic_summary(results)
            
            self.logger.info(f"Forensic analysis completed for {image_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Forensic analysis failed for {image_path}: {str(e)}")
            return {
                'error': str(e),
                'image_path': str(image_path),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_file_properties(self, image_path: Path) -> Dict[str, Any]:
        """Analyze basic file properties"""
        stat = image_path.stat()
        
        # File hashes
        hashes = {}
        try:
            with open(image_path, 'rb') as f:
                file_content = f.read()
                
                for algo in self.hash_algorithms:
                    if algo.lower() == 'md5':
                        hashes['md5'] = hashlib.md5(file_content).hexdigest()
                    elif algo.lower() == 'sha1':
                        hashes['sha1'] = hashlib.sha1(file_content).hexdigest()
                    elif algo.lower() == 'sha256':
                        hashes['sha256'] = hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate hashes: {str(e)}")
        
        # File type detection
        file_type = self._detect_file_type(image_path)
        
        return {
            'filename': image_path.name,
            'file_size': int(stat.st_size),
            'creation_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modification_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'access_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'file_extension': image_path.suffix.lower(),
            'file_type': file_type,
            'hashes': hashes,
            'permissions': oct(stat.st_mode)[-3:]
        }
    
    def _detect_file_type(self, image_path: Path) -> Dict[str, str]:
        """Detect file type using multiple methods"""
        file_type_info = {
            'extension_based': image_path.suffix.lower(),
            'magic_based': 'unknown',
            'pil_based': 'unknown',
            'header_based': 'unknown'
        }
        
        # Magic library detection
        if MAGIC_AVAILABLE:
            try:
                file_type_info['magic_based'] = magic.from_file(str(image_path), mime=True)
            except Exception as e:
                self.logger.warning(f"Magic detection failed: {str(e)}")
        
        # PIL-based detection
        try:
            with Image.open(image_path) as img:
                file_type_info['pil_based'] = img.format.lower() if img.format else 'unknown'
        except Exception as e:
            self.logger.warning(f"PIL detection failed: {str(e)}")
        
        # Header-based detection
        file_type_info['header_based'] = self._detect_by_header(image_path)
        
        return file_type_info
    
    def _detect_by_header(self, image_path: Path) -> str:
        """Detect file type by examining file header"""
        try:
            with open(image_path, 'rb') as f:
                header = f.read(16)
            
            # Common image file signatures
            signatures = {
                b'\xFF\xD8\xFF': 'jpeg',
                b'\x89PNG\r\n\x1a\n': 'png',
                b'BM': 'bmp',
                b'GIF87a': 'gif',
                b'GIF89a': 'gif',
                b'RIFF': 'webp',
                b'II*\x00': 'tiff',
                b'MM\x00*': 'tiff'
            }
            
            for sig, file_type in signatures.items():
                if header.startswith(sig):
                    if file_type == 'webp' and len(header) >= 12:
                        if header[8:12] == b'WEBP':
                            return 'webp'
                        else:
                            continue
                    return file_type
            
            return 'unknown'
            
        except Exception as e:
            self.logger.warning(f"Header detection failed: {str(e)}")
            return 'unknown'
    
    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata from image"""
        metadata = {
            'exif_data': {},
            'iptc_data': {},
            'xmp_data': {},
            'icc_profile': {},
            'custom_fields': {},
            'extraction_method': 'pil_basic'
        }
        
        # Basic EXIF extraction using PIL
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        
                        # Handle GPS data specially
                        if tag_name == 'GPSInfo':
                            gps_data = {}
                            for gps_tag_id, gps_value in value.items():
                                gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                gps_data[gps_tag_name] = str(gps_value)
                            metadata['exif_data']['GPS'] = gps_data
                        else:
                            # Convert complex data types to strings for JSON serialization
                            if isinstance(value, (bytes, tuple)):
                                value = str(value)
                            metadata['exif_data'][tag_name] = value
                
                # ICC Profile information
                if hasattr(img, 'info') and 'icc_profile' in img.info:
                    metadata['icc_profile'] = {
                        'present': True,
                        'size': len(img.info['icc_profile'])
                    }
        
        except Exception as e:
            self.logger.warning(f"PIL metadata extraction failed: {str(e)}")
        
        # Advanced metadata extraction using pyexiv2
        if PYEXIV2_AVAILABLE:
            try:
                with pyexiv2.Image(str(image_path)) as img:
                    metadata['extraction_method'] = 'pyexiv2_advanced'
                    
                    # EXIF data
                    exif_dict = img.read_exif()
                    if exif_dict:
                        metadata['exif_data'].update(exif_dict)
                    
                    # IPTC data
                    iptc_dict = img.read_iptc()
                    if iptc_dict:
                        metadata['iptc_data'] = iptc_dict
                    
                    # XMP data
                    xmp_dict = img.read_xmp()
                    if xmp_dict:
                        metadata['xmp_data'] = xmp_dict
                        
            except Exception as e:
                self.logger.warning(f"pyexiv2 metadata extraction failed: {str(e)}")
        
        # Analyze metadata for forensic indicators
        metadata['forensic_indicators'] = self._analyze_metadata_forensics(metadata)
        
        return metadata
    
    def _analyze_metadata_forensics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata for forensic indicators"""
        indicators = {
            'software_signatures': [],
            'timestamp_anomalies': [],
            'missing_expected_fields': [],
            'suspicious_values': [],
            'modification_indicators': []
        }
        
        exif_data = metadata.get('exif_data', {})
        
        # Check for software signatures
        software_fields = ['Software', 'ProcessingSoftware', 'HostComputer']
        for field in software_fields:
            if field in exif_data:
                indicators['software_signatures'].append({
                    'field': field,
                    'value': str(exif_data[field])
                })
        
        # Check timestamps
        timestamp_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        timestamps = {}
        for field in timestamp_fields:
            if field in exif_data:
                timestamps[field] = str(exif_data[field])
        
        # Look for timestamp anomalies
        if len(timestamps) > 1:
            datetime_values = []
            for ts_str in timestamps.values():
                try:
                    dt = datetime.strptime(ts_str, '%Y:%m:%d %H:%M:%S')
                    datetime_values.append(dt)
                except:
                    continue
            
            if len(datetime_values) > 1:
                time_diffs = []
                for i in range(1, len(datetime_values)):
                    diff = abs((datetime_values[i] - datetime_values[0]).total_seconds())
                    time_diffs.append(diff)
                
                if any(diff > 3600 for diff in time_diffs):
                    indicators['timestamp_anomalies'].append('Inconsistent EXIF timestamps detected')
        
        # Check for missing expected fields
        expected_fields = ['Make', 'Model', 'Orientation', 'XResolution', 'YResolution']
        for field in expected_fields:
            if field not in exif_data:
                indicators['missing_expected_fields'].append(field)
        
        # Check for suspicious values
        if 'ColorSpace' in exif_data and exif_data['ColorSpace'] not in [1, 65535]:
            indicators['suspicious_values'].append('Unusual ColorSpace value')
        
        if 'Orientation' in exif_data and exif_data['Orientation'] not in range(1, 9):
            indicators['suspicious_values'].append('Invalid Orientation value')
        
        return indicators
    
    def _verify_integrity(self, image_path: Path) -> Dict[str, Any]:
        """Verify file integrity and detect corruption"""
        integrity = {
            'file_readable': False,
            'image_loadable': False,
            'header_valid': False,
            'structure_valid': False,
            'corruption_indicators': [],
            'repair_suggestions': []
        }
        
        # Check if file is readable
        try:
            with open(image_path, 'rb') as f:
                file_size = len(f.read())
            integrity['file_readable'] = True
            integrity['actual_file_size'] = file_size
        except Exception as e:
            integrity['corruption_indicators'].append(f"File not readable: {str(e)}")
            return integrity
        
        # Check if image is loadable by PIL
        try:
            with Image.open(image_path) as img:
                img.verify()
            integrity['image_loadable'] = True
        except Exception as e:
            integrity['corruption_indicators'].append(f"PIL cannot load image: {str(e)}")
        
        # Check if image is loadable by OpenCV
        try:
            image = cv2.imread(str(image_path))
            if image is not None:
                integrity['opencv_loadable'] = True
            else:
                integrity['corruption_indicators'].append("OpenCV cannot load image")
        except Exception as e:
            integrity['corruption_indicators'].append(f"OpenCV error: {str(e)}")
        
        # Header validation
        integrity['header_valid'] = self._validate_header(image_path)
        if not integrity['header_valid']:
            integrity['corruption_indicators'].append("Invalid file header")
        
        # Structure validation
        integrity['structure_valid'] = self._validate_structure(image_path)
        if not integrity['structure_valid']:
            integrity['corruption_indicators'].append("Invalid file structure")
        
        # Generate repair suggestions
        if integrity['corruption_indicators']:
            integrity['repair_suggestions'] = self._generate_repair_suggestions(integrity)
        
        return integrity
    
    def _validate_header(self, image_path: Path) -> bool:
        """Validate file header structure"""
        try:
            with open(image_path, 'rb') as f:
                header = f.read(32)
            
            file_type = self._detect_by_header(image_path)
            
            if file_type == 'jpeg':
                return header.startswith(b'\xFF\xD8\xFF')
            elif file_type == 'png':
                return header.startswith(b'\x89PNG\r\n\x1a\n')
            elif file_type == 'bmp':
                return header.startswith(b'BM')
            elif file_type in ['gif87a', 'gif89a']:
                return header.startswith(b'GIF')
            elif file_type == 'tiff':
                return header.startswith((b'II*\x00', b'MM\x00*'))
            
            return True
            
        except Exception:
            return False
    
    def _validate_structure(self, image_path: Path) -> bool:
        """Validate internal file structure"""
        try:
            file_type = self._detect_by_header(image_path)
            
            if file_type == 'jpeg':
                return self._validate_jpeg_structure(image_path)
            elif file_type == 'png':
                return self._validate_png_structure(image_path)
            
            return True
            
        except Exception:
            return False
    
    def _validate_jpeg_structure(self, image_path: Path) -> bool:
        """Validate JPEG file structure"""
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
                
                if not data.startswith(b'\xFF\xD8'):
                    return False
                
                if not data.endswith(b'\xFF\xD9'):
                    return False
                
                return True
                
        except Exception:
            return False
    
    def _validate_png_structure(self, image_path: Path) -> bool:
        """Validate PNG file structure"""
        try:
            with open(image_path, 'rb') as f:
                signature = f.read(8)
                if signature != b'\x89PNG\r\n\x1a\n':
                    return False
                
                chunk_length = struct.unpack('>I', f.read(4))[0]
                chunk_type = f.read(4)
                if chunk_type != b'IHDR':
                    return False
                
                return True
                
        except Exception:
            return False
    
    def _generate_repair_suggestions(self, integrity: Dict[str, Any]) -> List[str]:
        """Generate suggestions for repairing corrupted files"""
        suggestions = []
        
        if "Invalid file header" in integrity.get('corruption_indicators', []):
            suggestions.append("Try opening with image editing software and re-saving")
        
        if "PIL cannot load image" in str(integrity.get('corruption_indicators', [])):
            suggestions.append("File may be corrupted - try using alternative image viewers")
        
        if not integrity.get('structure_valid', True):
            suggestions.append("File structure is damaged - consider using specialized recovery tools")
        
        if not suggestions:
            suggestions.append("File appears to be severely corrupted - recovery may not be possible")
        
        return suggestions
    
    def _analyze_compression(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image compression characteristics"""
        compression_info = {
            'format': 'unknown',
            'quality_estimate': None,
            'compression_artifacts': [],
            'recompression_indicators': [],
            'compression_history': []
        }
        
        try:
            file_type = self._detect_by_header(image_path)
            compression_info['format'] = file_type
            
            if file_type == 'jpeg':
                compression_info.update(self._analyze_jpeg_compression(image_path))
            elif file_type == 'png':
                compression_info.update(self._analyze_png_compression(image_path))
            
        except Exception as e:
            self.logger.warning(f"Compression analysis failed: {str(e)}")
        
        return compression_info
    
    def _analyze_jpeg_compression(self, image_path: Path) -> Dict[str, Any]:
        """Analyze JPEG compression characteristics"""
        jpeg_info = {
            'quality_estimate': None,
            'quantization_tables': [],
            'double_compression': False,
            'compression_artifacts': []
        }
        
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return jpeg_info
            
            quality = self._estimate_jpeg_quality(image_path)
            jpeg_info['quality_estimate'] = quality
            
            double_comp = self._detect_double_compression(image)
            jpeg_info['double_compression'] = double_comp
            
            blocking_score = self._analyze_blocking_artifacts(image)
            if blocking_score > 0.5:
                jpeg_info['compression_artifacts'].append('Blocking artifacts detected')
            
        except Exception as e:
            self.logger.warning(f"JPEG compression analysis failed: {str(e)}")
        
        return jpeg_info
    
    def _estimate_jpeg_quality(self, image_path: Path) -> Optional[int]:
        """Estimate JPEG quality factor"""
        try:
            with Image.open(image_path) as img:
                file_size = image_path.stat().st_size
                width, height = img.size
                pixels = width * height
                
                if pixels > 0:
                    bytes_per_pixel = file_size / pixels
                    if bytes_per_pixel > 2.0:
                        return 95
                    elif bytes_per_pixel > 1.5:
                        return 85
                    elif bytes_per_pixel > 1.0:
                        return 75
                    elif bytes_per_pixel > 0.5:
                        return 60
                    elif bytes_per_pixel > 0.3:
                        return 40
                    else:
                        return 20
                
        except Exception:
            pass
        
        return None
    
    def _detect_double_compression(self, image: np.ndarray) -> bool:
        """Detect signs of double JPEG compression"""
        try:
            if not SCIPY_AVAILABLE:
                return False
                
            height, width = image.shape
            block_scores = []
            
            for i in range(0, height - 7, 32):
                for j in range(0, width - 7, 32):
                    block = image[i:i+8, j:j+8].astype(np.float32)
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    coeffs = dct_block.flatten()[1:]
                    hist, _ = np.histogram(coeffs, bins=50, range=(-50, 50))
                    periodic_score = self._detect_periodicity(hist)
                    block_scores.append(periodic_score)
            
            if block_scores:
                avg_periodicity = np.mean(block_scores)
                return avg_periodicity > 0.3
            
        except Exception:
            pass
        
        return False
    
    def _detect_periodicity(self, histogram: np.ndarray) -> float:
        """Detect periodicity in histogram"""
        try:
            autocorr = np.correlate(histogram, histogram, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            if len(autocorr) > 1:
                autocorr = autocorr / autocorr[0]
                if len(autocorr) > 5:
                    secondary_peak = np.max(autocorr[2:min(10, len(autocorr))])
                    return secondary_peak
            
        except Exception:
            pass
        
        return 0.0
    
    def _analyze_blocking_artifacts(self, image: np.ndarray) -> float:
        """Analyze JPEG blocking artifacts"""
        try:
            height, width = image.shape
            h_diffs = []
            
            for i in range(0, height, 8):
                if i > 0 and i < height - 1:
                    diff = np.mean(np.abs(image[i, :] - image[i-1, :]))
                    h_diffs.append(diff)
            
            if h_diffs:
                return np.mean(h_diffs) / 255.0
            
        except Exception:
            pass
        
        return 0.0
    
    def _analyze_png_compression(self, image_path: Path) -> Dict[str, Any]:
        """Analyze PNG compression characteristics"""
        png_info = {
            'compression_method': None,
            'filter_method': None,
            'interlace_method': None,
            'color_type': None,
            'bit_depth': None
        }
        
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'info'):
                    png_info.update({
                        'compression_method': img.info.get('compression', 'unknown'),
                        'filter_method': img.info.get('filter', 'unknown'),
                        'interlace_method': img.info.get('interlace', 'unknown')
                    })
                
                png_info['color_type'] = img.mode
                png_info['bit_depth'] = getattr(img, 'bits', 8)
            
        except Exception as e:
            self.logger.warning(f"PNG compression analysis failed: {str(e)}")
        
        return png_info
    
    def _detect_modifications(self, image_path: Path) -> Dict[str, Any]:
        """Detect signs of image modification"""
        modification_analysis = {
            'modification_indicators': [],
            'suspicious_regions': [],
            'copy_move_detection': {},
            'splicing_detection': {},
            'resampling_detection': {},
            'noise_analysis': {}
        }
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return modification_analysis
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            modification_analysis['copy_move_detection'] = self._detect_copy_move(gray)
            modification_analysis['splicing_detection'] = self._detect_splicing(image)
            modification_analysis['noise_analysis'] = self._analyze_noise_consistency(image)
            modification_analysis['modification_indicators'] = self._compile_modification_indicators(modification_analysis)
            
        except Exception as e:
            self.logger.warning(f"Modification detection failed: {str(e)}")
        
        return modification_analysis
    
    def _detect_copy_move(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect copy-move forgery"""
        copy_move_info = {
            'detected': False,
            'confidence': 0.0,
            'suspicious_blocks': []
        }
        
        try:
            height, width = image.shape
            block_size = 16
            blocks = []
            positions = []
            
            for i in range(0, height - block_size, 8):
                for j in range(0, width - block_size, 8):
                    block = image[i:i+block_size, j:j+block_size]
                    blocks.append(block.flatten())
                    positions.append((i, j))
            
            if len(blocks) < 2:
                return copy_move_info
            
            blocks_array = np.array(blocks)
            threshold = 0.95
            similar_pairs = []
            
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    pos1, pos2 = positions[i], positions[j]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if distance < block_size * 2:
                        continue
                    
                    corr = np.corrcoef(blocks_array[i], blocks_array[j])[0, 1]
                    if not np.isnan(corr) and corr > threshold:
                        similar_pairs.append({
                            'block1': pos1,
                            'block2': pos2,
                            'correlation': float(corr),
                            'distance': float(distance)
                        })
            
            if similar_pairs:
                copy_move_info['detected'] = True
                copy_move_info['confidence'] = min(len(similar_pairs) / 10.0, 1.0)
                copy_move_info['suspicious_blocks'] = similar_pairs[:10]
            
        except Exception as e:
            self.logger.warning(f"Copy-move detection failed: {str(e)}")
        
        return copy_move_info
    
    def _detect_splicing(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect image splicing"""
        splicing_info = {
            'detected': False,
            'confidence': 0.0,
            'inconsistencies': []
        }
        
        try:
            # Simplified splicing detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            regions = []
            region_size = min(height, width) // 4
            
            for i in range(0, height - region_size, region_size):
                for j in range(0, width - region_size, region_size):
                    region = gray[i:i+region_size, j:j+region_size]
                    regions.append(region)
            
            brightness_values = [np.mean(region) for region in regions]
            brightness_std = np.std(brightness_values)
            
            if brightness_std > 30:
                splicing_info['detected'] = True
                splicing_info['confidence'] = min(brightness_std / 100.0, 1.0)
                splicing_info['inconsistencies'].append('Lighting inconsistency detected')
            
        except Exception as e:
            self.logger.warning(f"Splicing detection failed: {str(e)}")
        
        return splicing_info
    
    def _analyze_noise_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise consistency across the image"""
        noise_info = {
            'consistent': True,
            'noise_variations': [],
            'suspicious_regions': []
        }
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            height, width = gray.shape
            region_size = min(height, width) // 8
            
            noise_estimates = []
            
            for i in range(0, height - region_size, region_size):
                for j in range(0, width - region_size, region_size):
                    region = gray[i:i+region_size, j:j+region_size]
                    laplacian = cv2.Laplacian(region, cv2.CV_64F)
                    noise_estimates.append(noise_estimate)
            
            if len(noise_estimates) > 1:
                noise_std = np.std(noise_estimates)
                noise_mean = np.mean(noise_estimates)
                
                if noise_std / noise_mean > 0.5:  # High variation
                    noise_info['consistent'] = False
                
                noise_info['noise_variations'] = {
                    'mean': float(noise_mean),
                    'std': float(noise_std),
                    'coefficient_of_variation': float(noise_std / noise_mean) if noise_mean > 0 else 0
                }
            
        except Exception as e:
            self.logger.warning(f"Noise consistency analysis failed: {str(e)}")
        
        return noise_info
    
    def _compile_modification_indicators(self, analysis: Dict[str, Any]) -> List[str]:
        """Compile list of modification indicators"""
        indicators = []
        
        if analysis.get('copy_move_detection', {}).get('detected', False):
            indicators.append('Copy-move forgery detected')
        
        if analysis.get('splicing_detection', {}).get('detected', False):
            indicators.append('Image splicing detected')
        
        if not analysis.get('noise_analysis', {}).get('consistent', True):
            indicators.append('Noise inconsistency detected')
        
        return indicators
    
    def _analyze_file_signature(self, image_path: Path) -> Dict[str, Any]:
        """Analyze file signature and header information"""
        signature_info = {
            'header_analysis': {},
            'signature_match': True,
            'extension_mismatch': False,
            'embedded_data': {}
        }
        
        try:
            with open(image_path, 'rb') as f:
                header = f.read(64)
            
            signature_info['header_analysis'] = self._analyze_header_structure(header, image_path.suffix)
            
            detected_type = self._detect_by_header(image_path)
            expected_extensions = {
                'jpeg': ['.jpg', '.jpeg'],
                'png': ['.png'],
                'bmp': ['.bmp'],
                'gif': ['.gif'],
                'tiff': ['.tiff', '.tif']
            }
            
            if detected_type in expected_extensions:
                if image_path.suffix.lower() not in expected_extensions[detected_type]:
                    signature_info['extension_mismatch'] = True
                    signature_info['detected_type'] = detected_type
                    signature_info['file_extension'] = image_path.suffix.lower()
            
            signature_info['embedded_data'] = self._scan_for_embedded_data(image_path)
            
        except Exception as e:
            self.logger.warning(f"File signature analysis failed: {str(e)}")
        
        return signature_info
    
    def _analyze_header_structure(self, header: bytes, extension: str) -> Dict[str, Any]:
        """Analyze file header structure"""
        header_info = {
            'valid_signature': False,
            'header_bytes': header[:16].hex(),
            'anomalies': []
        }
        
        if extension.lower() in ['.jpg', '.jpeg']:
            header_info['valid_signature'] = header.startswith(b'\xFF\xD8\xFF')
            if not header_info['valid_signature']:
                header_info['anomalies'].append('Invalid JPEG signature')
        elif extension.lower() == '.png':
            header_info['valid_signature'] = header.startswith(b'\x89PNG\r\n\x1a\n')
            if not header_info['valid_signature']:
                header_info['anomalies'].append('Invalid PNG signature')
        elif extension.lower() == '.bmp':
            header_info['valid_signature'] = header.startswith(b'BM')
            if not header_info['valid_signature']:
                header_info['anomalies'].append('Invalid BMP signature')
        elif extension.lower() == '.gif':
            header_info['valid_signature'] = header.startswith((b'GIF87a', b'GIF89a'))
            if not header_info['valid_signature']:
                header_info['anomalies'].append('Invalid GIF signature')
        
        return header_info
    
    def _scan_for_embedded_data(self, image_path: Path) -> Dict[str, Any]:
        """Scan for embedded data signatures within the image file"""
        embedded_info = {
            'zip_signatures': [],
            'executable_signatures': [],
            'text_signatures': [],
            'other_signatures': []
        }
        
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            
            signatures = {
                'zip': [b'PK\x03\x04', b'PK\x05\x06'],
                'rar': [b'Rar!\x1a\x07\x00'],
                'pdf': [b'%PDF'],
                'exe': [b'MZ'],
                'elf': [b'\x7fELF']
            }
            
            for file_type, sigs in signatures.items():
                for sig in sigs:
                    pos = content.find(sig)
                    if pos != -1:
                        if file_type in ['zip', 'rar']:
                            embedded_info['zip_signatures'].append({
                                'type': file_type,
                                'offset': pos,
                                'signature': sig.hex()
                            })
                        elif file_type in ['exe', 'elf']:
                            embedded_info['executable_signatures'].append({
                                'type': file_type,
                                'offset': pos,
                                'signature': sig.hex()
                            })
                        else:
                            embedded_info['other_signatures'].append({
                                'type': file_type,
                                'offset': pos,
                                'signature': sig.hex()
                            })
        
        except Exception as e:
            self.logger.warning(f"Embedded data scan failed: {str(e)}")
        
        return embedded_info
    
    def _generate_forensic_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive forensic summary"""
        summary = {
            'overall_assessment': 'normal',
            'confidence_score': 0.0,
            'key_findings': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        risk_score = 0.0
        
        # Analyze metadata indicators
        metadata = results.get('metadata_analysis', {})
        forensic_indicators = metadata.get('forensic_indicators', {})
        
        if forensic_indicators.get('timestamp_anomalies'):
            risk_score += 0.2
            summary['risk_factors'].append('Timestamp anomalies in metadata')
        
        if forensic_indicators.get('missing_expected_fields'):
            risk_score += 0.1
            summary['risk_factors'].append('Missing expected metadata fields')
        
        # Analyze integrity issues
        integrity = results.get('integrity_analysis', {})
        if integrity.get('corruption_indicators'):
            risk_score += 0.3
            summary['risk_factors'].append('File integrity issues detected')
        
        # Analyze modification indicators
        modifications = results.get('modification_analysis', {})
        mod_indicators = modifications.get('modification_indicators', [])
        
        if mod_indicators:
            risk_score += 0.4
            summary['risk_factors'].extend(mod_indicators)
        
        # Analyze signature issues
        signature = results.get('signature_analysis', {})
        if signature.get('extension_mismatch'):
            risk_score += 0.2
            summary['risk_factors'].append('File extension mismatch')
        
        embedded_data = signature.get('embedded_data', {})
        if (embedded_data.get('zip_signatures') or 
            embedded_data.get('executable_signatures') or 
            embedded_data.get('other_signatures')):
            risk_score += 0.5
            summary['risk_factors'].append('Embedded data signatures detected')
        
        # Calculate overall assessment
        summary['confidence_score'] = min(risk_score, 1.0)
        
        if risk_score >= 0.7:
            summary['overall_assessment'] = 'highly_suspicious'
        elif risk_score >= 0.4:
            summary['overall_assessment'] = 'suspicious'
        elif risk_score >= 0.2:
            summary['overall_assessment'] = 'potentially_modified'
        else:
            summary['overall_assessment'] = 'normal'
        
        # Generate recommendations
        if summary['overall_assessment'] in ['highly_suspicious', 'suspicious']:
            summary['recommendations'].extend([
                'Conduct detailed steganography analysis',
                'Verify file provenance and chain of custody',
                'Consider professional forensic examination'
            ])
        elif summary['overall_assessment'] == 'potentially_modified':
            summary['recommendations'].extend([
                'Perform additional integrity checks',
                'Compare with original if available',
                'Monitor for additional suspicious indicators'
            ])
        else:
            summary['recommendations'].append('File appears forensically normal')
        
        # Key findings summary
        if embedded_data.get('zip_signatures'):
            summary['key_findings'].append(f"Found {len(embedded_data['zip_signatures'])} archive signatures")
        
        if embedded_data.get('executable_signatures'):
            summary['key_findings'].append(f"Found {len(embedded_data['executable_signatures'])} executable signatures")
        
        if mod_indicators:
            summary['key_findings'].append(f"Detected {len(mod_indicators)} modification indicators")
        
        if not summary['key_findings']:
            summary['key_findings'].append('No significant forensic anomalies detected')
        
        return summary
    
    def generate_forensic_report(self, results: Dict[str, Any], output_path: str):
        """Generate comprehensive forensic report"""
        try:
            report = {
                'forensic_analysis_report': {
                    'header': {
                        'analysis_timestamp': results.get('analysis_timestamp'),
                        'image_path': results.get('image_path'),
                        'analyzer_version': '1.0.0',
                        'analysis_type': 'comprehensive_forensic'
                    },
                    'executive_summary': results.get('forensic_summary', {}),
                    'detailed_findings': {
                        'file_properties': results.get('file_info', {}),
                        'metadata_analysis': results.get('metadata_analysis', {}),
                        'integrity_verification': results.get('integrity_analysis', {}),
                        'compression_analysis': results.get('compression_analysis', {}),
                        'modification_detection': results.get('modification_analysis', {}),
                        'signature_analysis': results.get('signature_analysis', {})
                    },
                    'risk_assessment': {
                        'overall_risk': results.get('forensic_summary', {}).get('overall_assessment', 'unknown'),
                        'confidence': results.get('forensic_summary', {}).get('confidence_score', 0.0),
                        'risk_factors': results.get('forensic_summary', {}).get('risk_factors', []),
                        'recommendations': results.get('forensic_summary', {}).get('recommendations', [])
                    }
                }
            }
            
            with open(f"{output_path}_forensic_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Forensic report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate forensic report: {str(e)}")
    
    def create_evidence_package(self, image_path: str, analysis_results: Dict[str, Any], output_dir: str):
        """Create a complete evidence package for forensic analysis"""
        try:
            import shutil
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence_dir = output_path / f"evidence_package_{timestamp}"
            evidence_dir.mkdir(exist_ok=True)
            
            # Copy original file
            original_file = Path(image_path)
            evidence_file = evidence_dir / f"original_{original_file.name}"
            shutil.copy2(original_file, evidence_file)
            
            # Generate comprehensive reports
            base_name = evidence_dir / "forensic_analysis"
            self.generate_forensic_report(analysis_results, str(base_name))
            
            self.logger.info(f"Evidence package created: {evidence_dir}")
            return str(evidence_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create evidence package: {str(e)}")
            return None