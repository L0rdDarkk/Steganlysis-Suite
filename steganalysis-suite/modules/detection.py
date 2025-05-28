#!/usr/bin/env python3
"""
StegAnalysis Suite - Core Detection Module
Traditional steganography detection algorithms including LSB, Chi-Square, and F5
"""

import numpy as np
import cv2
from PIL import Image
import logging
from scipy import stats
from scipy.fftpack import dct, idct
import math
from typing import Dict, Tuple, Any, Optional
from pathlib import Path


class DetectionEngine:
    """Core detection engine for traditional steganography detection algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.detection_config = config.get('detection', {})
        self.thresholds = self.detection_config.get('thresholds', {})
        
    def lsb_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Perform LSB (Least Significant Bit) analysis on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing LSB analysis results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width, channels = image.shape
            
            # Analyze each color channel
            channel_results = {}
            overall_suspicious = False
            max_lsb_ratio = 0
            
            for channel_idx, channel_name in enumerate(['blue', 'green', 'red']):
                channel = image[:, :, channel_idx]
                
                # Extract LSBs
                lsbs = channel & 1
                
                # Calculate statistics
                lsb_ratio = np.mean(lsbs)
                
                # Chi-square test on LSB plane
                observed = np.bincount(lsbs.flatten(), minlength=2)
                expected = np.array([lsbs.size / 2, lsbs.size / 2])
                chi2_stat, p_value = stats.chisquare(observed, expected)
                
                # Pattern analysis - look for regularity
                patterns = self._analyze_lsb_patterns(lsbs)
                
                # Determine if channel is suspicious
                threshold = self.thresholds.get('lsb_threshold', 0.7)
                is_suspicious = (
                    abs(lsb_ratio - 0.5) > 0.1 or  # Deviation from random
                    p_value < 0.05 or              # Chi-square test
                    patterns['regularity_score'] > threshold
                )
                
                channel_results[channel_name] = {
                    'lsb_ratio': float(lsb_ratio),
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'patterns': patterns,
                    'suspicious': is_suspicious
                }
                
                if is_suspicious:
                    overall_suspicious = True
                    max_lsb_ratio = max(max_lsb_ratio, patterns['regularity_score'])
            
            # Overall analysis
            result = {
                'detected': overall_suspicious,
                'confidence': min(max_lsb_ratio, 1.0),
                'channel_analysis': channel_results,
                'method': 'lsb_analysis',
                'image_info': {
                    'width': int(width),
                    'height': int(height),
                    'channels': int(channels),
                    'total_pixels': int(width * height)
                }
            }
            
            self.logger.info(f"LSB analysis completed for {image_path}: "
                           f"Detected={overall_suspicious}, Confidence={max_lsb_ratio:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"LSB analysis failed for {image_path}: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'error': str(e),
                'method': 'lsb_analysis'
            }
    
    def _analyze_lsb_patterns(self, lsb_plane: np.ndarray) -> Dict[str, float]:
        """Analyze patterns in LSB plane"""
        # Flatten the LSB plane
        lsbs = lsb_plane.flatten()
        
        # Calculate runs (consecutive identical bits)
        runs = []
        current_run = 1
        for i in range(1, len(lsbs)):
            if lsbs[i] == lsbs[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Expected vs actual run distribution
        expected_avg_run = 2.0  # For random data
        actual_avg_run = np.mean(runs)
        
        # Entropy calculation
        entropy = self._calculate_entropy(lsbs)
        
        # Periodicity detection using autocorrelation
        autocorr = np.correlate(lsbs - np.mean(lsbs), lsbs - np.mean(lsbs), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # Look for periodic patterns
        max_autocorr = np.max(autocorr[1:min(100, len(autocorr))])
        
        # Regularity score combines multiple factors
        regularity_score = (
            abs(actual_avg_run - expected_avg_run) / expected_avg_run * 0.3 +
            (1.0 - entropy) * 0.4 +
            max_autocorr * 0.3
        )
        
        return {
            'entropy': float(entropy),
            'avg_run_length': float(actual_avg_run),
            'max_autocorr': float(max_autocorr),
            'regularity_score': float(regularity_score),
            'total_runs': len(runs)
        }
    
    def chi_square_test(self, image_path: str) -> Dict[str, Any]:
        """
        Perform Chi-Square test for steganography detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing chi-square test results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            results = {}
            overall_detected = False
            max_confidence = 0.0
            
            # Test each color channel
            for channel_idx, channel_name in enumerate(['blue', 'green', 'red']):
                channel = image[:, :, channel_idx]
                
                # Perform chi-square test on pixel pairs
                chi2_result = self._chi_square_pixel_pairs(channel)
                
                # Perform chi-square test on LSB plane
                lsb_chi2_result = self._chi_square_lsb_plane(channel)
                
                # Combine results
                threshold = self.thresholds.get('chi_square', 0.05)
                detected = (chi2_result['p_value'] < threshold or 
                           lsb_chi2_result['p_value'] < threshold)
                
                confidence = 1.0 - min(chi2_result['p_value'], lsb_chi2_result['p_value'])
                
                results[channel_name] = {
                    'pixel_pairs': chi2_result,
                    'lsb_plane': lsb_chi2_result,
                    'detected': detected,
                    'confidence': float(confidence)
                }
                
                if detected:
                    overall_detected = True
                    max_confidence = max(max_confidence, confidence)
            
            result = {
                'detected': overall_detected,
                'confidence': max_confidence,
                'channel_results': results,
                'method': 'chi_square_test',
                'threshold_used': threshold
            }
            
            self.logger.info(f"Chi-square test completed for {image_path}: "
                           f"Detected={overall_detected}, Confidence={max_confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chi-square test failed for {image_path}: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'error': str(e),
                'method': 'chi_square_test'
            }
    
    def _chi_square_pixel_pairs(self, channel: np.ndarray) -> Dict[str, float]:
        """Perform chi-square test on pixel value pairs"""
        # Create pairs of adjacent pixels
        pairs = []
        flat_channel = channel.flatten()
        
        for i in range(0, len(flat_channel) - 1, 2):
            pairs.append((flat_channel[i], flat_channel[i + 1]))
        
        # Count frequency of each pair type
        pair_counts = {}
        for pair in pairs:
            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1
        
        # Expected frequencies (uniform distribution)
        total_pairs = len(pairs)
        unique_pairs = len(pair_counts)
        expected_freq = total_pairs / unique_pairs if unique_pairs > 0 else 0
        
        # Calculate chi-square statistic
        observed = list(pair_counts.values())
        expected = [expected_freq] * len(observed)
        
        if len(observed) > 1:
            chi2_stat, p_value = stats.chisquare(observed, expected)
        else:
            chi2_stat, p_value = 0.0, 1.0
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'unique_pairs': unique_pairs,
            'total_pairs': total_pairs
        }
    
    def _chi_square_lsb_plane(self, channel: np.ndarray) -> Dict[str, float]:
        """Perform chi-square test specifically on LSB plane"""
        # Extract LSBs
        lsbs = (channel & 1).flatten()
        
        # Count 0s and 1s
        observed = np.bincount(lsbs, minlength=2)
        expected = np.array([len(lsbs) / 2, len(lsbs) / 2])
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'zeros': int(observed[0]),
            'ones': int(observed[1])
        }
    
    def f5_detection(self, image_path: str) -> Dict[str, Any]:
        """
        F5 steganography detection algorithm
        Detects F5 algorithm artifacts in JPEG images
        
        Args:
            image_path: Path to the JPEG image file
            
        Returns:
            Dictionary containing F5 detection results
        """
        try:
            # Check if image is JPEG
            if not self._is_jpeg(image_path):
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'error': 'F5 detection only works on JPEG images',
                    'method': 'f5_detection'
                }
            
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to float for DCT operations
            image_float = image.astype(np.float32)
            
            # Perform block-wise DCT analysis
            block_size = 8
            height, width = image_float.shape
            
            f5_artifacts = []
            
            # Process 8x8 blocks
            for i in range(0, height - block_size + 1, block_size):
                for j in range(0, width - block_size + 1, block_size):
                    block = image_float[i:i+block_size, j:j+block_size]
                    
                    # Apply DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Analyze F5 artifacts
                    artifacts = self._analyze_f5_artifacts(dct_block)
                    f5_artifacts.append(artifacts)
            
            # Aggregate results
            if f5_artifacts:
                avg_artifacts = np.mean([a['artifact_score'] for a in f5_artifacts])
                histogram_analysis = self._f5_histogram_analysis(f5_artifacts)
                
                # Detection threshold
                threshold = self.thresholds.get('f5_threshold', 0.6)
                detected = avg_artifacts > threshold
                
                result = {
                    'detected': detected,
                    'confidence': float(min(avg_artifacts, 1.0)),
                    'average_artifact_score': float(avg_artifacts),
                    'histogram_analysis': histogram_analysis,
                    'blocks_analyzed': len(f5_artifacts),
                    'method': 'f5_detection',
                    'threshold_used': threshold
                }
            else:
                result = {
                    'detected': False,
                    'confidence': 0.0,
                    'error': 'No blocks could be analyzed',
                    'method': 'f5_detection'
                }
            
            self.logger.info(f"F5 detection completed for {image_path}: "
                           f"Detected={result.get('detected', False)}, "
                           f"Confidence={result.get('confidence', 0.0):.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"F5 detection failed for {image_path}: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'error': str(e),
                'method': 'f5_detection'
            }
    
    def _analyze_f5_artifacts(self, dct_block: np.ndarray) -> Dict[str, float]:
        """Analyze F5 artifacts in a DCT block"""
        # F5 algorithm modifies DCT coefficients
        # Look for characteristic patterns
        
        # Flatten DCT coefficients
        coeffs = dct_block.flatten()
        
        # Remove DC coefficient
        coeffs = coeffs[1:]
        
        # Count zero coefficients (F5 tends to increase these)
        zero_count = np.sum(coeffs == 0)
        zero_ratio = zero_count / len(coeffs)
        
        # Analyze coefficient distribution
        non_zero_coeffs = coeffs[coeffs != 0]
        
        if len(non_zero_coeffs) > 0:
            # Statistical measures
            coeff_variance = np.var(non_zero_coeffs)
            coeff_skewness = stats.skew(non_zero_coeffs)
            coeff_kurtosis = stats.kurtosis(non_zero_coeffs)
            
            # F5 artifact indicators
            # 1. Unusual zero coefficient ratio
            zero_artifact = abs(zero_ratio - 0.6)  # Expected ratio for natural images
            
            # 2. Distribution anomalies
            dist_artifact = abs(coeff_skewness) + abs(coeff_kurtosis - 3)
            
            # 3. Variance analysis
            var_artifact = min(coeff_variance / 100.0, 1.0)  # Normalize
            
            # Combined artifact score
            artifact_score = (zero_artifact * 0.4 + 
                            dist_artifact * 0.3 + 
                            var_artifact * 0.3)
        else:
            artifact_score = zero_ratio  # All zeros is suspicious
        
        return {
            'artifact_score': float(artifact_score),
            'zero_ratio': float(zero_ratio),
            'zero_count': int(zero_count),
            'total_coeffs': len(coeffs) + 1  # +1 for DC
        }
    
    def _f5_histogram_analysis(self, artifacts_list: list) -> Dict[str, Any]:
        """Analyze histogram of F5 artifacts across all blocks"""
        scores = [a['artifact_score'] for a in artifacts_list]
        zero_ratios = [a['zero_ratio'] for a in artifacts_list]
        
        return {
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_max': float(np.max(scores)),
            'zero_ratio_mean': float(np.mean(zero_ratios)),
            'zero_ratio_std': float(np.std(zero_ratios))
        }
    
    def _is_jpeg(self, image_path: str) -> bool:
        """Check if image is JPEG format"""
        try:
            with Image.open(image_path) as img:
                return img.format == 'JPEG'
        except:
            return False
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data"""
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def detect_all(self, image_path: str) -> Dict[str, Any]:
        """
        Run all detection algorithms on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing results from all detection methods
        """
        self.logger.info(f"Running all detection algorithms on {image_path}")
        
        algorithms = self.detection_config.get('algorithms', ['lsb', 'chi_square', 'f5'])
        results = {}
        
        if 'lsb' in algorithms:
            results['lsb'] = self.lsb_analysis(image_path)
        
        if 'chi_square' in algorithms:
            results['chi_square'] = self.chi_square_test(image_path)
        
        if 'f5' in algorithms:
            results['f5'] = self.f5_detection(image_path)
        
        # Calculate overall detection result
        detections = [r.get('detected', False) for r in results.values() if 'error' not in r]
        confidences = [r.get('confidence', 0.0) for r in results.values() if 'error' not in r]
        
        overall_detected = sum(detections) >= len(detections) // 2  # Majority vote
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'individual_results': results,
            'overall_detected': overall_detected,
            'overall_confidence': float(overall_confidence),
            'algorithms_used': algorithms,
            'image_path': image_path
        }