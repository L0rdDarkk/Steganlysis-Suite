#!/usr/bin/env python3
"""
StegAnalysis Suite - Statistical Analysis Module
Comprehensive statistical analysis for steganography detection
"""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.fftpack import fft2, fftshift
from sklearn.metrics import mutual_info_score
import pandas as pd


class StatisticalAnalyzer:
    """Statistical analysis engine for steganography detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stats_config = config.get('statistical_analysis', {})
        
        # Configuration parameters
        self.histogram_bins = self.stats_config.get('histogram_bins', 256)
        self.entropy_window = self.stats_config.get('entropy_window_size', 8)
        self.dct_block_size = self.stats_config.get('dct_block_size', 8)
        self.channels_to_analyze = self.stats_config.get('analyze_channels', ['red', 'green', 'blue'])
        
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all statistical analysis results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB for consistency
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = {
                'image_path': str(image_path),
                'image_properties': self._get_image_properties(image_rgb),
                'histogram_analysis': self._histogram_analysis(image_rgb),
                'entropy_analysis': self._entropy_analysis(image_rgb),
                'frequency_analysis': self._frequency_analysis(image_rgb),
                'texture_analysis': self._texture_analysis(image_rgb),
                'color_analysis': self._color_analysis(image_rgb),
                'noise_analysis': self._noise_analysis(image_rgb),
                'correlation_analysis': self._correlation_analysis(image_rgb),
                'distribution_analysis': self._distribution_analysis(image_rgb),
                'anomaly_scores': {}
            }
            
            # Calculate anomaly scores
            results['anomaly_scores'] = self._calculate_anomaly_scores(results)
            
            self.logger.info(f"Statistical analysis completed for {image_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed for {image_path}: {str(e)}")
            return {
                'error': str(e),
                'image_path': str(image_path)
            }
    
    def _get_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract basic image properties"""
        height, width, channels = image.shape
        
        return {
            'height': int(height),
            'width': int(width),
            'channels': int(channels),
            'total_pixels': int(height * width),
            'aspect_ratio': float(width / height),
            'bit_depth': image.dtype.name,
            'file_size_pixels': int(height * width * channels)
        }
    
    def _histogram_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze histogram properties for each channel"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i]
            
            # Calculate histogram
            hist, bins = np.histogram(channel, bins=self.histogram_bins, range=(0, 256))
            
            # Normalize histogram
            hist_norm = hist / np.sum(hist)
            
            # Statistical measures
            mean_intensity = np.mean(channel)
            std_intensity = np.std(channel)
            skewness = stats.skew(channel.flatten())
            kurtosis = stats.kurtosis(channel.flatten())
            
            # Histogram-specific measures
            hist_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            hist_variance = np.var(hist_norm)
            hist_uniformity = np.sum(hist_norm ** 2)
            
            # Peak analysis
            peaks, _ = signal.find_peaks(hist, height=np.max(hist) * 0.1)
            peak_count = len(peaks)
            
            # Gaps in histogram (potential sign of manipulation)
            zero_bins = np.sum(hist == 0)
            gap_ratio = zero_bins / len(hist)
            
            # Chi-square goodness of fit test (uniform distribution)
            expected_uniform = np.full(len(hist), np.sum(hist) / len(hist))
            chi2_uniform, p_uniform = stats.chisquare(hist, expected_uniform)
            
            # Chi-square test against normal distribution
            # Create expected normal distribution
            x = np.linspace(0, 255, len(hist))
            normal_pdf = stats.norm.pdf(x, mean_intensity, std_intensity)
            expected_normal = normal_pdf * np.sum(hist) / np.sum(normal_pdf)
            chi2_normal, p_normal = stats.chisquare(hist, expected_normal)
            
            results[channel_name] = {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'histogram_entropy': float(hist_entropy),
                'histogram_variance': float(hist_variance),
                'histogram_uniformity': float(hist_uniformity),
                'peak_count': int(peak_count),
                'gap_ratio': float(gap_ratio),
                'zero_bins': int(zero_bins),
                'chi2_uniform_stat': float(chi2_uniform),
                'chi2_uniform_p': float(p_uniform),
                'chi2_normal_stat': float(chi2_normal),
                'chi2_normal_p': float(p_normal),
                'histogram': hist.tolist(),
                'bins': bins.tolist()
            }
        
        return results
    
    def _entropy_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze entropy at different scales and regions"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i]
            
            # Global entropy
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            prob = hist / np.sum(hist)
            global_entropy = -np.sum(prob * np.log2(prob + 1e-10))
            
            # Local entropy using sliding window
            local_entropies = self._calculate_local_entropy(channel, self.entropy_window)
            
            # Block-wise entropy
            block_entropies = self._calculate_block_entropy(channel, block_size=16)
            
            # LSB entropy
            lsb_plane = channel & 1
            lsb_hist, _ = np.histogram(lsb_plane, bins=2, range=(0, 2))
            lsb_prob = lsb_hist / np.sum(lsb_hist)
            lsb_entropy = -np.sum(lsb_prob * np.log2(lsb_prob + 1e-10))
            
            # Conditional entropy (pixel pairs)
            conditional_entropy = self._calculate_conditional_entropy(channel)
            
            results[channel_name] = {
                'global_entropy': float(global_entropy),
                'local_entropy_mean': float(np.mean(local_entropies)),
                'local_entropy_std': float(np.std(local_entropies)),
                'local_entropy_min': float(np.min(local_entropies)),
                'local_entropy_max': float(np.max(local_entropies)),
                'block_entropy_mean': float(np.mean(block_entropies)),
                'block_entropy_std': float(np.std(block_entropies)),
                'lsb_entropy': float(lsb_entropy),
                'conditional_entropy': float(conditional_entropy),
                'entropy_variance': float(np.var(local_entropies))
            }
        
        return results
    
    def _calculate_local_entropy(self, channel: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local entropy using sliding window"""
        height, width = channel.shape
        local_entropies = []
        
        for i in range(0, height - window_size + 1, window_size // 2):
            for j in range(0, width - window_size + 1, window_size // 2):
                window = channel[i:i+window_size, j:j+window_size]
                hist, _ = np.histogram(window, bins=256, range=(0, 256))
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                local_entropies.append(entropy)
        
        return np.array(local_entropies)
    
    def _calculate_block_entropy(self, channel: np.ndarray, block_size: int) -> List[float]:
        """Calculate entropy for non-overlapping blocks"""
        height, width = channel.shape
        block_entropies = []
        
        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                block = channel[i:i+block_size, j:j+block_size]
                hist, _ = np.histogram(block, bins=256, range=(0, 256))
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                block_entropies.append(entropy)
        
        return block_entropies
    
    def _calculate_conditional_entropy(self, channel: np.ndarray) -> float:
        """Calculate conditional entropy H(X|Y) for adjacent pixels"""
        # Create pairs of adjacent pixels
        flat_channel = channel.flatten()
        x_values = flat_channel[:-1]
        y_values = flat_channel[1:]
        
        # Calculate joint histogram
        joint_hist, _, _ = np.histogram2d(x_values, y_values, bins=256, range=[[0, 256], [0, 256]])
        
        # Calculate probabilities
        joint_prob = joint_hist / np.sum(joint_hist)
        marginal_y = np.sum(joint_prob, axis=0)
        
        # Calculate conditional entropy
        conditional_entropy = 0.0
        for i in range(256):
            for j in range(256):
                if joint_prob[i, j] > 0 and marginal_y[j] > 0:
                    conditional_prob = joint_prob[i, j] / marginal_y[j]
                    conditional_entropy -= joint_prob[i, j] * np.log2(conditional_prob)
        
        return conditional_entropy
    
    def _frequency_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency domain characteristics"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i].astype(np.float32)
            
            # 2D FFT
            fft_result = fft2(channel)
            fft_shifted = fftshift(fft_result)
            magnitude_spectrum = np.abs(fft_shifted)
            power_spectrum = magnitude_spectrum ** 2
            
            # Frequency domain statistics
            dc_component = float(np.abs(fft_result[0, 0]))
            total_energy = float(np.sum(power_spectrum))
            high_freq_energy = self._calculate_high_freq_energy(power_spectrum)
            
            # Spectral centroid
            spectral_centroid = self._calculate_spectral_centroid(magnitude_spectrum)
            
            # Frequency distribution analysis
            freq_stats = self._analyze_frequency_distribution(magnitude_spectrum)
            
            results[channel_name] = {
                'dc_component': dc_component,
                'total_energy': total_energy,
                'high_freq_energy': float(high_freq_energy),
                'high_freq_ratio': float(high_freq_energy / total_energy) if total_energy > 0 else 0.0,
                'spectral_centroid': spectral_centroid,
                'freq_distribution': freq_stats
            }
        
        return results
    
    def _calculate_high_freq_energy(self, power_spectrum: np.ndarray) -> float:
        """Calculate energy in high frequency components"""
        height, width = power_spectrum.shape
        center_h, center_w = height // 2, width // 2
        
        # Define high frequency region (outer 25% of spectrum)
        radius = min(height, width) // 4
        
        total_high_freq = 0.0
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                if distance > radius:
                    total_high_freq += power_spectrum[i, j]
        
        return total_high_freq
    
    def _calculate_spectral_centroid(self, magnitude_spectrum: np.ndarray) -> Tuple[float, float]:
        """Calculate spectral centroid coordinates"""
        height, width = magnitude_spectrum.shape
        
        # Create coordinate matrices
        y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
        
        # Calculate weighted centroids
        total_magnitude = np.sum(magnitude_spectrum)
        if total_magnitude > 0:
            centroid_y = np.sum(y_coords * magnitude_spectrum) / total_magnitude
            centroid_x = np.sum(x_coords * magnitude_spectrum) / total_magnitude
        else:
            centroid_y = height / 2
            centroid_x = width / 2
        
        return (float(centroid_y), float(centroid_x))
    
    def _analyze_frequency_distribution(self, magnitude_spectrum: np.ndarray) -> Dict[str, float]:
        """Analyze distribution of frequency components"""
        # Convert to 1D for analysis
        freq_values = magnitude_spectrum.flatten()
        
        return {
            'mean': float(np.mean(freq_values)),
            'std': float(np.std(freq_values)),
            'skewness': float(stats.skew(freq_values)),
            'kurtosis': float(stats.kurtosis(freq_values)),
            'entropy': float(-np.sum((freq_values / np.sum(freq_values)) * 
                                   np.log2(freq_values / np.sum(freq_values) + 1e-10)))
        }
    
    def _texture_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture characteristics"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i]
            
            # Gradient analysis
            grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Edge detection
            edges = cv2.Canny(channel, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Local Binary Pattern-like analysis
            lbp_features = self._calculate_lbp_features(channel)
            
            # Co-occurrence matrix features
            glcm_features = self._calculate_glcm_features(channel)
            
            # Wavelet analysis (simplified)
            wavelet_features = self._calculate_wavelet_features(channel)
            
            results[channel_name] = {
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'gradient_max': float(np.max(gradient_magnitude)),
                'edge_density': float(edge_density),
                'lbp_features': lbp_features,
                'glcm_features': glcm_features,
                'wavelet_features': wavelet_features,
                'texture_energy': float(np.sum(gradient_magnitude**2)),
                'texture_homogeneity': float(self._calculate_homogeneity(channel))
            }
        
        return results
    
    def _calculate_lbp_features(self, channel: np.ndarray) -> Dict[str, float]:
        """Calculate Local Binary Pattern-like features"""
        height, width = channel.shape
        lbp_values = []
        
        # Simplified LBP calculation
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = channel[i, j]
                binary_string = ''
                
                # Check 8 neighbors
                neighbors = [
                    channel[i-1, j-1], channel[i-1, j], channel[i-1, j+1],
                    channel[i, j+1], channel[i+1, j+1], channel[i+1, j],
                    channel[i+1, j-1], channel[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp_values.append(int(binary_string, 2))
        
        if lbp_values:
            lbp_array = np.array(lbp_values)
            return {
                'lbp_mean': float(np.mean(lbp_array)),
                'lbp_std': float(np.std(lbp_array)),
                'lbp_entropy': float(self._calculate_entropy(lbp_array)),
                'lbp_uniformity': float(len(np.unique(lbp_array)) / 256.0)
            }
        else:
            return {'lbp_mean': 0.0, 'lbp_std': 0.0, 'lbp_entropy': 0.0, 'lbp_uniformity': 0.0}
    
    def _calculate_glcm_features(self, channel: np.ndarray) -> Dict[str, float]:
        """Calculate Gray Level Co-occurrence Matrix features"""
        # Simplified GLCM calculation
        # Reduce gray levels for computational efficiency
        normalized_channel = (channel // 16).astype(np.uint8)  # 16 gray levels
        
        # Calculate co-occurrence for horizontal direction (distance=1)
        height, width = normalized_channel.shape
        glcm = np.zeros((16, 16), dtype=np.float32)
        
        for i in range(height):
            for j in range(width - 1):
                val1 = normalized_channel[i, j]
                val2 = normalized_channel[i, j + 1]
                glcm[val1, val2] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        
        # Calculate texture features
        contrast = 0.0
        correlation = 0.0
        energy = 0.0
        homogeneity = 0.0
        
        mean_i = np.sum(np.arange(16).reshape(16, 1) * glcm)
        mean_j = np.sum(np.arange(16).reshape(1, 16) * glcm)
        std_i = np.sqrt(np.sum(((np.arange(16).reshape(16, 1) - mean_i) ** 2) * glcm))
        std_j = np.sqrt(np.sum(((np.arange(16).reshape(1, 16) - mean_j) ** 2) * glcm))
        
        for i in range(16):
            for j in range(16):
                contrast += ((i - j) ** 2) * glcm[i, j]
                energy += glcm[i, j] ** 2
                homogeneity += glcm[i, j] / (1 + abs(i - j))
                if std_i > 0 and std_j > 0:
                    correlation += ((i - mean_i) * (j - mean_j) * glcm[i, j]) / (std_i * std_j)
        
        return {
            'contrast': float(contrast),
            'correlation': float(correlation),
            'energy': float(energy),
            'homogeneity': float(homogeneity)
        }
    
    def _calculate_wavelet_features(self, channel: np.ndarray) -> Dict[str, float]:
        """Calculate simplified wavelet-like features"""
        # Simple high-pass and low-pass filtering as wavelet approximation
        kernel_high = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        kernel_low = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9
        
        high_freq = cv2.filter2D(channel.astype(np.float32), -1, kernel_high)
        low_freq = cv2.filter2D(channel.astype(np.float32), -1, kernel_low)
        
        return {
            'high_freq_energy': float(np.sum(high_freq ** 2)),
            'low_freq_energy': float(np.sum(low_freq ** 2)),
            'high_freq_mean': float(np.mean(np.abs(high_freq))),
            'low_freq_mean': float(np.mean(np.abs(low_freq))),
            'energy_ratio': float(np.sum(high_freq ** 2) / (np.sum(low_freq ** 2) + 1e-10))
        }
    
    def _calculate_homogeneity(self, channel: np.ndarray) -> float:
        """Calculate texture homogeneity"""
        # Calculate local variance
        kernel = np.ones((3, 3), np.float32) / 9
        mean_filtered = cv2.filter2D(channel.astype(np.float32), -1, kernel)
        variance = (channel.astype(np.float32) - mean_filtered) ** 2
        local_variance = cv2.filter2D(variance, -1, kernel)
        
        return np.mean(local_variance)
    
    def _color_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color characteristics and relationships"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        results = {
            'rgb_analysis': self._analyze_color_space(image, 'RGB'),
            'hsv_analysis': self._analyze_color_space(hsv, 'HSV'),
            'lab_analysis': self._analyze_color_space(lab, 'LAB'),
            'channel_correlations': self._calculate_channel_correlations(image),
            'color_distribution': self._analyze_color_distribution(image)
        }
        
        return results
    
    def _analyze_color_space(self, image: np.ndarray, color_space: str) -> Dict[str, Any]:
        """Analyze statistics for a specific color space"""
        if color_space == 'RGB':
            channel_names = ['red', 'green', 'blue']
        elif color_space == 'HSV':
            channel_names = ['hue', 'saturation', 'value']
        elif color_space == 'LAB':
            channel_names = ['lightness', 'a_channel', 'b_channel']
        else:
            channel_names = ['channel_0', 'channel_1', 'channel_2']
        
        results = {}
        
        for i, channel_name in enumerate(channel_names):
            channel = image[:, :, i]
            
            results[channel_name] = {
                'mean': float(np.mean(channel)),
                'std': float(np.std(channel)),
                'min': float(np.min(channel)),
                'max': float(np.max(channel)),
                'range': float(np.ptp(channel)),
                'skewness': float(stats.skew(channel.flatten())),
                'kurtosis': float(stats.kurtosis(channel.flatten())),
                'entropy': float(self._calculate_entropy(channel))
            }
        
        return results
    
    def _calculate_channel_correlations(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate correlations between color channels"""
        r, g, b = image[:, :, 0].flatten(), image[:, :, 1].flatten(), image[:, :, 2].flatten()
        
        corr_rg = float(np.corrcoef(r, g)[0, 1])
        corr_rb = float(np.corrcoef(r, b)[0, 1])
        corr_gb = float(np.corrcoef(g, b)[0, 1])
        
        return {
            'red_green_correlation': corr_rg if not np.isnan(corr_rg) else 0.0,
            'red_blue_correlation': corr_rb if not np.isnan(corr_rb) else 0.0,
            'green_blue_correlation': corr_gb if not np.isnan(corr_gb) else 0.0,
            'mean_correlation': float(np.mean([corr_rg, corr_rb, corr_gb]))
        }
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze overall color distribution characteristics"""
        # Convert to 1D color vectors
        pixels = image.reshape(-1, 3)
        
        # Color diversity (number of unique colors)
        unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
        total_pixels = len(pixels)
        color_diversity = unique_colors / total_pixels
        
        # Dominant colors analysis
        dominant_colors = self._find_dominant_colors(pixels, k=5)
        
        # Color variance
        color_variance = float(np.mean(np.var(pixels, axis=0)))
        
        return {
            'unique_colors': int(unique_colors),
            'color_diversity': float(color_diversity),
            'dominant_colors': dominant_colors,
            'color_variance': color_variance,
            'total_pixels': int(total_pixels)
        }
    
    def _find_dominant_colors(self, pixels: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Find dominant colors using simple clustering"""
        # Simple k-means clustering for dominant colors
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = []
            for i, color in enumerate(kmeans.cluster_centers_):
                count = np.sum(kmeans.labels_ == i)
                percentage = count / len(pixels) * 100
                colors.append({
                    'color': [int(c) for c in color],
                    'percentage': float(percentage),
                    'count': int(count)
                })
            
            return sorted(colors, key=lambda x: x['percentage'], reverse=True)
        except ImportError:
            # Fallback if sklearn not available
            return []
    
    def _noise_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise characteristics"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i].astype(np.float32)
            
            # Laplacian noise estimation
            laplacian = cv2.Laplacian(channel, cv2.CV_64F)
            noise_variance = float(np.var(laplacian))
            
            # High-frequency noise
            kernel_high = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            high_freq_response = cv2.filter2D(channel, -1, kernel_high)
            high_freq_energy = float(np.sum(high_freq_response ** 2))
            
            # Local noise estimation
            local_noise = self._estimate_local_noise(channel)
            
            # Signal-to-noise ratio estimation
            signal_power = float(np.var(channel))
            snr = signal_power / (noise_variance + 1e-10)
            
            results[channel_name] = {
                'noise_variance': noise_variance,
                'high_freq_energy': high_freq_energy,
                'local_noise_mean': float(np.mean(local_noise)),
                'local_noise_std': float(np.std(local_noise)),
                'snr_estimate': float(snr),
                'noise_ratio': float(noise_variance / (signal_power + 1e-10))
            }
        
        return results
    
    def _estimate_local_noise(self, channel: np.ndarray, window_size: int = 8) -> np.ndarray:
        """Estimate local noise using sliding window"""
        height, width = channel.shape
        noise_estimates = []
        
        for i in range(0, height - window_size + 1, window_size):
            for j in range(0, width - window_size + 1, window_size):
                window = channel[i:i+window_size, j:j+window_size]
                # Estimate noise as deviation from local mean
                local_mean = np.mean(window)
                noise = np.std(window - local_mean)
                noise_estimates.append(noise)
        
        return np.array(noise_estimates)
    
    def _correlation_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial correlations and dependencies"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i]
            
            # Autocorrelation analysis
            autocorr = self._calculate_autocorrelation(channel)
            
            # Mutual information between adjacent pixels
            mutual_info = self._calculate_mutual_information(channel)
            
            # Spatial correlation in different directions
            spatial_corr = self._calculate_spatial_correlations(channel)
            
            results[channel_name] = {
                'autocorrelation': autocorr,
                'mutual_information': mutual_info,
                'spatial_correlations': spatial_corr
            }
        
        return results
    
    def _calculate_autocorrelation(self, channel: np.ndarray) -> Dict[str, float]:
        """Calculate autocorrelation function"""
        flat_channel = channel.flatten()
        mean_val = np.mean(flat_channel)
        centered = flat_channel - mean_val
        
        # Calculate autocorrelation for different lags
        autocorr_values = []
        max_lag = min(100, len(centered) // 10)
        
        for lag in range(1, max_lag):
            if lag < len(centered):
                corr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorr_values.append(corr)
        
        if autocorr_values:
            return {
                'first_lag': float(autocorr_values[0]) if autocorr_values else 0.0,
                'mean_autocorr': float(np.mean(autocorr_values)),
                'decay_rate': float(autocorr_values[0] - autocorr_values[-1]) if len(autocorr_values) > 1 else 0.0
            }
        else:
            return {'first_lag': 0.0, 'mean_autocorr': 0.0, 'decay_rate': 0.0}
    
    def _calculate_mutual_information(self, channel: np.ndarray) -> float:
        """Calculate mutual information between adjacent pixels"""
        flat_channel = channel.flatten()
        x = flat_channel[:-1]
        y = flat_channel[1:]
        
        # Discretize for mutual information calculation
        x_discrete = (x // 16).astype(int)  # Reduce to 16 levels
        y_discrete = (y // 16).astype(int)
        
        try:
            mi = mutual_info_score(x_discrete, y_discrete)
            return float(mi)
        except:
            return 0.0
    
    def _calculate_spatial_correlations(self, channel: np.ndarray) -> Dict[str, float]:
        """Calculate spatial correlations in different directions"""
        height, width = channel.shape
        
        # Horizontal correlation
        if width > 1:
            h_corr = np.corrcoef(channel[:, :-1].flatten(), channel[:, 1:].flatten())[0, 1]
        else:
            h_corr = 0.0
        
        # Vertical correlation
        if height > 1:
            v_corr = np.corrcoef(channel[:-1, :].flatten(), channel[1:, :].flatten())[0, 1]
        else:
            v_corr = 0.0
        
        # Diagonal correlations
        if height > 1 and width > 1:
            d1_corr = np.corrcoef(channel[:-1, :-1].flatten(), channel[1:, 1:].flatten())[0, 1]
            d2_corr = np.corrcoef(channel[:-1, 1:].flatten(), channel[1:, :-1].flatten())[0, 1]
        else:
            d1_corr = d2_corr = 0.0
        
        return {
            'horizontal': float(h_corr) if not np.isnan(h_corr) else 0.0,
            'vertical': float(v_corr) if not np.isnan(v_corr) else 0.0,
            'diagonal_1': float(d1_corr) if not np.isnan(d1_corr) else 0.0,
            'diagonal_2': float(d2_corr) if not np.isnan(d2_corr) else 0.0
        }
    
    def _distribution_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical distributions"""
        results = {}
        
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name not in self.channels_to_analyze:
                continue
                
            channel = image[:, :, i].flatten()
            
            # Goodness of fit tests
            results[channel_name] = self._test_distributions(channel)
        
        return results
    
    def _test_distributions(self, data: np.ndarray) -> Dict[str, Any]:
        """Test data against various distributions"""
        # Kolmogorov-Smirnov test against uniform distribution
        uniform_ks_stat, uniform_p = stats.kstest(data, 'uniform')
        
        # Kolmogorov-Smirnov test against normal distribution
        # Normalize data for normal test
        normalized_data = (data - np.mean(data)) / np.std(data)
        normal_ks_stat, normal_p = stats.kstest(normalized_data, 'norm')
        
        # Anderson-Darling test for normality
        try:
            ad_stat, ad_critical, ad_significance = stats.anderson(normalized_data, dist='norm')
            ad_normal = {'statistic': float(ad_stat), 'critical_values': ad_critical.tolist()}
        except:
            ad_normal = {'statistic': 0.0, 'critical_values': []}
        
        # Shapiro-Wilk test (for smaller samples)
        if len(data) <= 5000:
            sw_stat, sw_p = stats.shapiro(data[:5000])  # Limit sample size
        else:
            sw_stat, sw_p = 0.0, 1.0
        
        return {
            'uniform_ks_test': {
                'statistic': float(uniform_ks_stat),
                'p_value': float(uniform_p)
            },
            'normal_ks_test': {
                'statistic': float(normal_ks_stat),
                'p_value': float(normal_p)
            },
            'anderson_darling_normal': ad_normal,
            'shapiro_wilk': {
                'statistic': float(sw_stat),
                'p_value': float(sw_p)
            }
        }
    
    def _calculate_anomaly_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate anomaly scores based on statistical analysis"""
        anomaly_scores = {}
        
        # Entropy-based anomalies
        entropy_scores = []
        for channel_data in results.get('entropy_analysis', {}).values():
            if isinstance(channel_data, dict):
                global_entropy = channel_data.get('global_entropy', 8.0)
                lsb_entropy = channel_data.get('lsb_entropy', 1.0)
                
                # LSB entropy should be close to 1.0 for natural images
                entropy_anomaly = abs(lsb_entropy - 1.0)
                entropy_scores.append(entropy_anomaly)
        
        anomaly_scores['entropy_anomaly'] = float(np.mean(entropy_scores)) if entropy_scores else 0.0
        
        # Histogram-based anomalies
        hist_scores = []
        for channel_data in results.get('histogram_analysis', {}).values():
            if isinstance(channel_data, dict):
                gap_ratio = channel_data.get('gap_ratio', 0.0)
                chi2_p = channel_data.get('chi2_uniform_p', 0.5)
                
                # High gap ratio or low p-value indicates anomaly
                hist_anomaly = gap_ratio + (1.0 - chi2_p)
                hist_scores.append(hist_anomaly)
        
        anomaly_scores['histogram_anomaly'] = float(np.mean(hist_scores)) if hist_scores else 0.0
        
        # Frequency domain anomalies
        freq_scores = []
        for channel_data in results.get('frequency_analysis', {}).values():
            if isinstance(channel_data, dict):
                high_freq_ratio = channel_data.get('high_freq_ratio', 0.0)
                freq_scores.append(high_freq_ratio)
        
        anomaly_scores['frequency_anomaly'] = float(np.mean(freq_scores)) if freq_scores else 0.0
        
        # Correlation anomalies
        corr_data = results.get('correlation_analysis', {})
        corr_scores = []
        for channel_data in corr_data.values():
            if isinstance(channel_data, dict):
                spatial_corr = channel_data.get('spatial_correlations', {})
                h_corr = spatial_corr.get('horizontal', 0.0)
                v_corr = spatial_corr.get('vertical', 0.0)
                
                # Very low correlation might indicate manipulation
                corr_anomaly = 2.0 - (abs(h_corr) + abs(v_corr))
                corr_scores.append(max(0.0, corr_anomaly))
        
        anomaly_scores['correlation_anomaly'] = float(np.mean(corr_scores)) if corr_scores else 0.0
        
        # Overall anomaly score
        individual_scores = [score for score in anomaly_scores.values() if score > 0]
        anomaly_scores['overall_anomaly'] = float(np.mean(individual_scores)) if individual_scores else 0.0
        
        return anomaly_scores
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return entropy
    
    def generate_statistical_report(self, results: Dict[str, Any], output_path: str):
        """Generate a detailed statistical analysis report"""
        try:
            # Create visualizations
            self._create_statistical_plots(results, output_path)
            
            # Generate summary statistics
            summary = self._create_statistical_summary(results)
            
            # Save detailed results
            import json
            with open(f"{output_path}_detailed.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save summary
            with open(f"{output_path}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Statistical report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate statistical report: {str(e)}")
    
    def _create_statistical_plots(self, results: Dict[str, Any], output_path: str):
        """Create visualization plots for statistical analysis"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Statistical Analysis Report', fontsize=16)
            
            # Plot histograms for each channel
            hist_data = results.get('histogram_analysis', {})
            for i, (channel, data) in enumerate(hist_data.items()):
                if i < 3 and isinstance(data, dict):
                    hist = data.get('histogram', [])
                    if hist:
                        axes[0, i].plot(hist)
                        axes[0, i].set_title(f'{channel.capitalize()} Channel Histogram')
                        axes[0, i].set_xlabel('Pixel Value')
                        axes[0, i].set_ylabel('Frequency')
            
            # Plot entropy analysis
            entropy_data = results.get('entropy_analysis', {})
            channel_names = list(entropy_data.keys())
            global_entropies = [entropy_data[ch].get('global_entropy', 0) for ch in channel_names if isinstance(entropy_data[ch], dict)]
            lsb_entropies = [entropy_data[ch].get('lsb_entropy', 0) for ch in channel_names if isinstance(entropy_data[ch], dict)]
            
            if global_entropies and lsb_entropies:
                x = range(len(channel_names))
                axes[1, 0].bar([i - 0.2 for i in x], global_entropies, 0.4, label='Global Entropy', alpha=0.7)
                axes[1, 0].bar([i + 0.2 for i in x], lsb_entropies, 0.4, label='LSB Entropy', alpha=0.7)
                axes[1, 0].set_title('Entropy Analysis')
                axes[1, 0].set_xlabel('Channel')
                axes[1, 0].set_ylabel('Entropy')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(channel_names)
                axes[1, 0].legend()
            
            # Plot anomaly scores
            anomaly_data = results.get('anomaly_scores', {})
            if anomaly_data:
                anomaly_names = list(anomaly_data.keys())
                anomaly_values = list(anomaly_data.values())
                
                axes[1, 1].bar(anomaly_names, anomaly_values, alpha=0.7)
                axes[1, 1].set_title('Anomaly Scores')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Plot frequency analysis
            freq_data = results.get('frequency_analysis', {})
            if freq_data:
                channel_names = list(freq_data.keys())
                high_freq_ratios = [freq_data[ch].get('high_freq_ratio', 0) for ch in channel_names if isinstance(freq_data[ch], dict)]
                
                if high_freq_ratios:
                    axes[1, 2].bar(channel_names, high_freq_ratios, alpha=0.7)
                    axes[1, 2].set_title('High Frequency Energy Ratio')
                    axes[1, 2].set_xlabel('Channel')
                    axes[1, 2].set_ylabel('Ratio')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create statistical plots: {str(e)}")
    
    def _create_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of statistical analysis results"""
        summary = {
            'image_path': results.get('image_path', ''),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'image_properties': results.get('image_properties', {}),
            'key_findings': {},
            'anomaly_assessment': {},
            'recommendations': []
        }
        
        # Extract key findings
        anomaly_scores = results.get('anomaly_scores', {})
        summary['anomaly_assessment'] = {
            'overall_anomaly_score': anomaly_scores.get('overall_anomaly', 0.0),
            'entropy_suspicious': anomaly_scores.get('entropy_anomaly', 0.0) > 0.5,
            'histogram_suspicious': anomaly_scores.get('histogram_anomaly', 0.0) > 0.5,
            'frequency_suspicious': anomaly_scores.get('frequency_anomaly', 0.0) > 0.3,
            'correlation_suspicious': anomaly_scores.get('correlation_anomaly', 0.0) > 0.5
        }
        
        # Generate recommendations
        if summary['anomaly_assessment']['overall_anomaly_score'] > 0.4:
            summary['recommendations'].append("High anomaly score detected - further investigation recommended")
        
        if summary['anomaly_assessment']['entropy_suspicious']:
            summary['recommendations'].append("LSB entropy anomaly detected - possible LSB steganography")
        
        if summary['anomaly_assessment']['histogram_suspicious']:
            summary['recommendations'].append("Histogram anomalies detected - possible pixel value manipulation")
        
        if summary['anomaly_assessment']['frequency_suspicious']:
            summary['recommendations'].append("Frequency domain anomalies detected - possible DCT-based steganography")
        
        if not summary['recommendations']:
            summary['recommendations'].append("No significant anomalies detected - image appears normal")
        
        return summary