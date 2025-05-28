#!/usr/bin/env python3
"""
StegAnalysis Suite - Benchmark Tool
Performance benchmarking and algorithm comparison
"""

import os
import json
import logging
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import argparse
from datetime import datetime
import psutil
import threading
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import analysis modules
import sys
sys.path.append('..')
from modules.detection import DetectionEngine
from modules.ml_detection import MLDetectionEngine
from modules.analysis import StatisticalAnalyzer
from modules.forensics import ForensicAnalyzer


class BenchmarkSuite:
    """Comprehensive benchmarking suite for steganography detection algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Benchmark configuration
        self.benchmark_config = config.get('benchmark', {})
        self.output_dir = Path(self.benchmark_config.get('output_dir', 'benchmarks'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.iterations = self.benchmark_config.get('iterations', 5)
        self.warmup_iterations = self.benchmark_config.get('warmup_iterations', 2)
        self.memory_monitoring = self.benchmark_config.get('memory_monitoring', True)
        self.cpu_monitoring = self.benchmark_config.get('cpu_monitoring', True)
        
        # Initialize engines
        self.detection_engine = DetectionEngine(config)
        self.ml_engine = MLDetectionEngine(config)
        self.statistical_analyzer = StatisticalAnalyzer(config)
        self.forensic_analyzer = ForensicAnalyzer(config)
        
        # Results storage
        self.benchmark_results = {}
        self.system_monitor = SystemMonitor() if self.memory_monitoring or self.cpu_monitoring else None
        
    def run_comprehensive_benchmark(self, 
                                   test_images: List[str],
                                   output_prefix: str = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on all algorithms
        
        Args:
            test_images: List of test image paths
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with benchmark results
        """
        if output_prefix is None:
            output_prefix = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting comprehensive benchmark with {len(test_images)} images")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'test_configuration': {
                'num_images': len(test_images),
                'iterations': self.iterations,
                'warmup_iterations': self.warmup_iterations
            },
            'algorithm_results': {},
            'comparison_analysis': {},
            'system_metrics': {}
        }
        
        # Algorithms to benchmark
        algorithms = {
            'lsb_detection': self._benchmark_lsb,
            'chi_square': self._benchmark_chi_square,
            'sample_pairs': self._benchmark_sample_pairs,
            'ml_cnn': self._benchmark_ml_cnn,
            'ml_svm': self._benchmark_ml_svm,
            'statistical_analysis': self._benchmark_statistical,
            'forensic_analysis': self._benchmark_forensic
        }
        
        # Run benchmarks for each algorithm
        for algo_name, benchmark_func in algorithms.items():
            self.logger.info(f"Benchmarking {algo_name}...")
            
            try:
                # Start system monitoring
                if self.system_monitor:
                    self.system_monitor.start_monitoring()
                
                # Run benchmark
                algo_results = self._run_algorithm_benchmark(
                    benchmark_func, test_images, algo_name
                )
                
                # Stop system monitoring
                if self.system_monitor:
                    system_metrics = self.system_monitor.stop_monitoring()
                    algo_results['system_metrics'] = system_metrics
                
                benchmark_results['algorithm_results'][algo_name] = algo_results
                
            except Exception as e:
                self.logger.error(f"Error benchmarking {algo_name}: {str(e)}")
                benchmark_results['algorithm_results'][algo_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generate comparison analysis
        benchmark_results['comparison_analysis'] = self._analyze_benchmark_results(
            benchmark_results['algorithm_results']
        )
        
        # Save results
        self._save_benchmark_results(benchmark_results, output_prefix)
        
        # Generate visualizations
        self._generate_benchmark_plots(benchmark_results, output_prefix)
        
        self.logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    def benchmark_accuracy(self, 
                          test_dataset: Dict[str, List[str]],
                          output_prefix: str = None) -> Dict[str, Any]:
        """
        Benchmark detection accuracy on labeled dataset
        
        Args:
            test_dataset: Dictionary with 'clean' and 'steganographic' image lists
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with accuracy benchmark results
        """
        if output_prefix is None:
            output_prefix = f"accuracy_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info("Starting accuracy benchmark")
        
        accuracy_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'clean_images': len(test_dataset.get('clean', [])),
                'steganographic_images': len(test_dataset.get('steganographic', []))
            },
            'algorithm_accuracy': {},
            'confusion_matrices': {},
            'roc_analysis': {}
        }
        
        # Prepare ground truth
        all_images = []
        ground_truth = []
        
        for img_path in test_dataset.get('clean', []):
            all_images.append(img_path)
            ground_truth.append(0)  # Clean
        
        for img_path in test_dataset.get('steganographic', []):
            all_images.append(img_path)
            ground_truth.append(1)  # Steganographic
        
        # Test each algorithm
        algorithms = ['lsb', 'chi_square', 'sample_pairs', 'ml_cnn', 'ml_svm']
        
        for algo in algorithms:
            self.logger.info(f"Testing accuracy for {algo}")
            
            try:
                predictions = []
                confidences = []
                
                for img_path in tqdm(all_images, desc=f"Processing {algo}"):
                    pred, conf = self._get_algorithm_prediction(img_path, algo)
                    predictions.append(pred)
                    confidences.append(conf)
                
                # Calculate accuracy metrics
                accuracy_metrics = self._calculate_accuracy_metrics(
                    ground_truth, predictions, confidences
                )
                
                accuracy_results['algorithm_accuracy'][algo] = accuracy_metrics
                
            except Exception as e:
                self.logger.error(f"Error testing {algo}: {str(e)}")
                accuracy_results['algorithm_accuracy'][algo] = {'error': str(e)}
        
        # Generate ROC curves
        accuracy_results['roc_analysis'] = self._generate_roc_analysis(
            ground_truth, accuracy_results['algorithm_accuracy']
        )
        
        # Save accuracy results
        self._save_accuracy_results(accuracy_results, output_prefix)
        
        return accuracy_results
    
    def benchmark_scalability(self, 
                             image_counts: List[int],
                             test_images: List[str],
                             output_prefix: str = None) -> Dict[str, Any]:
        """
        Benchmark scalability with different image counts
        
        Args:
            image_counts: List of image counts to test
            test_images: Pool of test images
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with scalability results
        """
        if output_prefix is None:
            output_prefix = f"scalability_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info("Starting scalability benchmark")
        
        scalability_results = {
            'timestamp': datetime.now().isoformat(),
            'image_counts': image_counts,
            'algorithm_scalability': {}
        }
        
        algorithms = ['lsb', 'chi_square', 'sample_pairs', 'statistical_analysis']
        
        for algo in algorithms:
            self.logger.info(f"Testing scalability for {algo}")
            
            algo_scalability = {}
            
            for count in image_counts:
                # Select subset of images
                test_subset = test_images[:min(count, len(test_images))]
                
                # Measure processing time
                start_time = time.time()
                
                try:
                    for img_path in test_subset:
                        self._run_single_algorithm(img_path, algo)
                    
                    processing_time = time.time() - start_time
                    
                    algo_scalability[count] = {
                        'processing_time': processing_time,
                        'images_per_second': len(test_subset) / processing_time,
                        'time_per_image': processing_time / len(test_subset)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in scalability test for {algo} with {count} images: {str(e)}")
                    algo_scalability[count] = {'error': str(e)}
            
            scalability_results['algorithm_scalability'][algo] = algo_scalability
        
        # Analyze scalability trends
        scalability_results['trend_analysis'] = self._analyze_scalability_trends(
            scalability_results['algorithm_scalability']
        )
        
        # Save and visualize results
        self._save_scalability_results(scalability_results, output_prefix)
        self._plot_scalability_results(scalability_results, output_prefix)
        
        return scalability_results
    
    def benchmark_memory_usage(self, 
                              test_images: List[str],
                              output_prefix: str = None) -> Dict[str, Any]:
        """
        Benchmark memory usage patterns
        
        Args:
            test_images: List of test image paths
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with memory usage results
        """
        if output_prefix is None:
            output_prefix = f"memory_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info("Starting memory usage benchmark")
        
        memory_results = {
            'timestamp': datetime.now().isoformat(),
            'algorithm_memory': {}
        }
        
        algorithms = ['lsb', 'chi_square', 'sample_pairs', 'ml_cnn', 'statistical_analysis']
        
        for algo in algorithms:
            self.logger.info(f"Measuring memory usage for {algo}")
            
            try:
                memory_monitor = MemoryProfiler()
                memory_monitor.start()
                
                # Process test images
                for img_path in test_images[:10]:  # Limit for memory testing
                    self._run_single_algorithm(img_path, algo)
                
                memory_stats = memory_monitor.stop()
                memory_results['algorithm_memory'][algo] = memory_stats
                
            except Exception as e:
                self.logger.error(f"Error measuring memory for {algo}: {str(e)}")
                memory_results['algorithm_memory'][algo] = {'error': str(e)}
        
        # Save memory results
        self._save_memory_results(memory_results, output_prefix)
        
        return memory_results
    
    def _run_algorithm_benchmark(self, 
                                benchmark_func: Callable,
                                test_images: List[str],
                                algo_name: str) -> Dict[str, Any]:
        """Run benchmark for a specific algorithm"""
        
        # Warmup runs
        self.logger.debug(f"Running {self.warmup_iterations} warmup iterations for {algo_name}")
        for _ in range(self.warmup_iterations):
            try:
                benchmark_func(test_images[:min(5, len(test_images))])
            except Exception as e:
                self.logger.debug(f"Warmup error for {algo_name}: {str(e)}")
        
        # Benchmark runs
        timing_results = []
        success_counts = []
        error_counts = []
        
        for iteration in range(self.iterations):
            self.logger.debug(f"Running iteration {iteration + 1}/{self.iterations} for {algo_name}")
            
            start_time = time.time()
            successes = 0
            errors = 0
            
            try:
                for img_path in test_images:
                    try:
                        benchmark_func([img_path])
                        successes += 1
                    except Exception:
                        errors += 1
                
                end_time = time.time()
                timing_results.append(end_time - start_time)
                success_counts.append(successes)
                error_counts.append(errors)
                
            except Exception as e:
                self.logger.error(f"Benchmark iteration failed for {algo_name}: {str(e)}")
        
        # Calculate statistics
        if timing_results:
            results = {
                'algorithm': algo_name,
                'iterations': self.iterations,
                'total_images': len(test_images),
                'timing_stats': {
                    'mean_time': statistics.mean(timing_results),
                    'median_time': statistics.median(timing_results),
                    'min_time': min(timing_results),
                    'max_time': max(timing_results),
                    'std_time': statistics.stdev(timing_results) if len(timing_results) > 1 else 0
                },
                'throughput_stats': {
                    'mean_images_per_second': len(test_images) / statistics.mean(timing_results),
                    'mean_time_per_image': statistics.mean(timing_results) / len(test_images)
                },
                'reliability_stats': {
                    'mean_success_rate': statistics.mean(success_counts) / len(test_images),
                    'mean_error_rate': statistics.mean(error_counts) / len(test_images)
                },
                'raw_data': {
                    'timing_results': timing_results,
                    'success_counts': success_counts,
                    'error_counts': error_counts
                }
            }
        else:
            results = {
                'algorithm': algo_name,
                'error': 'No successful benchmark runs',
                'status': 'failed'
            }
        
        return results
    
    # Algorithm-specific benchmark functions
    def _benchmark_lsb(self, test_images: List[str]):
        """Benchmark LSB detection"""
        for img_path in test_images:
            self.detection_engine.detect_lsb(img_path)
    
    def _benchmark_chi_square(self, test_images: List[str]):
        """Benchmark Chi-square detection"""
        for img_path in test_images:
            self.detection_engine.detect_chi_square(img_path)
    
    def _benchmark_sample_pairs(self, test_images: List[str]):
        """Benchmark Sample Pairs detection"""
        for img_path in test_images:
            self.detection_engine.detect_sample_pairs(img_path)
    
    def _benchmark_ml_cnn(self, test_images: List[str]):
        """Benchmark ML CNN detection"""
        for img_path in test_images:
            try:
                self.ml_engine.predict_cnn(img_path)
            except Exception as e:
                # ML models might not be available
                raise e
    
    def _benchmark_ml_svm(self, test_images: List[str]):
        """Benchmark ML SVM detection"""
        for img_path in test_images:
            try:
                self.ml_engine.predict_svm(img_path)
            except Exception as e:
                # ML models might not be available
                raise e
    
    def _benchmark_statistical(self, test_images: List[str]):
        """Benchmark statistical analysis"""
        for img_path in test_images:
            self.statistical_analyzer.analyze_image(img_path)
    
    def _benchmark_forensic(self, test_images: List[str]):
        """Benchmark forensic analysis"""
        for img_path in test_images:
            self.forensic_analyzer.analyze_image(img_path)
    
    def _get_algorithm_prediction(self, img_path: str, algorithm: str) -> tuple:
        """Get prediction and confidence from specific algorithm"""
        try:
            if algorithm == 'lsb':
                result = self.detection_engine.detect_lsb(img_path)
                return result.get('detected', False), result.get('confidence', 0.5)
            elif algorithm == 'chi_square':
                result = self.detection_engine.detect_chi_square(img_path)
                return result.get('detected', False), result.get('confidence', 0.5)
            elif algorithm == 'sample_pairs':
                result = self.detection_engine.detect_sample_pairs(img_path)
                return result.get('detected', False), result.get('confidence', 0.5)
            elif algorithm == 'ml_cnn':
                result = self.ml_engine.predict_cnn(img_path)
                return result.get('prediction', 0) == 1, result.get('confidence', 0.5)
            elif algorithm == 'ml_svm':
                result = self.ml_engine.predict_svm(img_path)
                return result.get('prediction', 0) == 1, result.get('confidence', 0.5)
            else:
                return False, 0.5
        except Exception:
            return False, 0.0
    
    def _run_single_algorithm(self, img_path: str, algorithm: str):
        """Run single algorithm on image"""
        self._get_algorithm_prediction(img_path, algorithm)
    
    def _calculate_accuracy_metrics(self, ground_truth: List[int], 
                                   predictions: List[int], 
                                   confidences: List[float]) -> Dict[str, Any]:
        """Calculate accuracy metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                    f1_score, confusion_matrix, roc_auc_score)
        
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, zero_division=0),
            'recall': recall_score(ground_truth, predictions, zero_division=0),
            'f1_score': f1_score(ground_truth, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(ground_truth, predictions).tolist()
        }
        
        try:
            metrics['auc_score'] = roc_auc_score(ground_truth, confidences)
        except ValueError:
            metrics['auc_score'] = 0.0
        
        return metrics
    
    def _analyze_benchmark_results(self, algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare benchmark results"""
        comparison = {
            'performance_ranking': [],
            'speed_comparison': {},
            'reliability_comparison': {},
            'efficiency_analysis': {}
        }
        
        # Extract timing data
        timing_data = {}
        for algo, results in algorithm_results.items():
            if 'timing_stats' in results:
                timing_data[algo] = results['timing_stats']['mean_time']
        
        # Performance ranking by speed
        sorted_algos = sorted(timing_data.items(), key=lambda x: x[1])
        comparison['performance_ranking'] = [
            {'algorithm': algo, 'mean_time': time, 'rank': i+1}
            for i, (algo, time) in enumerate(sorted_algos)
        ]
        
        # Speed comparison
        if timing_data:
            fastest_time = min(timing_data.values())
            comparison['speed_comparison'] = {
                algo: {
                    'mean_time': time,
                    'relative_speed': fastest_time / time,
                    'speed_factor': time / fastest_time
                }
                for algo, time in timing_data.items()
            }
        
        return comparison
    
    def _analyze_scalability_trends(self, scalability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scalability trends"""
        trends = {}
        
        for algo, data in scalability_data.items():
            if isinstance(data, dict) and 'error' not in data:
                # Extract time data points
                counts = []
                times = []
                
                for count, metrics in data.items():
                    if isinstance(metrics, dict) and 'processing_time' in metrics:
                        counts.append(int(count))
                        times.append(metrics['processing_time'])
                
                if len(counts) > 1:
                    # Calculate linear regression for trend
                    x = np.array(counts)
                    y = np.array(times)
                    
                    if len(x) > 1:
                        coeffs = np.polyfit(x, y, 1)
                        trends[algo] = {
                            'slope': coeffs[0],
                            'intercept': coeffs[1],
                            'complexity': 'linear' if abs(coeffs[0]) < 0.01 else 'super-linear'
                        }
        
        return trends
    
    def _generate_roc_analysis(self, ground_truth: List[int], 
                              algorithm_accuracy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ROC curve analysis"""
        roc_data = {}
        
        # This would generate ROC curves for each algorithm
        # Implementation depends on having confidence scores for each prediction
        
        return roc_data
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'platform': os.name,
                'python_version': sys.version
            }
        except Exception:
            return {'error': 'Could not gather system info'}
    
    def _save_benchmark_results(self, results: Dict[str, Any], output_prefix: str):
        """Save benchmark results to file"""
        try:
            output_path = self.output_dir / f"{output_prefix}_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Benchmark results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {str(e)}")
    
    def _save_accuracy_results(self, results: Dict[str, Any], output_prefix: str):
        """Save accuracy results to file"""
        try:
            output_path = self.output_dir / f"{output_prefix}_accuracy.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving accuracy results: {str(e)}")
    
    def _save_scalability_results(self, results: Dict[str, Any], output_prefix: str):
        """Save scalability results to file"""
        try:
            output_path = self.output_dir / f"{output_prefix}_scalability.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving scalability results: {str(e)}")
    
    def _save_memory_results(self, results: Dict[str, Any], output_prefix: str):
        """Save memory results to file"""
        try:
            output_path = self.output_dir / f"{output_prefix}_memory.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving memory results: {str(e)}")
    
    def _generate_benchmark_plots(self, results: Dict[str, Any], output_prefix: str):
        """Generate benchmark visualization plots"""
        try:
            # Performance comparison plot
            self._plot_performance_comparison(results, output_prefix)
            
            # Timing distribution plot
            self._plot_timing_distributions(results, output_prefix)
            
        except Exception as e:
            self.logger.error(f"Error generating benchmark plots: {str(e)}")
    
    def _plot_performance_comparison(self, results: Dict[str, Any], output_prefix: str):
        """Plot performance comparison"""
        try:
            algorithm_results = results.get('algorithm_results', {})
            
            algorithms = []
            mean_times = []
            
            for algo, data in algorithm_results.items():
                if 'timing_stats' in data:
                    algorithms.append(algo)
                    mean_times.append(data['timing_stats']['mean_time'])
            
            if algorithms and mean_times:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(algorithms, mean_times)
                plt.title('Algorithm Performance Comparison')
                plt.xlabel('Algorithm')
                plt.ylabel('Mean Processing Time (seconds)')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, time in zip(bars, mean_times):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{time:.3f}s', ha='center', va='bottom')
                
                plt.tight_layout()
                plot_path = self.output_dir / f"{output_prefix}_performance.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error plotting performance comparison: {str(e)}")
    
    def _plot_timing_distributions(self, results: Dict[str, Any], output_prefix: str):
        """Plot timing distributions"""
        try:
            algorithm_results = results.get('algorithm_results', {})
            
            plt.figure(figsize=(12, 8))
            
            timing_data = []
            for algo, data in algorithm_results.items():
                if 'raw_data' in data and 'timing_results' in data['raw_data']:
                    for time_val in data['raw_data']['timing_results']:
                        timing_data.append({'Algorithm': algo, 'Time': time_val})
            
            if timing_data:
                df = pd.DataFrame(timing_data)
                sns.boxplot(data=df, x='Algorithm', y='Time')
                plt.title('Algorithm Timing Distributions')
                plt.xlabel('Algorithm')
                plt.ylabel('Processing Time (seconds)')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plot_path = self.output_dir / f"{output_prefix}_distributions.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error plotting timing distributions: {str(e)}")
    
    def _plot_scalability_results(self, results: Dict[str, Any], output_prefix: str):
        """Plot scalability results"""
        try:
            scalability_data = results.get('algorithm_scalability', {})
            
            plt.figure(figsize=(10, 6))
            
            for algo, data in scalability_data.items():
                if isinstance(data, dict):
                    counts = []
                    times = []
                    
                    for count, metrics in data.items():
                        if isinstance(metrics, dict) and 'processing_time' in metrics:
                            counts.append(int(count))
                            times.append(metrics['processing_time'])
                    
                    if counts and times:
                        plt.plot(counts, times, marker='o', label=algo)
            
            plt.title('Algorithm Scalability')
            plt.xlabel('Number of Images')
            plt.ylabel('Processing Time (seconds)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{output_prefix}_scalability.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting scalability results: {str(e)}")


class SystemMonitor:
    """Monitor system resources during benchmarking"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.timestamps.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected data"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return {
            'cpu_usage': {
                'mean': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max': max(self.cpu_usage) if self.cpu_usage else 0,
                'min': min(self.cpu_usage) if self.cpu_usage else 0,
                'samples': len(self.cpu_usage)
            },
            'memory_usage': {
                'mean': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'max': max(self.memory_usage) if self.memory_usage else 0,
                'min': min(self.memory_usage) if self.memory_usage else 0,
                'samples': len(self.memory_usage)
            },
            'duration_seconds': len(self.timestamps) * 0.5 if self.timestamps else 0
        }
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            try:
                # Sample every 0.5 seconds
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_info.percent)
                self.timestamps.append(time.time())
                
                time.sleep(0.5)
                
            except Exception:
                break


class MemoryProfiler:
    """Profile memory usage for specific operations"""
    
    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.end_memory = 0
    
    def start(self):
        """Start memory profiling"""
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        except Exception:
            self.start_memory = 0
            self.peak_memory = 0
    
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return memory statistics"""
        try:
            process = psutil.Process()
            self.end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'start_memory_mb': self.start_memory,
                'end_memory_mb': self.end_memory,
                'peak_memory_mb': self.peak_memory,
                'memory_delta_mb': self.end_memory - self.start_memory
            }
        except Exception:
            return {
                'start_memory_mb': 0,
                'end_memory_mb': 0,
                'peak_memory_mb': 0,
                'memory_delta_mb': 0
            }


def create_test_dataset(base_dir: str) -> Dict[str, List[str]]:
    """Create test dataset structure for benchmarking"""
    base_path = Path(base_dir)
    
    test_dataset = {
        'clean': [],
        'steganographic': []
    }
    
    # Find clean images
    clean_dir = base_path / "clean"
    if clean_dir.exists():
        for pattern in ['*.jpg', '*.png', '*.bmp']:
            test_dataset['clean'].extend([str(p) for p in clean_dir.glob(pattern)])
    
    # Find steganographic images
    stego_dir = base_path / "steganographic"
    if stego_dir.exists():
        for pattern in ['*.jpg', '*.png', '*.bmp']:
            test_dataset['steganographic'].extend([str(p) for p in stego_dir.glob(pattern)])
    
    return test_dataset


def main():
    """Command-line interface for benchmarking"""
    parser = argparse.ArgumentParser(description="Benchmark steganography detection algorithms")
    parser.add_argument('test_dir', help='Directory containing test images')
    parser.add_argument('--benchmark-type', choices=['performance', 'accuracy', 'scalability', 'memory', 'all'],
                       default='performance', help='Type of benchmark to run')
    parser.add_argument('--output-prefix', help='Prefix for output files')
    parser.add_argument('--iterations', type=int, default=5, help='Number of benchmark iterations')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--algorithms', nargs='+', 
                       help='Specific algorithms to benchmark')
    parser.add_argument('--image-counts', nargs='+', type=int,
                       default=[10, 25, 50, 100], help='Image counts for scalability testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = {
        'benchmark': {
            'output_dir': 'benchmarks',
            'iterations': args.iterations,
            'warmup_iterations': 2,
            'memory_monitoring': True,
            'cpu_monitoring': True
        },
        'detection': {
            'algorithms': ['lsb', 'chi_square', 'sample_pairs']
        },
        'ml_models': {
            'enabled': True,
            'models_dir': 'models/trained'
        },
        'statistical_analysis': {
            'enabled': True
        },
        'forensic_analysis': {
            'enabled': True
        }
    }
    
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(config)
    
    # Prepare test data
    if Path(args.test_dir).exists():
        test_dataset = create_test_dataset(args.test_dir)
        all_test_images = test_dataset['clean'] + test_dataset['steganographic']
        
        if not all_test_images:
            print(f"No test images found in {args.test_dir}")
            return
        
        print(f"Found {len(all_test_images)} test images")
        print(f"Clean: {len(test_dataset['clean'])}, Steganographic: {len(test_dataset['steganographic'])}")
    else:
        print(f"Test directory not found: {args.test_dir}")
        return
    
    # Run benchmarks
    if args.benchmark_type == 'performance' or args.benchmark_type == 'all':
        print("Running performance benchmark...")
        results = benchmark_suite.run_comprehensive_benchmark(
            all_test_images[:50],  # Limit for performance testing
            args.output_prefix
        )
        print(f"Performance benchmark completed!")
        
        # Print summary
        if 'comparison_analysis' in results:
            ranking = results['comparison_analysis'].get('performance_ranking', [])
            if ranking:
                print("\nPerformance Ranking (by speed):")
                for rank_info in ranking:
                    print(f"  {rank_info['rank']}. {rank_info['algorithm']}: {rank_info['mean_time']:.3f}s")
    
    if args.benchmark_type == 'accuracy' or args.benchmark_type == 'all':
        if test_dataset['clean'] and test_dataset['steganographic']:
            print("Running accuracy benchmark...")
            accuracy_results = benchmark_suite.benchmark_accuracy(
                test_dataset,
                args.output_prefix
            )
            print("Accuracy benchmark completed!")
            
            # Print accuracy summary
            print("\nAccuracy Results:")
            for algo, metrics in accuracy_results.get('algorithm_accuracy', {}).items():
                if 'accuracy' in metrics:
                    print(f"  {algo}: {metrics['accuracy']:.3f} accuracy, {metrics['f1_score']:.3f} F1-score")
        else:
            print("Accuracy benchmark requires both clean and steganographic images")
    
    if args.benchmark_type == 'scalability' or args.benchmark_type == 'all':
        print("Running scalability benchmark...")
        scalability_results = benchmark_suite.benchmark_scalability(
            args.image_counts,
            all_test_images,
            args.output_prefix
        )
        print("Scalability benchmark completed!")
    
    if args.benchmark_type == 'memory' or args.benchmark_type == 'all':
        print("Running memory usage benchmark...")
        memory_results = benchmark_suite.benchmark_memory_usage(
            all_test_images[:20],  # Limit for memory testing
            args.output_prefix
        )
        print("Memory benchmark completed!")
    
    print(f"\nBenchmark results saved to: benchmarks/")


if __name__ == "__main__":
    main()