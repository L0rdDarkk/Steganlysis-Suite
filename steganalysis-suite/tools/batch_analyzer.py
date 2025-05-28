#!/usr/bin/env python3
"""
StegAnalysis Suite - Batch Analyzer
Tool for batch processing and analysis of multiple images
"""

import os
import json
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import argparse
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

import yaml
import pandas as pd
from tqdm import tqdm

# Import analysis modules
import sys
sys.path.append('..')
from modules.detection import DetectionEngine
from modules.ml_detection import MLDetectionEngine
from modules.analysis import StatisticalAnalyzer
from modules.forensics import ForensicAnalyzer
from modules.reports import ReportGenerator


class BatchAnalyzer:
    """Batch processing and analysis of steganographic images"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis engines
        self.detection_engine = DetectionEngine(self.config)
        self.ml_engine = MLDetectionEngine(self.config)
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.forensic_analyzer = ForensicAnalyzer(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Batch processing settings
        self.batch_config = self.config.get('batch_processing', {})
        self.max_workers = self.batch_config.get('max_workers', multiprocessing.cpu_count())
        self.chunk_size = self.batch_config.get('chunk_size', 10)
        self.use_multiprocessing = self.batch_config.get('use_multiprocessing', True)
        
        # Output settings
        self.output_dir = Path(self.batch_config.get('output_dir', 'reports/batch'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_callback = None
        self.results_cache = {}
        
    def analyze_directory(self, 
                         input_dir: str, 
                         output_prefix: str = None,
                         recursive: bool = True,
                         file_patterns: List[str] = None,
                         progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Analyze all images in a directory
        
        Args:
            input_dir: Directory containing images to analyze
            output_prefix: Prefix for output files
            recursive: Whether to search subdirectories
            file_patterns: List of file patterns to match
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary with batch analysis results
        """
        self.progress_callback = progress_callback
        
        if output_prefix is None:
            output_prefix = f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting batch analysis of directory: {input_dir}")
        
        # Find images to process
        image_files = self._find_images(input_dir, recursive, file_patterns)
        
        if not image_files:
            self.logger.warning("No images found to process")
            return {'error': 'No images found'}
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        start_time = time.time()
        results = self._process_images_batch(image_files)
        processing_time = time.time() - start_time
        
        # Compile batch results
        batch_results = self._compile_batch_results(results, processing_time)
        
        # Generate reports
        report_files = self._generate_batch_reports(batch_results, output_prefix)
        
        # Save detailed results
        self._save_detailed_results(results, output_prefix)
        
        batch_results['report_files'] = report_files
        batch_results['output_prefix'] = output_prefix
        
        self.logger.info(f"Batch analysis completed in {processing_time:.2f} seconds")
        return batch_results
    
    def analyze_file_list(self, 
                         file_list: List[str], 
                         output_prefix: str = None,
                         progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Analyze a specific list of image files
        
        Args:
            file_list: List of image file paths
            output_prefix: Prefix for output files
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary with batch analysis results
        """
        self.progress_callback = progress_callback
        
        if output_prefix is None:
            output_prefix = f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting batch analysis of {len(file_list)} files")
        
        # Validate files exist
        valid_files = [f for f in file_list if Path(f).exists()]
        if len(valid_files) != len(file_list):
            self.logger.warning(f"{len(file_list) - len(valid_files)} files not found")
        
        if not valid_files:
            return {'error': 'No valid files to process'}
        
        # Process images
        start_time = time.time()
        results = self._process_images_batch(valid_files)
        processing_time = time.time() - start_time
        
        # Compile batch results
        batch_results = self._compile_batch_results(results, processing_time)
        
        # Generate reports
        report_files = self._generate_batch_reports(batch_results, output_prefix)
        
        # Save detailed results
        self._save_detailed_results(results, output_prefix)
        
        batch_results['report_files'] = report_files
        batch_results['output_prefix'] = output_prefix
        
        self.logger.info(f"Batch analysis completed in {processing_time:.2f} seconds")
        return batch_results
    
    def compare_methods(self, 
                       input_dir: str, 
                       methods: List[str] = None,
                       output_prefix: str = None) -> Dict[str, Any]:
        """
        Compare different detection methods on the same dataset
        
        Args:
            input_dir: Directory containing test images
            methods: List of detection methods to compare
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with comparison results
        """
        if methods is None:
            methods = ['lsb', 'chi_square', 'sample_pairs', 'ml_cnn', 'ml_svm']
        
        if output_prefix is None:
            output_prefix = f"method_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting method comparison with {len(methods)} methods")
        
        # Find images
        image_files = self._find_images(input_dir, recursive=True)
        
        if not image_files:
            return {'error': 'No images found'}
        
        # Run each method separately
        comparison_results = {}
        
        for method in methods:
            self.logger.info(f"Running method: {method}")
            
            # Temporarily modify config to use only this method
            original_config = self.config.copy()
            self.config['detection']['algorithms'] = [method]
            
            # Process with this method
            method_results = self._process_images_batch(image_files)
            comparison_results[method] = method_results
            
            # Restore original config
            self.config = original_config
        
        # Analyze comparison results
        comparison_analysis = self._analyze_method_comparison(comparison_results)
        
        # Generate comparison report
        self._generate_comparison_report(comparison_analysis, output_prefix)
        
        return comparison_analysis
    
    def benchmark_performance(self, 
                            input_dir: str, 
                            num_iterations: int = 3,
                            output_prefix: str = None) -> Dict[str, Any]:
        """
        Benchmark processing performance
        
        Args:
            input_dir: Directory containing test images
            num_iterations: Number of iterations for benchmarking
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with benchmark results
        """
        if output_prefix is None:
            output_prefix = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting performance benchmark with {num_iterations} iterations")
        
        # Find a subset of images for benchmarking
        image_files = self._find_images(input_dir, recursive=True)[:50]  # Limit for benchmarking
        
        if not image_files:
            return {'error': 'No images found'}
        
        benchmark_results = {
            'iterations': num_iterations,
            'image_count': len(image_files),
            'timing_results': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
        for iteration in range(num_iterations):
            self.logger.info(f"Benchmark iteration {iteration + 1}/{num_iterations}")
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Process images
            results = self._process_images_batch(image_files)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Record metrics
            iteration_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            benchmark_results['timing_results'].append({
                'iteration': iteration + 1,
                'total_time': iteration_time,
                'avg_time_per_image': iteration_time / len(image_files),
                'images_per_second': len(image_files) / iteration_time
            })
            
            benchmark_results['memory_usage'].append({
                'iteration': iteration + 1,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'memory_delta_mb': memory_delta
            })
        
        # Calculate statistics
        benchmark_results['statistics'] = self._calculate_benchmark_statistics(benchmark_results)
        
        # Save benchmark report
        self._save_benchmark_results(benchmark_results, output_prefix)
        
        return benchmark_results
    
    def _find_images(self, 
                    input_dir: str, 
                    recursive: bool = True, 
                    file_patterns: List[str] = None) -> List[str]:
        """Find image files in directory"""
        if file_patterns is None:
            file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        input_path = Path(input_dir)
        image_files = []
        
        for pattern in file_patterns:
            if recursive:
                image_files.extend(input_path.rglob(pattern))
            else:
                image_files.extend(input_path.glob(pattern))
        
        # Convert to strings and sort
        image_files = sorted([str(f) for f in image_files])
        
        return image_files
    
    def _process_images_batch(self, image_files: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of images using parallel processing"""
        results = []
        
        if self.use_multiprocessing and len(image_files) > 1:
            # Use multiprocessing for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks in chunks
                futures = []
                
                for i in range(0, len(image_files), self.chunk_size):
                    chunk = image_files[i:i + self.chunk_size]
                    future = executor.submit(self._process_image_chunk, chunk)
                    futures.append(future)
                
                # Collect results with progress tracking
                if self.progress_callback:
                    with tqdm(total=len(futures), desc="Processing chunks") as pbar:
                        for future in futures:
                            chunk_results = future.result()
                            results.extend(chunk_results)
                            pbar.update(1)
                            if self.progress_callback:
                                self.progress_callback(len(results), len(image_files))
                else:
                    for future in futures:
                        chunk_results = future.result()
                        results.extend(chunk_results)
        else:
            # Sequential processing
            if self.progress_callback:
                with tqdm(total=len(image_files), desc="Processing images") as pbar:
                    for i, image_file in enumerate(image_files):
                        result = self._analyze_single_image(image_file)
                        results.append(result)
                        pbar.update(1)
                        if self.progress_callback:
                            self.progress_callback(i + 1, len(image_files))
            else:
                for image_file in image_files:
                    result = self._analyze_single_image(image_file)
                    results.append(result)
        
        return results
    
    def _process_image_chunk(self, image_chunk: List[str]) -> List[Dict[str, Any]]:
        """Process a chunk of images in a separate process"""
        chunk_results = []
        
        for image_file in image_chunk:
            try:
                result = self._analyze_single_image(image_file)
                chunk_results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {str(e)}")
                chunk_results.append({
                    'image_path': image_file,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return chunk_results
    
    def _analyze_single_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single image using all available methods"""
        try:
            result = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'algorithms_used': []
            }
            
            # Basic detection
            detection_results = self.detection_engine.analyze_image(image_path)
            result['detection_results'] = detection_results
            result['algorithms_used'].extend(detection_results.keys())
            
            # ML detection
            try:
                ml_results = self.ml_engine.predict(image_path)
                result['ml_results'] = ml_results
                result['algorithms_used'].append('ml_detection')
            except Exception as e:
                self.logger.debug(f"ML detection failed for {image_path}: {str(e)}")
                result['ml_results'] = {'error': str(e)}
            
            # Statistical analysis
            try:
                stats_results = self.statistical_analyzer.analyze_image(image_path)
                result['statistical_analysis'] = stats_results
                result['algorithms_used'].append('statistical_analysis')
            except Exception as e:
                self.logger.debug(f"Statistical analysis failed for {image_path}: {str(e)}")
                result['statistical_analysis'] = {'error': str(e)}
            
            # Forensic analysis
            try:
                forensic_results = self.forensic_analyzer.analyze_image(image_path)
                result['forensic_analysis'] = forensic_results
                result['algorithms_used'].append('forensic_analysis')
            except Exception as e:
                self.logger.debug(f"Forensic analysis failed for {image_path}: {str(e)}")
                result['forensic_analysis'] = {'error': str(e)}
            
            # Overall verdict
            result['overall_verdict'] = self._determine_overall_verdict(result)
            result['confidence_score'] = self._calculate_confidence_score(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {image_path}: {str(e)}")
            return {
                'image_path': image_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _determine_overall_verdict(self, analysis_result: Dict[str, Any]) -> str:
        """Determine overall verdict from all analysis results"""
        detection_results = analysis_result.get('detection_results', {})
        ml_results = analysis_result.get('ml_results', {})
        
        # Count positive detections
        positive_detections = 0
        total_detections = 0
        
        # Traditional detection algorithms
        for algo, result in detection_results.items():
            if isinstance(result, dict) and 'detected' in result:
                total_detections += 1
                if result['detected']:
                    positive_detections += 1
            elif isinstance(result, bool):
                total_detections += 1
                if result:
                    positive_detections += 1
        
        # ML detection results
        if 'prediction' in ml_results:
            total_detections += 1
            if ml_results['prediction'] == 1:  # Steganographic
                positive_detections += 1
        
        # Determine verdict based on majority voting
        if total_detections == 0:
            return 'unknown'
        
        detection_ratio = positive_detections / total_detections
        
        if detection_ratio >= 0.6:
            return 'steganographic'
        elif detection_ratio >= 0.3:
            return 'suspicious'
        else:
            return 'clean'
    
    def _calculate_confidence_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidences = []
        
        # Traditional detection confidences
        detection_results = analysis_result.get('detection_results', {})
        for algo, result in detection_results.items():
            if isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
        
        # ML confidence
        ml_results = analysis_result.get('ml_results', {})
        if 'confidence' in ml_results:
            confidences.append(ml_results['confidence'])
        
        # Calculate weighted average
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.0
    
    def _compile_batch_results(self, results: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """Compile individual results into batch summary"""
        batch_summary = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'total_images': len(results),
            'successful_analyses': len([r for r in results if 'error' not in r]),
            'failed_analyses': len([r for r in results if 'error' in r]),
            'verdicts': {},
            'algorithm_performance': {},
            'confidence_statistics': {},
            'detailed_results': results
        }
        
        # Count verdicts
        for verdict in ['steganographic', 'suspicious', 'clean', 'unknown']:
            batch_summary['verdicts'][verdict] = len([
                r for r in results 
                if r.get('overall_verdict') == verdict
            ])
        
        # Algorithm performance statistics
        algorithm_stats = {}
        for result in results:
            if 'detection_results' in result:
                for algo, algo_result in result['detection_results'].items():
                    if algo not in algorithm_stats:
                        algorithm_stats[algo] = {'detections': 0, 'total': 0}
                    
                    algorithm_stats[algo]['total'] += 1
                    if isinstance(algo_result, dict) and algo_result.get('detected', False):
                        algorithm_stats[algo]['detections'] += 1
                    elif isinstance(algo_result, bool) and algo_result:
                        algorithm_stats[algo]['detections'] += 1
        
        # Calculate detection rates
        for algo in algorithm_stats:
            if algorithm_stats[algo]['total'] > 0:
                algorithm_stats[algo]['detection_rate'] = (
                    algorithm_stats[algo]['detections'] / algorithm_stats[algo]['total']
                )
            else:
                algorithm_stats[algo]['detection_rate'] = 0.0
        
        batch_summary['algorithm_performance'] = algorithm_stats
        
        # Confidence statistics
        confidences = [r.get('confidence_score', 0.0) for r in results if 'confidence_score' in r]
        if confidences:
            batch_summary['confidence_statistics'] = {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'median': sorted(confidences)[len(confidences) // 2]
            }
        
        return batch_summary
    
    def _generate_batch_reports(self, batch_results: Dict[str, Any], output_prefix: str) -> Dict[str, str]:
        """Generate comprehensive batch reports"""
        report_files = {}
        
        try:
            # Generate batch report using report generator
            detailed_results = batch_results.get('detailed_results', [])
            output_path = str(self.output_dir / output_prefix)
            
            generated_files = self.report_generator.generate_batch_report(
                detailed_results, output_path
            )
            
            report_files.update(generated_files)
            
        except Exception as e:
            self.logger.error(f"Error generating batch reports: {str(e)}")
        
        return report_files
    
    def _save_detailed_results(self, results: List[Dict[str, Any]], output_prefix: str):
        """Save detailed results to JSON and CSV files"""
        try:
            # Save JSON
            json_path = self.output_dir / f"{output_prefix}_detailed.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save CSV summary
            csv_data = []
            for result in results:
                row = {
                    'image_path': result.get('image_path', ''),
                    'verdict': result.get('overall_verdict', ''),
                    'confidence': result.get('confidence_score', 0.0),
                    'timestamp': result.get('timestamp', ''),
                    'error': result.get('error', '')
                }
                
                # Add detection results
                detection_results = result.get('detection_results', {})
                for algo, algo_result in detection_results.items():
                    if isinstance(algo_result, dict):
                        row[f'{algo}_detected'] = algo_result.get('detected', False)
                        row[f'{algo}_confidence'] = algo_result.get('confidence', 0.0)
                    else:
                        row[f'{algo}_detected'] = bool(algo_result)
                        row[f'{algo}_confidence'] = 0.5 if algo_result else 0.0
                
                csv_data.append(row)
            
            # Save CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_path = self.output_dir / f"{output_prefix}_summary.csv"
                df.to_csv(csv_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving detailed results: {str(e)}")
    
    def _analyze_method_comparison(self, comparison_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze comparison results between different methods"""
        comparison_analysis = {
            'methods_compared': list(comparison_results.keys()),
            'method_statistics': {},
            'agreement_matrix': {},
            'performance_ranking': []
        }
        
        # Calculate statistics for each method
        for method, results in comparison_results.items():
            method_stats = {
                'total_images': len(results),
                'detections': len([r for r in results if r.get('overall_verdict') == 'steganographic']),
                'detection_rate': 0.0,
                'avg_confidence': 0.0,
                'processing_time': 0.0
            }
            
            if method_stats['total_images'] > 0:
                method_stats['detection_rate'] = method_stats['detections'] / method_stats['total_images']
                
                confidences = [r.get('confidence_score', 0.0) for r in results]
                method_stats['avg_confidence'] = sum(confidences) / len(confidences)
            
            comparison_analysis['method_statistics'][method] = method_stats
        
        return comparison_analysis
    
    def _generate_comparison_report(self, comparison_analysis: Dict[str, Any], output_prefix: str):
        """Generate method comparison report"""
        try:
            report_path = self.output_dir / f"{output_prefix}_comparison.json"
            with open(report_path, 'w') as f:
                json.dump(comparison_analysis, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {str(e)}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _calculate_benchmark_statistics(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate benchmark statistics"""
        timing_results = benchmark_results['timing_results']
        
        total_times = [r['total_time'] for r in timing_results]
        images_per_second = [r['images_per_second'] for r in timing_results]
        
        statistics = {
            'avg_total_time': sum(total_times) / len(total_times),
            'min_total_time': min(total_times),
            'max_total_time': max(total_times),
            'avg_images_per_second': sum(images_per_second) / len(images_per_second),
            'min_images_per_second': min(images_per_second),
            'max_images_per_second': max(images_per_second)
        }
        
        return statistics
    
    def _save_benchmark_results(self, benchmark_results: Dict[str, Any], output_prefix: str):
        """Save benchmark results"""
        try:
            report_path = self.output_dir / f"{output_prefix}_benchmark.json"
            with open(report_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
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
            },
            'reports': {
                'output_dir': 'reports/generated',
                'formats': ['json', 'html']
            },
            'batch_processing': {
                'max_workers': multiprocessing.cpu_count(),
                'chunk_size': 10,
                'use_multiprocessing': True,
                'output_dir': 'reports/batch'
            }
        }


def main():
    """Command-line interface for batch analysis"""
    parser = argparse.ArgumentParser(description="Batch steganography analysis")
    parser.add_argument('input_dir', help='Directory containing images to analyze')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output-prefix', '-o', help='Output files prefix')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Search subdirectories recursively')
    parser.add_argument('--workers', '-w', type=int, 
                       help='Number of worker processes')
    parser.add_argument('--patterns', nargs='+', 
                       help='File patterns to match (e.g., *.jpg *.png)')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Compare different detection methods')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize analyzer
    analyzer = BatchAnalyzer(args.config)
    
    if args.workers:
        analyzer.max_workers = args.workers
    
    # Run analysis
    if args.compare_methods:
        results = analyzer.compare_methods(
            args.input_dir, 
            output_prefix=args.output_prefix
        )
        print(f"Method comparison completed. Results: {results}")
        
    elif args.benchmark:
        results = analyzer.benchmark_performance(
            args.input_dir,
            output_prefix=args.output_prefix
        )
        print(f"Benchmark completed. Results: {results}")
        
    else:
        results = analyzer.analyze_directory(
            args.input_dir,
            output_prefix=args.output_prefix,
            recursive=args.recursive,
            file_patterns=args.patterns
        )
        
        print(f"\nBatch Analysis Complete!")
        print(f"Total images processed: {results.get('total_images', 0)}")
        print(f"Successful analyses: {results.get('successful_analyses', 0)}")
        print(f"Failed analyses: {results.get('failed_analyses', 0)}")
        print(f"Processing time: {results.get('processing_time_seconds', 0):.2f} seconds")
        print(f"Verdicts: {results.get('verdicts', {})}")
        print(f"Report files: {results.get('report_files', {})}")


if __name__ == "__main__":
    main()