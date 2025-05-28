#!/usr/bin/env python3
"""
StegAnalysis Suite - Modules Package
Core detection and analysis modules for steganography detection
"""

import os
# Set TensorFlow logging before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__version__ = "1.0.0"
__author__ = "StegAnalysis Team"

# Import core modules with error handling
try:
    from .detection import DetectionEngine
    print("✅ DetectionEngine imported")
except ImportError as e:
    print(f"❌ DetectionEngine import failed: {e}")
    DetectionEngine = None

try:
    from .ml_detection import MLDetectionEngine
    print("✅ MLDetectionEngine imported")
except ImportError as e:
    print(f"❌ MLDetectionEngine import failed: {e}")
    MLDetectionEngine = None

try:
    from .analysis import StatisticalAnalyzer
    print("✅ StatisticalAnalyzer imported")
except ImportError as e:
    print(f"❌ StatisticalAnalyzer import failed: {e}")
    StatisticalAnalyzer = None

try:
    from .forensics import ForensicAnalyzer
    print("✅ ForensicAnalyzer imported")
except ImportError as e:
    print(f"❌ ForensicAnalyzer import failed: {e}")
    ForensicAnalyzer = None

try:
    from .extraction import DataExtractor
    print("✅ DataExtractor imported")
except ImportError as e:
    print(f"❌ DataExtractor import failed: {e}")
    DataExtractor = None

try:
    from .reports import ReportGenerator
    print("✅ ReportGenerator imported")
except ImportError as e:
    print(f"❌ ReportGenerator import failed: {e}")
    ReportGenerator = None

# Create alias for backward compatibility
MLDetection = MLDetectionEngine

__all__ = [
    'DetectionEngine',
    'MLDetectionEngine',
    'MLDetection',
    'StatisticalAnalyzer',
    'ForensicAnalyzer',
    'DataExtractor',
    'ReportGenerator'
]