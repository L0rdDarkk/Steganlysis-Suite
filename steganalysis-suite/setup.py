#!/usr/bin/env python3
"""
StegAnalysis Suite Setup Script
Advanced Steganography Detection and Analysis Suite
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Advanced Steganography Detection and Analysis Suite"

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering out comments and options"""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle platform-specific requirements
                    if ';' in line:
                        requirements.append(line)
                    else:
                        requirements.append(line)
    return requirements

# Base requirements (always installed)
base_requirements = [
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "Pillow>=8.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "PyYAML>=6.0",
    "click>=8.0.0",
    "tqdm>=4.62.0",
    "python-magic>=0.4.24",
    "reportlab>=3.6.0",
    "jinja2>=3.0.0",
]

# Optional dependencies
extras_require = {
    'gpu': [
        'tensorflow-gpu>=2.10.0,<2.13.0',
        'cupy-cuda11x>=10.0.0; platform_machine=="x86_64"',
        'pynvml>=11.0.0',
    ],
    'gui': [
        'PyQt5>=5.15.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
    ],
    'ml': [
        'tensorflow>=2.10.0',
        'keras>=2.10.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',
    ],
    'forensics': [
        'pyexiv2>=2.7.0',
        'pillow-heif>=0.10.0',
        'rawpy>=0.17.0',
        'imageio>=2.19.0',
    ],
    'stats': [
        'statsmodels>=0.13.0',
        'pingouin>=0.5.0',
        'scikit-image>=0.19.0',
    ],
    'performance': [
        'numba>=0.56.0',
        'joblib>=1.1.0',
        'psutil>=5.8.0',
        'ray>=1.13.0',
    ],
    'web': [
        'flask>=2.0.0',
        'dash>=2.0.0',
        'streamlit>=1.0.0',
    ],
    'cloud': [
        'boto3>=1.20.0',
        'google-cloud-storage>=2.0.0',
        'azure-storage-blob>=12.0.0',
    ],
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.910',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'jupyter': [
        'jupyter>=1.0.0',
        'ipywidgets>=7.7.0',
        'notebook>=6.4.0',
    ]
}

# Complete installation with all features
extras_require['all'] = list(set(
    sum(extras_require.values(), [])
))

# Platform-specific requirements
if sys.platform == "win32":
    base_requirements.append("pywin32>=227")
elif sys.platform.startswith("linux"):
    base_requirements.append("python-magic>=0.4.24")
elif sys.platform == "darwin":
    base_requirements.append("pyobjc>=8.0")

# Check Python version
if sys.version_info < (3, 8):
    raise RuntimeError("StegAnalysis Suite requires Python 3.8 or higher")

setup(
    name="steganalysis-suite",
    version="1.0.0",
    author="StegAnalysis Team",
    author_email="team@steganalysis.org",
    description="Advanced Steganography Detection and Analysis Suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steganalysis/steganalysis-suite",
    project_urls={
        "Bug Reports": "https://github.com/steganalysis/steganalysis-suite/issues",
        "Source": "https://github.com/steganalysis/steganalysis-suite",
        "Documentation": "https://steganalysis-suite.readthedocs.io/",
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_data={
        "steganalysis_suite": [
            "config/*.yaml",
            "models/trained/*.h5",
            "models/trained/*.pkl",
            "reports/templates/*.html",
            "reports/templates/*.json",
            "gui/icons/*.png",
            "gui/styles/*.qss",
        ]
    },
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require=extras_require,
    
    # Entry points
    entry_points={
        "console_scripts": [
            "steganalysis=main:main",
            "steg-detect=main:main",
            "steg-analyze=tools.batch_analyzer:main",
            "steg-train=tools.model_trainer:main",
            "steg-benchmark=tools.benchmark:main",
        ],
        "gui_scripts": [
            "steganalysis-gui=main:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security :: Cryptography",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: X11 Applications :: Qt",
    ],
    
    # Keywords for PyPI
    keywords=[
        "steganography", "steganalysis", "image-processing", 
        "machine-learning", "computer-vision", "forensics",
        "security", "detection", "deep-learning", "gpu"
    ],
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
    zip_safe=False,
    
    # Custom commands
    cmdclass={},
    
    # Data files
    data_files=[
        ("config", ["config.yaml"]),
        ("docs", ["README.md", "LICENSE"]),
    ],
    
    # Options
    options={
        "build_exe": {
            "packages": [
                "numpy", "opencv-python", "PIL", "sklearn", 
                "matplotlib", "pandas", "scipy", "yaml"
            ],
            "include_files": [
                ("config.yaml", "config.yaml"),
                ("models/", "models/"),
                ("reports/templates/", "reports/templates/"),
            ]
        }
    },
    
    # Test suite
    test_suite="tests",
    tests_require=[
        "pytest>=6.2.0",
        "pytest-cov>=3.0.0",
        "pytest-mock>=3.6.0",
    ],
)