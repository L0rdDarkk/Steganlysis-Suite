#!/usr/bin/env python3
"""
StegAnalysis Suite - Main Entry Point
Advanced Steganography Detection and Analysis Suite
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import with error handling
try:
    from modules.detection import DetectionEngine
except ImportError:
    DetectionEngine = None

try:
    from modules.ml_detection import MLDetectionEngine
except ImportError:
    MLDetectionEngine = None

try:
    from modules.analysis import StatisticalAnalyzer
except ImportError:
    StatisticalAnalyzer = None

try:
    from modules.forensics import ForensicAnalyzer
except ImportError:
    ForensicAnalyzer = None

try:
    from modules.reports import ReportGenerator
except ImportError:
    ReportGenerator = None


class StegAnalysisApp:
    """Main application class for StegAnalysis Suite"""
    
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Initialize engines with error handling
        self.detection_engine = DetectionEngine(self.config) if DetectionEngine else None
        self.ml_engine = MLDetectionEngine(self.config) if MLDetectionEngine else None
        self.statistical_analyzer = StatisticalAnalyzer(self.config) if StatisticalAnalyzer else None
        self.forensic_analyzer = ForensicAnalyzer(self.config) if ForensicAnalyzer else None
        self.report_generator = ReportGenerator(self.config) if ReportGenerator else None
        
        logging.info("StegAnalysis Suite initialized successfully")
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return self.default_config()
    
    def default_config(self):
        """Default configuration if file not found"""
        return {
            'detection': {
                'algorithms': ['lsb', 'chi_square', 'f5'],
                'thresholds': {
                    'chi_square': 0.05,
                    'lsb_threshold': 0.7,
                    'cnn_confidence': 0.8,
                    'svm_confidence': 0.75
                }
            },
            'ml_models': {
                'cnn': {
                    'model_path': 'models/trained/cnn_stego_detector.h5',
                    'input_shape': [224, 224, 3],
                    'batch_size': 32
                },
                'svm': {
                    'model_path': 'models/trained/svm_stego_detector.pkl',
                    'scaler_path': 'models/trained/scaler.pkl'
                }
            },
            'gpu': {'enabled': False, 'memory_growth': True},
            'processing': {'max_workers': 4, 'timeout': 300},
            'reports': {
                'output_dir': 'reports/generated',
                'formats': ['pdf', 'json']
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/steganalysis.log'
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            "datasets/images/clean",
            "datasets/images/steganographic", 
            "datasets/images/test",
            "datasets/metadata",
            "models/trained",
            "reports/generated",
            "reports/templates",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def quick_detection(self, image_path):
        """Quick detection using available models"""
        print(f"🔍 Quick Detection: {image_path}")
        print("=" * 50)
        
        try:
            # Try to use trained CNN model directly
            import tensorflow as tf
            import cv2
            import numpy as np
            
            # Load the trained model
            model_path = "models/trained/cnn_stego_detector.h5"
            if Path(model_path).exists():
                model = tf.keras.models.load_model(model_path)
                
                # Load and preprocess image
                img = cv2.imread(str(image_path))
                if img is not None:
                    img_resized = cv2.resize(img, (224, 224))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_normalized = img_rgb.astype(np.float32) / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(img_batch, verbose=0)
                    confidence = prediction[0][0]
                    
                    if confidence > 0.5:
                        print(f"🔴 STEGANOGRAPHIC CONTENT DETECTED")
                        print(f"📊 Confidence: {confidence:.2%}")
                        print(f"🔬 Method: CNN Detection")
                        print(f"⚠️  Recommended: Isolate file for further analysis")
                    else:
                        print(f"🟢 CLEAN IMAGE")
                        print(f"📊 Confidence: {(1-confidence):.2%}")
                        print(f"🔬 Method: CNN Detection")
                        print(f"✅ Image appears clean - no steganographic content detected")
                else:
                    print("❌ Could not load image")
            else:
                print("❌ CNN model not found. Train models first with:")
                print("   python tools/model_trainer.py")
                
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            print("Make sure you have trained the models first!")
    
    def batch_analyze(self, input_dir, output_file=None):
        """Batch analyze multiple images"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_dir} not found")
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"**/*{ext}"))
            image_files.extend(input_path.glob(f"**/*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to analyze...")
        
        # Analyze each image
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔍 Analyzing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                # Use quick detection for each image
                import tensorflow as tf
                import cv2
                import numpy as np
                
                model_path = "models/trained/cnn_stego_detector.h5"
                if Path(model_path).exists():
                    model = tf.keras.models.load_model(model_path)
                    
                    img = cv2.imread(str(image_file))
                    if img is not None:
                        img_resized = cv2.resize(img, (224, 224))
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        img_normalized = img_rgb.astype(np.float32) / 255.0
                        img_batch = np.expand_dims(img_normalized, axis=0)
                        
                        prediction = model.predict(img_batch, verbose=0)
                        confidence = prediction[0][0]
                        
                        if confidence > 0.5:
                            verdict = "steganographic"
                            print(f"   🔴 STEGANOGRAPHIC (Confidence: {confidence:.2%})")
                        else:
                            verdict = "clean"
                            print(f"   🟢 CLEAN (Confidence: {(1-confidence):.2%})")
                        
                        results.append({
                            'filename': image_file.name,
                            'verdict': verdict,
                            'confidence': confidence,
                            'path': str(image_file)
                        })
                    else:
                        print(f"   ❌ Could not load image")
                        results.append({
                            'filename': image_file.name,
                            'verdict': 'error',
                            'confidence': 0,
                            'path': str(image_file)
                        })
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'filename': image_file.name,
                    'verdict': 'error',
                    'confidence': 0,
                    'path': str(image_file)
                })
        
        # Print summary
        total = len(results)
        steganographic = sum(1 for r in results if r['verdict'] == 'steganographic')
        clean = sum(1 for r in results if r['verdict'] == 'clean')
        errors = sum(1 for r in results if r['verdict'] == 'error')
        
        print("\n" + "=" * 50)
        print("📊 BATCH ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Images: {total}")
        print(f"🔴 Steganographic: {steganographic}")
        print(f"🟢 Clean: {clean}")
        print(f"❌ Errors: {errors}")
        
        return results
    
    def run_gui(self):
        """Launch the graphical user interface"""
        try:
            from PyQt5.QtWidgets import QApplication
            # Try different import paths for the main window
            try:
                from gui.main_window import StegAnalysisMainWindow
                window = StegAnalysisMainWindow()
            except ImportError:
                try:
                    from gui.main_window import MainWindow
                    window = MainWindow()
                except ImportError:
                    # Create a simple window if advanced one fails
                    window = self.create_simple_gui()
                    
            app = QApplication(sys.argv)
            window.show()
            return app.exec_()
        except ImportError as e:
            logging.error(f"Failed to import GUI components: {e}")
            print("ERROR: PyQt5 not available. Cannot launch GUI.")
            print("GUI requires PyQt5. Install with: pip install PyQt5")
            return 1
    
    def create_simple_gui(self):
        """Create a simple GUI window as fallback"""
        from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
        
        class SimpleGUI(QMainWindow):
            def __init__(self, app_instance):
                super().__init__()
                self.app = app_instance
                self.setWindowTitle("StegAnalysis Suite v1.0")
                self.setGeometry(300, 300, 800, 600)
                
                # Central widget
                central = QWidget()
                self.setCentralWidget(central)
                layout = QVBoxLayout(central)
                
                # Header
                header = QLabel("🔬 StegAnalysis Suite v1.0")
                header.setStyleSheet("font-size: 24px; font-weight: bold; text-align: center; padding: 20px;")
                layout.addWidget(header)
                
                # Description
                desc = QLabel("Professional Steganography Detection Tool\n\n"
                             "Click 'Analyze Image' to detect steganographic content in images")
                desc.setStyleSheet("font-size: 14px; text-align: center; padding: 20px;")
                layout.addWidget(desc)
                
                # Analyze button
                analyze_btn = QPushButton("🔍 Analyze Image")
                analyze_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 16px;
                        padding: 15px 30px;
                        background-color: #0078D4;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #106EBE;
                    }
                """)
                analyze_btn.clicked.connect(self.analyze_image)
                layout.addWidget(analyze_btn)
                
                # Status
                self.status_label = QLabel("Ready for analysis")
                self.status_label.setStyleSheet("font-size: 12px; text-align: center; padding: 10px; color: #666;")
                layout.addWidget(self.status_label)
                
                # Apply dark theme
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #2B2B2B;
                        color: #FFFFFF;
                    }
                    QWidget {
                        background-color: #2B2B2B;
                        color: #FFFFFF;
                    }
                    QLabel {
                        color: #FFFFFF;
                    }
                """)
            
            def analyze_image(self):
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select Image", "", 
                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
                )
                
                if file_path:
                    self.status_label.setText(f"Analyzing: {file_path.split('/')[-1]}")
                    try:
                        self.app.quick_detection(file_path)
                        QMessageBox.information(self, "Analysis Complete", 
                                              f"Analysis complete for {file_path.split('/')[-1]}\n\n"
                                              "Check the terminal for detailed results.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
                    finally:
                        self.status_label.setText("Ready for analysis")
        
        return SimpleGUI(self)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StegAnalysis Suite")
    
    # Add detection arguments
    parser.add_argument('--detect', action='store_true',
                       help='Run detection mode')
    parser.add_argument('--image', type=str,
                       help='Single image file to analyze')
    parser.add_argument('--folder', type=str,
                       help='Folder containing images to analyze')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    
    # Existing arguments
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--gui', action='store_true', help='Launch GUI')
    parser.add_argument('--analyze', help='Analyze single image')
    parser.add_argument('--batch', help='Batch analyze directory')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['lsb', 'chi_square', 'f5', 'cnn', 'svm'],
                       help='Detection algorithms to use')
    
    args = parser.parse_args()
    
    # Initialize the application
    app = StegAnalysisApp(args.config)
    
    if args.gui:
        print("🚀 Launching GUI interface...")
        return app.run_gui()
    elif args.detect and args.image:
        if not Path(args.image).exists():
            print(f"❌ Error: Image file not found: {args.image}")
            return 1
        app.quick_detection(args.image)
        return 0
    elif args.detect and args.folder:
        if not Path(args.folder).exists():
            print(f"❌ Error: Folder not found: {args.folder}")
            return 1
        results = app.batch_analyze(args.folder, args.output if args.report else None)
        return 0
    else:
        print("🔬 StegAnalysis Suite v1.0")
        print("Advanced Steganography Detection and Analysis")
        print("=" * 50)
        print("Usage:")
        print("  --gui                    Launch graphical interface")
        print("  --detect --image FILE    Quick detection on single image")
        print("  --detect --folder DIR    Batch detection on folder")
        print("  --help                   Show full help")
        print()
        print("Examples:")
        print("  python main.py --gui")
        print("  python main.py --detect --image photo.jpg")
        print("  python main.py --detect --folder test_images/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())