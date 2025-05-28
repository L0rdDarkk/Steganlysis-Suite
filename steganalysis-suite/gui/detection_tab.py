#!/usr/bin/env python3
"""
StegAnalysis Suite - Detection Tab (Clean Version)
Professional interface for steganography detection
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import threading
import time
import numpy as np
from datetime import datetime

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
        QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
        QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
        QHeaderView, QFileDialog, QCheckBox, QComboBox, QSpinBox,
        QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QMessageBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMimeData
    from PyQt5.QtGui import QPixmap, QIcon, QDragEnterEvent, QDropEvent, QFont, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    raise ImportError("PyQt5 required. Install with: pip install PyQt5")


# Utility functions
def create_card_style():
    return """
        QFrame {
            background-color: #2B2B2B;
            border: 1px solid #555555;
            border-radius: 8px;
            margin: 2px;
        }
    """

def create_button_style(button_type='primary'):
    if button_type == 'primary':
        return """
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
        """
    else:  # secondary
        return """
            QPushButton {
                background-color: #4A4A4A;
                color: white;
                border: 1px solid #666666;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: #3A3A3A;
            }
        """

def create_table_style():
    return """
        QTableWidget {
            background-color: #2B2B2B;
            color: white;
            border: 1px solid #555555;
            gridline-color: #555555;
            selection-background-color: #0078D4;
        }
        QHeaderView::section {
            background-color: #4A4A4A;
            color: white;
            padding: 8px;
            border: 1px solid #555555;
            font-weight: bold;
        }
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #555555;
        }
        QTableWidget::item:selected {
            background-color: #0078D4;
        }
    """

def get_current_theme_color(color_type):
    colors = {
        'primary': '#0078D4',
        'success': '#44FF44',
        'warning': '#FFAA00',
        'error': '#FF4444',
        'info': '#00AAFF',
        'text': '#FFFFFF',
        'text_secondary': '#CCCCCC',
        'background': '#2B2B2B',
        'surface': '#3A3A3A',
        'border': '#555555'
    }
    return colors.get(color_type, '#FFFFFF')

def show_error_message(parent, title, message):
    QMessageBox.critical(parent, title, message)

def show_success_message(parent, title, message):
    QMessageBox.information(parent, title, message)

def scale_pixmap_to_fit(pixmap, max_width, max_height):
    return pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class DetectionWorker(QThread):
    """Worker thread for running detection algorithms"""
    
    progress_updated = pyqtSignal(int)
    detection_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, image_path: str, selected_methods: List[str], settings: Dict[str, Any]):
        super().__init__()
        self.image_path = image_path
        self.selected_methods = selected_methods
        self.settings = settings
        self.results = {}
        
    def run(self):
        """Run detection algorithms"""
        try:
            if not PIL_AVAILABLE:
                self.error_occurred.emit("PIL (Pillow) is required for image processing. Please install: pip install Pillow")
                return
                
            total_methods = len(self.selected_methods)
            current_method = 0
            
            self.log_message.emit(f"Starting detection on: {os.path.basename(self.image_path)}")
            
            # Load image once
            image = Image.open(self.image_path)
            img_array = np.array(image)
            
            for method in self.selected_methods:
                current_method += 1
                progress = int((current_method / total_methods) * 100)
                self.progress_updated.emit(progress)
                
                self.log_message.emit(f"Running {method} detection...")
                
                if method == "chi_square":
                    result = self.chi_square_test(img_array)
                elif method == "lsb_analysis":
                    result = self.lsb_analysis(img_array)
                elif method == "histogram_analysis":
                    result = self.histogram_analysis(img_array)
                elif method == "pixel_analysis":
                    result = self.pixel_analysis(img_array)
                elif method == "statistical_analysis":
                    result = self.statistical_analysis(img_array)
                else:
                    result = {"error": f"Unknown method: {method}"}
                
                self.results[method] = result
                time.sleep(0.2)  # Small delay for UI responsiveness
            
            self.detection_finished.emit(self.results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def chi_square_test(self, img_array):
        """Perform Chi-Square test for randomness"""
        try:
            if len(img_array.shape) == 3:
                # Convert to grayscale
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # Calculate histogram
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            
            # Expected frequency for uniform distribution
            expected_freq = len(gray.flatten()) / 256
            
            # Calculate chi-square statistic
            chi_square_stat = np.sum((hist - expected_freq) ** 2 / expected_freq)
            
            # Simple threshold-based decision
            threshold = 255 * 1.5  # Adjustable threshold
            is_steganographic = chi_square_stat > threshold
            confidence = min(100, (chi_square_stat / threshold) * 50 + 50) if is_steganographic else max(0, 100 - (chi_square_stat / threshold) * 50)
            
            return {
                'is_steganographic': bool(is_steganographic),
                'confidence': float(confidence),
                'chi_square_stat': float(chi_square_stat),
                'threshold': float(threshold),
                'processing_time': 0.1
            }
        except Exception as e:
            return {"error": str(e)}
    
    def lsb_analysis(self, img_array):
        """Analyze Least Significant Bits"""
        try:
            if len(img_array.shape) == 3:
                # Analyze all channels
                results = {}
                overall_suspicious = False
                total_confidence = 0
                
                for i, channel in enumerate(['Red', 'Green', 'Blue']):
                    channel_data = img_array[:, :, i]
                    lsb_bits = channel_data & 1  # Extract LSB
                    
                    # Calculate ratio of 1s to 0s
                    ones_count = np.sum(lsb_bits)
                    total_pixels = lsb_bits.size
                    ratio = ones_count / total_pixels
                    
                    # Ideal ratio should be around 0.5 for natural images
                    deviation = abs(ratio - 0.5)
                    
                    # If deviation is too small, it might indicate steganography
                    is_suspicious = deviation < 0.02  # Very uniform distribution
                    confidence = (0.02 - deviation) / 0.02 * 100 if is_suspicious else deviation * 1000
                    confidence = max(0, min(100, confidence))
                    
                    results[channel] = {
                        'ratio': float(ratio),
                        'deviation': float(deviation),
                        'suspicious': bool(is_suspicious),
                        'confidence': float(confidence)
                    }
                    
                    if is_suspicious:
                        overall_suspicious = True
                    total_confidence += confidence
                
                avg_confidence = total_confidence / 3
                
                return {
                    'is_steganographic': overall_suspicious,
                    'confidence': float(avg_confidence),
                    'channel_analysis': results,
                    'processing_time': 0.15
                }
            else:
                # Grayscale analysis
                lsb_bits = img_array & 1
                ones_count = np.sum(lsb_bits)
                total_pixels = lsb_bits.size
                ratio = ones_count / total_pixels
                deviation = abs(ratio - 0.5)
                
                is_suspicious = deviation < 0.02
                confidence = (0.02 - deviation) / 0.02 * 100 if is_suspicious else deviation * 1000
                confidence = max(0, min(100, confidence))
                
                return {
                    'is_steganographic': bool(is_suspicious),
                    'confidence': float(confidence),
                    'ratio': float(ratio),
                    'deviation': float(deviation),
                    'processing_time': 0.1
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def histogram_analysis(self, img_array):
        """Analyze histogram patterns"""
        try:
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # Calculate histogram
            hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            
            # Calculate statistics
            mean_freq = np.mean(hist)
            std_freq = np.std(hist)
            
            # Look for unusual patterns
            # Steganography might create very uniform distributions
            uniformity = std_freq / mean_freq if mean_freq > 0 else 0
            
            # Also check for pairs analysis (for LSB embedding)
            pairs_of_values = []
            for i in range(0, 256, 2):
                if i + 1 < 256:
                    pairs_of_values.append((hist[i], hist[i + 1]))
            
            # Calculate chi-square for pairs
            pair_chi_square = 0
            for h1, h2 in pairs_of_values:
                expected = (h1 + h2) / 2
                if expected > 0:
                    pair_chi_square += ((h1 - expected) ** 2 + (h2 - expected) ** 2) / expected
            
            is_steganographic = pair_chi_square > 100 or uniformity < 0.8
            confidence = min(100, max(pair_chi_square / 100 * 50, (0.8 - uniformity) / 0.8 * 50))
            
            return {
                'is_steganographic': bool(is_steganographic),
                'confidence': float(confidence),
                'uniformity': float(uniformity),
                'pair_chi_square': float(pair_chi_square),
                'mean_frequency': float(mean_freq),
                'std_frequency': float(std_freq),
                'processing_time': 0.12
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def pixel_analysis(self, img_array):
        """Analyze pixel value patterns"""
        try:
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # Calculate differences between adjacent pixels
            h_diff = np.abs(np.diff(gray, axis=1))
            v_diff = np.abs(np.diff(gray, axis=0))
            
            # Calculate statistics
            h_mean = np.mean(h_diff)
            v_mean = np.mean(v_diff)
            h_std = np.std(h_diff)
            v_std = np.std(v_diff)
            
            # Check for unusual smoothness (might indicate embedding)
            smoothness_h = h_std / h_mean if h_mean > 0 else 0
            smoothness_v = v_std / v_mean if v_mean > 0 else 0
            avg_smoothness = (smoothness_h + smoothness_v) / 2
            
            # Very smooth images might have hidden data
            is_steganographic = avg_smoothness < 0.5
            confidence = (0.5 - avg_smoothness) / 0.5 * 100 if is_steganographic else 20
            confidence = max(0, min(100, confidence))
            
            return {
                'is_steganographic': bool(is_steganographic),
                'confidence': float(confidence),
                'smoothness_horizontal': float(smoothness_h),
                'smoothness_vertical': float(smoothness_v),
                'average_smoothness': float(avg_smoothness),
                'h_mean_diff': float(h_mean),
                'v_mean_diff': float(v_mean),
                'processing_time': 0.18
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def statistical_analysis(self, img_array):
        """Perform comprehensive statistical analysis"""
        try:
            if len(img_array.shape) == 3:
                # Analyze each channel
                results = {}
                for i, channel in enumerate(['Red', 'Green', 'Blue']):
                    channel_data = img_array[:, :, i].flatten()
                    results[channel] = self.calculate_channel_stats(channel_data)
                
                # Overall assessment
                suspicious_channels = sum(1 for r in results.values() if r.get('suspicious', False))
                overall_suspicious = suspicious_channels >= 2
                avg_confidence = np.mean([r.get('confidence', 0) for r in results.values()])
                
                return {
                    'is_steganographic': bool(overall_suspicious),
                    'confidence': float(avg_confidence),
                    'suspicious_channels': int(suspicious_channels),
                    'channel_stats': results,
                    'processing_time': 0.25
                }
            else:
                # Grayscale analysis
                stats = self.calculate_channel_stats(img_array.flatten())
                return {
                    'is_steganographic': stats.get('suspicious', False),
                    'confidence': stats.get('confidence', 0),
                    **stats,
                    'processing_time': 0.2
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_channel_stats(self, data):
        """Calculate statistical measures for a data channel"""
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Calculate entropy
            _, counts = np.unique(data, return_counts=True)
            probabilities = counts / len(data)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Calculate skewness (simplified)
            skewness = np.mean(((data - mean_val) / std_val) ** 3) if std_val > 0 else 0
            
            # Calculate kurtosis (simplified)
            kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3 if std_val > 0 else 0
            
            # Suspicion based on entropy and distribution
            # High entropy with unusual skewness/kurtosis might indicate steganography
            entropy_threshold = 7.0  # Typical for natural images
            is_suspicious = entropy > entropy_threshold and (abs(skewness) > 1 or abs(kurtosis) > 2)
            
            confidence = 0
            if is_suspicious:
                confidence = min(100, (entropy - entropy_threshold) * 20 + abs(skewness) * 10 + abs(kurtosis) * 5)
            else:
                confidence = max(0, 50 - abs(entropy - entropy_threshold) * 10)
            
            return {
                'mean': float(mean_val),
                'std_dev': float(std_val),
                'entropy': float(entropy),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'suspicious': bool(is_suspicious),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            return {"error": str(e)}


class ImageDisplayWidget(QLabel):
    """Custom widget for displaying images with drag and drop support"""
    
    image_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {get_current_theme_color('border')};
                border-radius: 10px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
                font-size: 16px;
                font-weight: bold;
            }}
        """)
        self.setText("Drop image here or click to browse\n\nSupported formats: PNG, JPG, BMP, TIFF")
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and self.is_image_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_image_file(file_path):
                self.image_dropped.emit(file_path)
                event.acceptProposedAction()
    
    def mousePressEvent(self, event):
        """Handle mouse click to open file dialog"""
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All Files (*)"
            )
            if file_path:
                self.image_dropped.emit(file_path)
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if file is a supported image format"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        return Path(file_path).suffix.lower() in image_extensions
    
    def display_image(self, image_path: str):
        """Display image in the widget"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale image to fit widget while maintaining aspect ratio
                scaled_pixmap = scale_pixmap_to_fit(pixmap, self.width() - 20, self.height() - 20)
                self.setPixmap(scaled_pixmap)
                self.setStyleSheet(f"""
                    QLabel {{
                        border: 1px solid {get_current_theme_color('border')};
                        border-radius: 10px;
                        background-color: {get_current_theme_color('background')};
                        padding: 10px;
                    }}
                """)
            else:
                self.setText("Error loading image")
        except Exception as e:
            self.setText(f"Error: {str(e)}")


class DetectionResultsWidget(QWidget):
    """Widget for displaying detection results in a professional format"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the results display UI"""
        layout = QVBoxLayout(self)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Detection Method", "Result", "Confidence", "Details"
        ])
        
        # Apply table styling
        self.results_table.setStyleSheet(create_table_style())
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.results_table)
        
        # Summary section
        summary_group = QGroupBox("Detection Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_label = QLabel("No detection results available")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"""
            QLabel {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
            }}
        """)
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
    
    def update_results(self, results: Dict[str, Any]):
        """Update the results display with new detection results"""
        self.results_table.setRowCount(0)
        
        if not results:
            self.summary_label.setText("No detection results available")
            return
        
        # Populate results table
        for method, result in results.items():
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            # Method name
            method_item = QTableWidgetItem(self.format_method_name(method))
            method_item.setFont(QFont("", 11, QFont.Bold))
            self.results_table.setItem(row, 0, method_item)
            
            # Result
            if "error" in result:
                result_text = "Error"
                result_item = QTableWidgetItem(result_text)
                result_item.setForeground(QColor('#FF4444'))
            else:
                is_steganographic = result.get('is_steganographic', False)
                result_text = "STEGANOGRAPHIC" if is_steganographic else "CLEAN"
                result_item = QTableWidgetItem(result_text)
                
                if is_steganographic:
                    result_item.setForeground(QColor('#FFAA00'))
                else:
                    result_item.setForeground(QColor('#44FF44'))
            
            result_item.setFont(QFont("", 11, QFont.Bold))
            self.results_table.setItem(row, 1, result_item)
            
            # Confidence
            confidence = result.get('confidence', 0.0)
            if isinstance(confidence, (int, float)):
                confidence_text = f"{confidence:.1f}%"
            else:
                confidence_text = "N/A"
            self.results_table.setItem(row, 2, QTableWidgetItem(confidence_text))
            
            # Details
            details = self.format_details(result)
            self.results_table.setItem(row, 3, QTableWidgetItem(details))
        
        # Resize columns to content
        self.results_table.resizeColumnsToContents()
        
        # Update summary
        self.update_summary(results)
    
    def format_method_name(self, method: str) -> str:
        """Format method name for display"""
        name_mapping = {
            'chi_square': 'Chi-Square Test',
            'lsb_analysis': 'LSB Analysis',
            'histogram_analysis': 'Histogram Analysis',
            'pixel_analysis': 'Pixel Analysis',
            'statistical_analysis': 'Statistical Analysis'
        }
        return name_mapping.get(method, method.replace('_', ' ').title())
    
    def format_details(self, result: Dict[str, Any]) -> str:
        """Format result details for display"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        details = []
        
        # Add relevant details based on result content
        for key, value in result.items():
            if key in ['is_steganographic', 'confidence', 'processing_time']:
                continue
            if key == 'processing_time':
                details.append(f"Time: {value:.3f}s")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float):
                    details.append(f"{key}: {value:.3f}")
                else:
                    details.append(f"{key}: {value}")
        
        return " | ".join(details[:3]) if details else "No additional details"
    
    def update_summary(self, results: Dict[str, Any]):
        """Update the summary section"""
        total_methods = len(results)
        steganographic_count = 0
        clean_count = 0
        error_count = 0
        
        for result in results.values():
            if "error" in result:
                error_count += 1
            elif result.get('is_steganographic', False):
                steganographic_count += 1
            else:
                clean_count += 1
        
        # Overall assessment
        if error_count == total_methods:
            overall_result = "Unable to analyze (all methods failed)"
            status_color = get_current_theme_color('error')
        elif steganographic_count > clean_count:
            overall_result = "LIKELY STEGANOGRAPHIC"
            status_color = get_current_theme_color('warning')
        elif clean_count > steganographic_count:
            overall_result = "LIKELY CLEAN"
            status_color = get_current_theme_color('success')
        else:
            overall_result = "INCONCLUSIVE"
            status_color = get_current_theme_color('info')
        
        summary_text = f"""
        <div style="font-size: 14px;">
            <p><b>Overall Assessment:</b> <span style="color: {status_color}; font-weight: bold;">{overall_result}</span></p>
            <p><b>Detection Results:</b></p>
            <ul>
                <li>Steganographic: {steganographic_count} method(s)</li>
                <li>Clean: {clean_count} method(s)</li>
                <li>Errors: {error_count} method(s)</li>
            </ul>
            <p><b>Recommendation:</b> {self.get_recommendation(steganographic_count, clean_count, error_count)}</p>
        </div>
        """
        
        self.summary_label.setText(summary_text)
    
    def get_recommendation(self, stego_count: int, clean_count: int, error_count: int) -> str:
        """Generate recommendation based on results"""
        total = stego_count + clean_count + error_count
        
        if error_count == total:
            return "Unable to provide recommendation due to analysis errors."
        
        if stego_count > clean_count:
            return "High probability of steganographic content. Further investigation recommended."
        elif clean_count > stego_count:
            return "Low probability of steganographic content. Image appears clean."
        else:
            return "Mixed results. Consider running additional detection methods or manual analysis."


class DetectionSettingsWidget(QWidget):
    """Widget for detection settings and method selection"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the settings UI"""
        layout = QVBoxLayout(self)
        
        # Detection methods group
        methods_group = QGroupBox("Detection Methods")
        methods_layout = QVBoxLayout(methods_group)
        
        self.method_checkboxes = {}
        methods = [
            ("chi_square", "Chi-Square Test", "Statistical analysis for LSB steganography"),
            ("lsb_analysis", "LSB Analysis", "Direct analysis of least significant bits"),
            ("histogram_analysis", "Histogram Analysis", "Analyze pixel distribution patterns"),
            ("pixel_analysis", "Pixel Analysis", "Analyze pixel value relationships"),
            ("statistical_analysis", "Statistical Analysis", "Comprehensive statistical measures")
        ]
        
        for method_id, method_name, description in methods:
            checkbox = QCheckBox(method_name)
            checkbox.setChecked(True)  # Default: all methods enabled
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    font-weight: bold;
                    spacing: 8px;
                    color: white;
                }}
                QCheckBox::indicator {{
                    width: 18px;
                    height: 18px;
                }}
            """)
            
            self.method_checkboxes[method_id] = checkbox
            methods_layout.addWidget(checkbox)
        
        layout.addWidget(methods_group)
        
        # Advanced settings group
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout(advanced_group)
        
        # Sensitivity setting
        advanced_layout.addWidget(QLabel("Sensitivity:"), 0, 0)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(1)
        advanced_layout.addWidget(self.sensitivity_slider, 0, 1)
        
        self.sensitivity_label = QLabel("5")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(str(v))
        )
        advanced_layout.addWidget(self.sensitivity_label, 0, 2)
        
        # Confidence threshold
        advanced_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.confidence_spinbox = QSpinBox()
        self.confidence_spinbox.setRange(50, 99)
        self.confidence_spinbox.setValue(75)
        self.confidence_spinbox.setSuffix("%")
        advanced_layout.addWidget(self.confidence_spinbox, 1, 1)
        
        layout.addWidget(advanced_group)
        
        # Quick presets
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        preset_buttons = [
            ("Fast", "Enable fast detection methods only"),
            ("Balanced", "Balance between speed and accuracy"),
            ("Thorough", "Enable all methods for maximum accuracy")
        ]
        
        for preset_name, tooltip in preset_buttons:
            btn = QPushButton(preset_name)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(create_button_style('secondary'))
            btn.clicked.connect(lambda checked, name=preset_name: self.apply_preset(name))
            presets_layout.addWidget(btn)
        
        layout.addWidget(presets_group)
        
        # Settings management
        settings_group = QGroupBox("Settings Management")
        settings_layout = QHBoxLayout(settings_group)
        
        export_btn = QPushButton("Export Settings")
        export_btn.setStyleSheet(create_button_style('secondary'))
        export_btn.clicked.connect(self.export_settings)
        settings_layout.addWidget(export_btn)
        
        import_btn = QPushButton("Import Settings")
        import_btn.setStyleSheet(create_button_style('secondary'))
        import_btn.clicked.connect(self.import_settings)
        settings_layout.addWidget(import_btn)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setStyleSheet(create_button_style('secondary'))
        reset_btn.clicked.connect(self.reset_to_defaults)
        settings_layout.addWidget(reset_btn)
        
        layout.addWidget(settings_group)
        
        layout.addStretch()
    
    def get_selected_methods(self) -> List[str]:
        """Get list of selected detection methods"""
        return [method for method, checkbox in self.method_checkboxes.items() 
                if checkbox.isChecked()]
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current detection settings"""
        return {
            'selected_methods': self.get_selected_methods(),
            'sensitivity': self.sensitivity_slider.value(),
            'confidence_threshold': self.confidence_spinbox.value()
        }
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration"""
        if preset_name == "Fast":
            # Enable only fast methods
            for method, checkbox in self.method_checkboxes.items():
                checkbox.setChecked(method in ['chi_square', 'lsb_analysis'])
            self.sensitivity_slider.setValue(3)
            self.confidence_spinbox.setValue(70)
            
        elif preset_name == "Balanced":
            # Enable most methods except slow ones
            for method, checkbox in self.method_checkboxes.items():
                checkbox.setChecked(method in ['chi_square', 'lsb_analysis', 'histogram_analysis'])
            self.sensitivity_slider.setValue(5)
            self.confidence_spinbox.setValue(75)
            
        elif preset_name == "Thorough":
            # Enable all methods
            for checkbox in self.method_checkboxes.values():
                checkbox.setChecked(True)
            self.sensitivity_slider.setValue(8)
            self.confidence_spinbox.setValue(80)
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        for checkbox in self.method_checkboxes.values():
            checkbox.setChecked(True)
        self.sensitivity_slider.setValue(5)
        self.confidence_spinbox.setValue(75)
    
    def export_settings(self):
        """Export current settings to file"""
        settings = self.get_settings()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Settings",
            "detection_settings.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                show_success_message(self, "Export Successful", f"Settings exported to {file_path}")
            except Exception as e:
                show_error_message(self, "Export Failed", f"Failed to export settings: {str(e)}")
    
    def import_settings(self):
        """Import settings from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Settings",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                # Apply loaded settings
                for method, checkbox in self.method_checkboxes.items():
                    checkbox.setChecked(method in settings.get('selected_methods', []))
                
                self.sensitivity_slider.setValue(settings.get('sensitivity', 5))
                self.confidence_spinbox.setValue(settings.get('confidence_threshold', 75))
                
                show_success_message(self, "Import Successful", f"Settings imported from {file_path}")
                
            except Exception as e:
                show_error_message(self, "Import Failed", f"Failed to import settings: {str(e)}")


class DetectionTab(QWidget):
    """Main detection tab widget"""
    
    detection_started = pyqtSignal()
    detection_finished = pyqtSignal(dict)
    detection_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    
    def __init__(self, detection_engine=None, ml_detection=None, model_manager=None, parent=None):
        super().__init__(parent)
        
        self.detection_engine = detection_engine
        self.ml_detection = ml_detection
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_image_path = None
        self.detection_worker = None
        self.detection_results = {}
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the detection tab UI"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel: Image display and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Center panel: Results display
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # Right panel: Settings
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (40% : 40% : 20%)
        main_splitter.setSizes([400, 400, 200])
    
    def create_left_panel(self) -> QWidget:
        """Create the left panel with image display and controls"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setStyleSheet(create_card_style())
        
        layout = QVBoxLayout(panel)
        
        # Image display
        image_group = QGroupBox("Image Analysis")
        image_layout = QVBoxLayout(image_group)
        
        self.image_display = ImageDisplayWidget()
        image_layout.addWidget(self.image_display)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet(f"""
            QLabel {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                color: white;
            }}
        """)
        image_layout.addWidget(self.image_info_label)
        
        layout.addWidget(image_group)
        
        # Control buttons
        controls_group = QGroupBox("Detection Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Main detection button
        self.detect_btn = QPushButton("🔍 Start Detection")
        self.detect_btn.setStyleSheet(create_button_style('primary'))
        self.detect_btn.setMinimumHeight(45)
        self.detect_btn.setEnabled(False)
        controls_layout.addWidget(self.detect_btn)
        
        # Quick detection button
        self.quick_detect_btn = QPushButton("⚡ Quick Detection")
        self.quick_detect_btn.setStyleSheet(create_button_style('secondary'))
        self.quick_detect_btn.setEnabled(False)
        controls_layout.addWidget(self.quick_detect_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                background-color: {get_current_theme_color('surface')};
                text-align: center;
                color: {get_current_theme_color('text')};
                font-weight: bold;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 {get_current_theme_color('primary')}, 
                                          stop: 1 #40E0D0);
                border-radius: 4px;
            }}
        """)
        controls_layout.addWidget(self.progress_bar)
        
        # Additional actions
        actions_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(create_button_style('secondary'))
        actions_layout.addWidget(self.clear_btn)
        
        self.save_results_btn = QPushButton("Save Results")
        self.save_results_btn.setStyleSheet(create_button_style('secondary'))
        self.save_results_btn.setEnabled(False)
        actions_layout.addWidget(self.save_results_btn)
        
        controls_layout.addLayout(actions_layout)
        
        layout.addWidget(controls_group)
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        """Create the center panel with results display"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setStyleSheet(create_card_style())
        
        layout = QVBoxLayout(panel)
        
        # Results display
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_widget = DetectionResultsWidget()
        results_layout.addWidget(self.results_widget)
        
        layout.addWidget(results_group)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with settings"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setStyleSheet(create_card_style())
        
        layout = QVBoxLayout(panel)
        
        # Settings widget
        self.settings_widget = DetectionSettingsWidget()
        
        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.settings_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(scroll_area)
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections"""
        # Image display connections
        self.image_display.image_dropped.connect(self.load_image)
        
        # Button connections
        self.detect_btn.clicked.connect(self.start_detection)
        self.quick_detect_btn.clicked.connect(self.quick_detection)
        self.clear_btn.clicked.connect(self.clear_all)
        self.save_results_btn.clicked.connect(self.save_results)
    
    def load_image(self, image_path: str):
        """Load an image for analysis"""
        try:
            if not os.path.exists(image_path):
                show_error_message(self, "Error", f"Image file not found: {image_path}")
                return
            
            self.current_image_path = image_path
            
            # Display image
            self.image_display.display_image(image_path)
            
            # Update image info
            file_info = Path(image_path)
            file_size = file_info.stat().st_size
            
            # Get image dimensions
            if PIL_AVAILABLE:
                try:
                    image = Image.open(image_path)
                    width, height = image.size
                    info_text = f"""
                    <b>File:</b> {file_info.name}<br>
                    <b>Size:</b> {file_size / 1024:.1f} KB<br>
                    <b>Dimensions:</b> {width} × {height}<br>
                    <b>Format:</b> {file_info.suffix.upper()}
                    """
                except Exception:
                    info_text = f"<b>File:</b> {file_info.name}<br><b>Error:</b> Could not read image"
            else:
                info_text = f"""
                <b>File:</b> {file_info.name}<br>
                <b>Size:</b> {file_size / 1024:.1f} KB<br>
                <b>Note:</b> Install Pillow for full image info
                """
            
            self.image_info_label.setText(info_text)
            
            # Enable detection buttons
            self.detect_btn.setEnabled(True)
            self.quick_detect_btn.setEnabled(True)
            
            self.log_message.emit(f"Image loaded: {file_info.name}")
            
        except Exception as e:
            show_error_message(self, "Error", f"Failed to load image: {str(e)}")
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to load image {image_path}: {str(e)}")
    
    def start_detection(self):
        """Start the detection process"""
        if not self.current_image_path:
            show_error_message(self, "Error", "Please load an image first")
            return
        
        selected_methods = self.settings_widget.get_selected_methods()
        if not selected_methods:
            show_error_message(self, "Error", "Please select at least one detection method")
            return
        
        # Disable UI during detection
        self.detect_btn.setEnabled(False)
        self.quick_detect_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get settings
        settings = self.settings_widget.get_settings()
        
        # Start detection worker
        self.detection_worker = DetectionWorker(
            self.current_image_path,
            selected_methods,
            settings
        )
        
        # Connect worker signals
        self.detection_worker.progress_updated.connect(self.progress_bar.setValue)
        self.detection_worker.detection_finished.connect(self.on_detection_finished)
        self.detection_worker.error_occurred.connect(self.on_detection_error)
        self.detection_worker.log_message.connect(self.on_log_message)
        
        # Start detection
        self.detection_started.emit()
        self.detection_worker.start()
    
    def quick_detection(self):
        """Run quick detection with preset methods"""
        # Set quick detection preset
        self.settings_widget.apply_preset("Fast")
        self.start_detection()
    
    def on_detection_finished(self, results: Dict[str, Any]):
        """Handle detection completion"""
        self.detection_results = results
        
        # Update results display
        self.results_widget.update_results(results)
        
        # Re-enable UI
        self.detect_btn.setEnabled(True)
        self.quick_detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_results_btn.setEnabled(True)
        
        # Emit finished signal
        self.detection_finished.emit(results)
        
        self.log_message.emit("Detection completed successfully")
    
    def on_detection_error(self, error_message: str):
        """Handle detection error"""
        show_error_message(self, "Detection Error", f"Detection failed: {error_message}")
        
        # Re-enable UI
        self.detect_btn.setEnabled(True)
        self.quick_detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_message.emit(f"Detection error: {error_message}")
    
    def on_log_message(self, message: str):
        """Handle log message from worker"""
        self.log_message.emit(message)
    
    def clear_all(self):
        """Clear all data and reset UI"""
        self.current_image_path = None
        self.detection_results = {}
        
        # Reset image display
        self.image_display.setText("Drop image here or click to browse\n\nSupported formats: PNG, JPG, BMP, TIFF")
        self.image_display.setPixmap(QPixmap())
        self.image_display.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {get_current_theme_color('border')};
                border-radius: 10px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
                font-size: 16px;
                font-weight: bold;
            }}
        """)
        
        # Reset info and results
        self.image_info_label.setText("No image loaded")
        self.results_widget.update_results({})
        
        # Disable buttons
        self.detect_btn.setEnabled(False)
        self.quick_detect_btn.setEnabled(False)
        self.save_results_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.log_message.emit("Interface cleared")
    
    def save_results(self):
        """Save detection results to file"""
        if not self.detection_results:
            show_error_message(self, "Error", "No results to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detection Results",
            f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Prepare results for saving
                save_data = {
                    'image_path': self.current_image_path,
                    'detection_timestamp': datetime.now().isoformat(),
                    'settings': self.settings_widget.get_settings(),
                    'results': self.detection_results
                }
                
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                show_success_message(self, "Success", f"Results saved to {file_path}")
                self.log_message.emit(f"Results saved to {os.path.basename(file_path)}")
                
            except Exception as e:
                show_error_message(self, "Error", f"Failed to save results: {str(e)}")
    
    # Method to allow external access for quick detection
    def run_quick_detection(self):
        """External interface for quick detection"""
        if self.current_image_path:
            self.quick_detection()
        else:
            show_error_message(self, "Error", "Please load an image first")


if __name__ == "__main__":
    # Test the detection tab
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create and show detection tab
    detection_tab = DetectionTab()
    detection_tab.show()
    
    sys.exit(app.exec_())