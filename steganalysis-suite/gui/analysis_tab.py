#!/usr/bin/env python3
"""
StegAnalysis Suite - Analysis Tab
Advanced forensic analysis and deep inspection interface
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import threading
import time
from datetime import datetime
import numpy as np

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
        QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
        QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
        QHeaderView, QFileDialog, QCheckBox, QComboBox, QSpinBox,
        QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QPlainTextEdit,
        QSizePolicy, QListWidget, QListWidgetItem
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
    from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QTextCursor
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
            QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
            QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
            QHeaderView, QFileDialog, QCheckBox, QComboBox, QSpinBox,
            QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QPlainTextEdit,
            QSizePolicy, QListWidget, QListWidgetItem
        )
        from PySide2.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer, Slot as pyqtSlot
        from PySide2.QtGui import QPixmap, QIcon, QFont, QPalette, QTextCursor
        PYQT_AVAILABLE = True
    except ImportError:
        PYQT_AVAILABLE = False

# Import utility functions
from .utils import (
    create_icon, create_card_style, create_button_style, create_table_style,
    show_error_message, show_success_message, scale_pixmap_to_fit,
    create_status_indicator, get_current_theme_color
)

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modules import ForensicAnalyzer
from models import ModelManager, create_model_evaluator


class AnalysisWorker(QThread):
    """Worker thread for running forensic analysis"""
    
    progress_updated = pyqtSignal(int)
    analysis_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    partial_results = pyqtSignal(str, dict)  # analysis_type, results
    
    def __init__(self, image_path: str, forensic_analyzer, analysis_types: List[str], settings: Dict[str, Any]):
        super().__init__()
        self.image_path = image_path
        self.forensic_analyzer = forensic_analyzer
        self.analysis_types = analysis_types
        self.settings = settings
        self.results = {}
        
    def run(self):
        """Run forensic analysis"""
        try:
            total_analyses = len(self.analysis_types)
            current_analysis = 0
            
            self.log_message.emit(f"Starting forensic analysis on: {os.path.basename(self.image_path)}")
            
            for analysis_type in self.analysis_types:
                current_analysis += 1
                progress = int((current_analysis / total_analyses) * 100)
                self.progress_updated.emit(progress)
                
                self.log_message.emit(f"Running {analysis_type} analysis...")
                
                try:
                    if analysis_type == "metadata_extraction":
                        result = self.forensic_analyzer.extract_metadata(self.image_path)
                    elif analysis_type == "histogram_analysis":
                        result = self.forensic_analyzer.histogram_analysis(self.image_path)
                    elif analysis_type == "noise_analysis":
                        result = self.forensic_analyzer.noise_analysis(self.image_path)
                    elif analysis_type == "compression_artifacts":
                        result = self.forensic_analyzer.compression_analysis(self.image_path)
                    elif analysis_type == "pixel_correlation":
                        result = self.forensic_analyzer.pixel_correlation_analysis(self.image_path)
                    elif analysis_type == "frequency_analysis":
                        result = self.forensic_analyzer.frequency_domain_analysis(self.image_path)
                    elif analysis_type == "statistical_tests":
                        result = self.forensic_analyzer.statistical_tests(self.image_path)
                    elif analysis_type == "visual_inspection":
                        result = self.forensic_analyzer.visual_inspection_analysis(self.image_path)
                    else:
                        result = {"error": f"Unknown analysis type: {analysis_type}"}
                    
                    self.results[analysis_type] = result
                    self.partial_results.emit(analysis_type, result)
                    
                except Exception as e:
                    error_result = {"error": str(e)}
                    self.results[analysis_type] = error_result
                    self.partial_results.emit(analysis_type, error_result)
                    self.log_message.emit(f"Error in {analysis_type}: {str(e)}")
                
                time.sleep(0.1)  # Small delay for UI responsiveness
            
            self.analysis_finished.emit(self.results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ForensicVisualizationWidget(QWidget):
    """Widget for displaying forensic analysis visualizations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.analysis_results = {}
        
    def init_ui(self):
        """Initialize the visualization UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different visualizations
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background-color: #2B2B2B;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #4A4A4A;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
                color: #FFFFFF;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #0078D4;
            }
        """)
        
        # Histogram tab
        self.histogram_widget = self.create_histogram_widget()
        self.viz_tabs.addTab(self.histogram_widget, "📊 Histogram")
        
        # Frequency analysis tab
        self.frequency_widget = self.create_frequency_widget()
        self.viz_tabs.addTab(self.frequency_widget, "🌊 Frequency")
        
        # Correlation tab
        self.correlation_widget = self.create_correlation_widget()
        self.viz_tabs.addTab(self.correlation_widget, "🔗 Correlation")
        
        # Noise analysis tab
        self.noise_widget = self.create_noise_widget()
        self.viz_tabs.addTab(self.noise_widget, "🔍 Noise")
        
        layout.addWidget(self.viz_tabs)
        
        # Export visualization button
        export_btn = QPushButton("📈 Export Visualizations")
        export_btn.setStyleSheet(create_button_style('secondary'))
        export_btn.clicked.connect(self.export_visualizations)
        layout.addWidget(export_btn)
    
    def create_histogram_widget(self) -> QWidget:
        """Create histogram visualization widget"""
        widget = QFrame()
        widget.setStyleSheet(create_card_style())
        layout = QVBoxLayout(widget)
        
        # Histogram display area
        self.histogram_display = QLabel()
        self.histogram_display.setAlignment(Qt.AlignCenter)
        self.histogram_display.setMinimumSize(400, 300)
        self.histogram_display.setStyleSheet(f"""
            QLabel {{
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
            }}
        """)
        self.histogram_display.setText("No histogram data available")
        layout.addWidget(self.histogram_display)
        
        # Histogram statistics
        self.histogram_stats = QTextEdit()
        self.histogram_stats.setMaximumHeight(150)
        self.histogram_stats.setStyleSheet(f"""
            QTextEdit {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }}
        """)
        self.histogram_stats.setReadOnly(True)
        layout.addWidget(self.histogram_stats)
        
        return widget
    
    def create_frequency_widget(self) -> QWidget:
        """Create frequency analysis widget"""
        widget = QFrame()
        widget.setStyleSheet(create_card_style())
        layout = QVBoxLayout(widget)
        
        # Frequency display area
        self.frequency_display = QLabel()
        self.frequency_display.setAlignment(Qt.AlignCenter)
        self.frequency_display.setMinimumSize(400, 300)
        self.frequency_display.setStyleSheet(f"""
            QLabel {{
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
            }}
        """)
        self.frequency_display.setText("No frequency analysis data available")
        layout.addWidget(self.frequency_display)
        
        # Frequency statistics
        self.frequency_stats = QTextEdit()
        self.frequency_stats.setMaximumHeight(150)
        self.frequency_stats.setStyleSheet(f"""
            QTextEdit {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }}
        """)
        self.frequency_stats.setReadOnly(True)
        layout.addWidget(self.frequency_stats)
        
        return widget
    
    def create_correlation_widget(self) -> QWidget:
        """Create correlation analysis widget"""
        widget = QFrame()
        widget.setStyleSheet(create_card_style())
        layout = QVBoxLayout(widget)
        
        # Correlation matrix display
        self.correlation_display = QLabel()
        self.correlation_display.setAlignment(Qt.AlignCenter)
        self.correlation_display.setMinimumSize(400, 300)
        self.correlation_display.setStyleSheet(f"""
            QLabel {{
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
            }}
        """)
        self.correlation_display.setText("No correlation data available")
        layout.addWidget(self.correlation_display)
        
        # Correlation statistics
        self.correlation_stats = QTextEdit()
        self.correlation_stats.setMaximumHeight(150)
        self.correlation_stats.setStyleSheet(f"""
            QTextEdit {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }}
        """)
        self.correlation_stats.setReadOnly(True)
        layout.addWidget(self.correlation_stats)
        
        return widget
    
    def create_noise_widget(self) -> QWidget:
        """Create noise analysis widget"""
        widget = QFrame()
        widget.setStyleSheet(create_card_style())
        layout = QVBoxLayout(widget)
        
        # Noise analysis display
        self.noise_display = QLabel()
        self.noise_display.setAlignment(Qt.AlignCenter)
        self.noise_display.setMinimumSize(400, 300)
        self.noise_display.setStyleSheet(f"""
            QLabel {{
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
            }}
        """)
        self.noise_display.setText("No noise analysis data available")
        layout.addWidget(self.noise_display)
        
        # Noise statistics
        self.noise_stats = QTextEdit()
        self.noise_stats.setMaximumHeight(150)
        self.noise_stats.setStyleSheet(f"""
            QTextEdit {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }}
        """)
        self.noise_stats.setReadOnly(True)
        layout.addWidget(self.noise_stats)
        
        return widget
    
    def update_visualizations(self, results: Dict[str, Any]):
        """Update visualizations with analysis results"""
        self.analysis_results = results
        
        # Update histogram
        if "histogram_analysis" in results:
            self.update_histogram_display(results["histogram_analysis"])
        
        # Update frequency analysis
        if "frequency_analysis" in results:
            self.update_frequency_display(results["frequency_analysis"])
        
        # Update correlation analysis
        if "pixel_correlation" in results:
            self.update_correlation_display(results["pixel_correlation"])
        
        # Update noise analysis
        if "noise_analysis" in results:
            self.update_noise_display(results["noise_analysis"])
    
    def update_histogram_display(self, histogram_data: Dict[str, Any]):
        """Update histogram visualization"""
        if "error" in histogram_data:
            self.histogram_display.setText(f"Error: {histogram_data['error']}")
            self.histogram_stats.setText("No statistics available due to error")
            return
        
        # Create histogram statistics text
        stats_text = "Histogram Analysis Results:\n"
        stats_text += "=" * 30 + "\n"
        
        if "channel_stats" in histogram_data:
            for channel, stats in histogram_data["channel_stats"].items():
                stats_text += f"\n{channel.upper()} Channel:\n"
                stats_text += f"  Mean: {stats.get('mean', 0):.2f}\n"
                stats_text += f"  Std Dev: {stats.get('std', 0):.2f}\n"
                stats_text += f"  Skewness: {stats.get('skewness', 0):.4f}\n"
                stats_text += f"  Kurtosis: {stats.get('kurtosis', 0):.4f}\n"
        
        if "anomalies" in histogram_data:
            stats_text += f"\nAnomalies Detected: {len(histogram_data['anomalies'])}\n"
            for anomaly in histogram_data["anomalies"][:5]:  # Show first 5
                stats_text += f"  - {anomaly}\n"
        
        self.histogram_stats.setText(stats_text)
        self.histogram_display.setText("Histogram visualization\n(Data processed successfully)")
    
    def update_frequency_display(self, frequency_data: Dict[str, Any]):
        """Update frequency analysis visualization"""
        if "error" in frequency_data:
            self.frequency_display.setText(f"Error: {frequency_data['error']}")
            self.frequency_stats.setText("No statistics available due to error")
            return
        
        # Create frequency statistics text
        stats_text = "Frequency Domain Analysis:\n"
        stats_text += "=" * 30 + "\n"
        
        if "dct_analysis" in frequency_data:
            dct_data = frequency_data["dct_analysis"]
            stats_text += f"\nDCT Analysis:\n"
            stats_text += f"  DC Component: {dct_data.get('dc_component', 0):.2f}\n"
            stats_text += f"  AC Energy: {dct_data.get('ac_energy', 0):.2f}\n"
            stats_text += f"  High Freq Ratio: {dct_data.get('high_freq_ratio', 0):.4f}\n"
        
        if "fft_analysis" in frequency_data:
            fft_data = frequency_data["fft_analysis"]
            stats_text += f"\nFFT Analysis:\n"
            stats_text += f"  Dominant Frequency: {fft_data.get('dominant_freq', 0):.2f}\n"
            stats_text += f"  Spectral Centroid: {fft_data.get('spectral_centroid', 0):.2f}\n"
        
        self.frequency_stats.setText(stats_text)
        self.frequency_display.setText("Frequency analysis visualization\n(Data processed successfully)")
    
    def update_correlation_display(self, correlation_data: Dict[str, Any]):
        """Update correlation analysis visualization"""
        if "error" in correlation_data:
            self.correlation_display.setText(f"Error: {correlation_data['error']}")
            self.correlation_stats.setText("No statistics available due to error")
            return
        
        # Create correlation statistics text
        stats_text = "Pixel Correlation Analysis:\n"
        stats_text += "=" * 30 + "\n"
        
        if "correlation_coefficients" in correlation_data:
            corr_data = correlation_data["correlation_coefficients"]
            stats_text += f"\nCorrelation Coefficients:\n"
            for direction, value in corr_data.items():
                stats_text += f"  {direction}: {value:.4f}\n"
        
        if "autocorrelation" in correlation_data:
            autocorr = correlation_data["autocorrelation"]
            stats_text += f"\nAutocorrelation Peak: {autocorr:.4f}\n"
        
        self.correlation_stats.setText(stats_text)
        self.correlation_display.setText("Correlation matrix visualization\n(Data processed successfully)")
    
    def update_noise_display(self, noise_data: Dict[str, Any]):
        """Update noise analysis visualization"""
        if "error" in noise_data:
            self.noise_display.setText(f"Error: {noise_data['error']}")
            self.noise_stats.setText("No statistics available due to error")
            return
        
        # Create noise statistics text
        stats_text = "Noise Analysis Results:\n"
        stats_text += "=" * 30 + "\n"
        
        if "noise_estimation" in noise_data:
            noise_est = noise_data["noise_estimation"]
            stats_text += f"\nNoise Level: {noise_est.get('level', 0):.4f}\n"
            stats_text += f"Noise Type: {noise_est.get('type', 'Unknown')}\n"
        
        if "snr" in noise_data:
            stats_text += f"Signal-to-Noise Ratio: {noise_data['snr']:.2f} dB\n"
        
        if "anomalies" in noise_data:
            stats_text += f"\nNoise Anomalies: {len(noise_data['anomalies'])}\n"
        
        self.noise_stats.setText(stats_text)
        self.noise_display.setText("Noise pattern visualization\n(Data processed successfully)")
    
    def export_visualizations(self):
        """Export all visualizations to files"""
        if not self.analysis_results:
            show_error_message(self, "Error", "No visualization data to export")
            return
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder_path:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export each visualization as text report
            for analysis_type, data in self.analysis_results.items():
                if "error" not in data:
                    filename = f"{analysis_type}_visualization_{timestamp}.txt"
                    filepath = os.path.join(folder_path, filename)
                    
                    with open(filepath, 'w') as f:
                        f.write(f"Visualization Export: {analysis_type}\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Export Date: {datetime.now().isoformat()}\n")
                        f.write(f"Analysis Data: {json.dumps(data, indent=2)}\n")
            
            show_success_message(self, "Export Complete", f"Visualizations exported to {folder_path}")
            
        except Exception as e:
            show_error_message(self, "Export Error", f"Failed to export visualizations: {str(e)}")


class MetadataDisplayWidget(QWidget):
    """Widget for displaying image metadata and EXIF information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the metadata display UI"""
        layout = QVBoxLayout(self)
        
        # Metadata tree widget
        self.metadata_tree = QTreeWidget()
        self.metadata_tree.setHeaderLabels(["Property", "Value"])
        self.metadata_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }}
            QTreeWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {get_current_theme_color('border')};
            }}
            QTreeWidget::item:selected {{
                background-color: {get_current_theme_color('primary')};
            }}
        """)
        
        # Enable alternating row colors
        self.metadata_tree.setAlternatingRowColors(True)
        self.metadata_tree.setRootIsDecorated(True)
        
        layout.addWidget(self.metadata_tree)
        
        # Metadata summary
        summary_group = QGroupBox("Metadata Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(120)
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }}
        """)
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Export metadata button
        export_btn = QPushButton("📄 Export Metadata")
        export_btn.setStyleSheet(create_button_style('secondary'))
        export_btn.clicked.connect(self.export_metadata)
        layout.addWidget(export_btn)
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update metadata display with extracted data"""
        self.metadata_tree.clear()
        self.metadata = metadata
        
        if "error" in metadata:
            error_item = QTreeWidgetItem(["Error", metadata["error"]])
            self.metadata_tree.addTopLevelItem(error_item)
            self.summary_text.setText("Failed to extract metadata due to error")
            return
        
        # Populate metadata tree
        for category, data in metadata.items():
            if isinstance(data, dict):
                category_item = QTreeWidgetItem([category.replace('_', ' ').title(), ""])
                category_item.setFont(0, QFont("", 11, QFont.Bold))
                
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=2)
                    else:
                        value_str = str(value)
                    
                    child_item = QTreeWidgetItem([str(key), value_str])
                    category_item.addChild(child_item)
                
                self.metadata_tree.addTopLevelItem(category_item)
                category_item.setExpanded(True)
            else:
                item = QTreeWidgetItem([category.replace('_', ' ').title(), str(data)])
                self.metadata_tree.addTopLevelItem(item)
        
        # Resize columns
        self.metadata_tree.resizeColumnToContents(0)
        
        # Update summary
        self.update_summary(metadata)
    
    def update_summary(self, metadata: Dict[str, Any]):
        """Update metadata summary"""
        summary_lines = []
        
        # Basic information
        if "basic_info" in metadata:
            basic = metadata["basic_info"]
            summary_lines.append(f"File Size: {basic.get('file_size', 'Unknown')}")
            summary_lines.append(f"Dimensions: {basic.get('width', '?')} × {basic.get('height', '?')}")
            summary_lines.append(f"Format: {basic.get('format', 'Unknown')}")
            summary_lines.append(f"Mode: {basic.get('mode', 'Unknown')}")
        
        # EXIF information
        if "exif" in metadata and metadata["exif"]:
            exif_count = len(metadata["exif"])
            summary_lines.append(f"EXIF Tags: {exif_count} found")
            
            # Look for camera information
            exif_data = metadata["exif"]
            if "Make" in exif_data and "Model" in exif_data:
                summary_lines.append(f"Camera: {exif_data['Make']} {exif_data['Model']}")
        
        # Security assessment
        security_notes = []
        if "exif" in metadata and metadata["exif"]:
            if any(key in metadata["exif"] for key in ["GPS", "GPSInfo", "DateTime"]):
                security_notes.append("⚠️ Contains potentially sensitive metadata")
        
        if "suspicious_fields" in metadata:
            security_notes.append(f"⚠️ {len(metadata['suspicious_fields'])} suspicious fields detected")
        
        summary_text = "\n".join(summary_lines)
        if security_notes:
            summary_text += "\n\nSecurity Notes:\n" + "\n".join(security_notes)
        
        self.summary_text.setText(summary_text)
    
    def export_metadata(self):
        """Export metadata to file"""
        if not hasattr(self, 'metadata'):
            show_error_message(self, "Error", "No metadata to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metadata",
            f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.metadata, f, indent=2, default=str)
                else:
                    # Export as formatted text
                    with open(file_path, 'w') as f:
                        f.write("Image Metadata Report\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Export Date: {datetime.now().isoformat()}\n\n")
                        
                        def write_dict(d, indent=0):
                            for key, value in d.items():
                                if isinstance(value, dict):
                                    f.write("  " * indent + f"{key}:\n")
                                    write_dict(value, indent + 1)
                                else:
                                    f.write("  " * indent + f"{key}: {value}\n")
                        
                        write_dict(self.metadata)
                
                show_success_message(self, "Export Complete", f"Metadata exported to {file_path}")
                
            except Exception as e:
                show_error_message(self, "Export Error", f"Failed to export metadata: {str(e)}")


class AnalysisTab(QWidget):
    """Main analysis tab widget for forensic analysis"""
    
    analysis_started = pyqtSignal()
    analysis_finished = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    
    def __init__(self, forensic_analyzer, model_manager, parent=None):
        super().__init__(parent)
        
        self.forensic_analyzer = forensic_analyzer
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_image_path = None
        self.analysis_worker = None
        self.analysis_results = {}
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the analysis tab UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setSizes([300, 700])
        main_layout.addWidget(main_splitter)
        
        # Left panel - Controls and configuration
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Analysis results and visualizations
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Status bar
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)
    
    def create_left_panel(self) -> QWidget:
        """Create the left control panel"""
        panel = QFrame()
        panel.setStyleSheet(create_card_style())
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("🔬 Forensic Analysis")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Image selection section
        image_group = QGroupBox("Image Selection")
        image_layout = QVBoxLayout(image_group)
        
        # Current image display
        self.current_image_label = QLabel("No image selected")
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setMinimumHeight(100)
        self.current_image_label.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {get_current_theme_color('border')};
                border-radius: 8px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
                padding: 20px;
            }}
        """)
        image_layout.addWidget(self.current_image_label)
        
        # Browse button
        self.browse_btn = QPushButton("📁 Browse Image")
        self.browse_btn.setStyleSheet(create_button_style('primary'))
        self.browse_btn.clicked.connect(self.browse_image)
        image_layout.addWidget(self.browse_btn)
        
        layout.addWidget(image_group)
        
        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout(options_group)
        
        # Analysis type checkboxes
        self.analysis_checkboxes = {}
        analysis_types = [
            ("metadata_extraction", "📄 Metadata Extraction"),
            ("histogram_analysis", "📊 Histogram Analysis"),
            ("noise_analysis", "🔍 Noise Analysis"),
            ("compression_artifacts", "🗜️ Compression Artifacts"),
            ("pixel_correlation", "🔗 Pixel Correlation"),
            ("frequency_analysis", "🌊 Frequency Analysis"),
            ("statistical_tests", "📈 Statistical Tests"),
            ("visual_inspection", "👁️ Visual Inspection")
        ]
        
        for analysis_type, label in analysis_types:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)  # Default all checked
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 12px;
                    padding: 4px;
                    color: {get_current_theme_color('text')};
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border-radius: 3px;
                    border: 2px solid {get_current_theme_color('border')};
                    background-color: {get_current_theme_color('surface')};
                }}
                QCheckBox::indicator:checked {{
                    background-color: {get_current_theme_color('primary')};
                    border-color: {get_current_theme_color('primary')};
                }}
            """)
            self.analysis_checkboxes[analysis_type] = checkbox
            options_layout.addWidget(checkbox)
        
        layout.addWidget(options_group)
        
        # Analysis controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Start analysis button
        self.start_analysis_btn = QPushButton("🚀 Start Analysis")
        self.start_analysis_btn.setStyleSheet(create_button_style('success'))
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.start_analysis_btn.setEnabled(False)
        controls_layout.addWidget(self.start_analysis_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 5px;
                text-align: center;
                background-color: {get_current_theme_color('surface')};
            }}
            QProgressBar::chunk {{
                background-color: {get_current_theme_color('success')};
                border-radius: 4px;
            }}
        """)
        controls_layout.addWidget(self.progress_bar)
        
        # Stop analysis button
        self.stop_analysis_btn = QPushButton("⏹️ Stop Analysis")
        self.stop_analysis_btn.setStyleSheet(create_button_style('danger'))
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setVisible(False)
        controls_layout.addWidget(self.stop_analysis_btn)
        
        layout.addWidget(controls_group)
        
        # Settings section
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Sensitivity slider
        settings_layout.addWidget(QLabel("Sensitivity:"), 0, 0)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(1)
        settings_layout.addWidget(self.sensitivity_slider, 0, 1)
        
        self.sensitivity_label = QLabel("5")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(str(v))
        )
        settings_layout.addWidget(self.sensitivity_label, 0, 2)
        
        # Depth setting
        settings_layout.addWidget(QLabel("Analysis Depth:"), 1, 0)
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["Quick", "Standard", "Deep", "Comprehensive"])
        self.depth_combo.setCurrentText("Standard")
        settings_layout.addWidget(self.depth_combo, 1, 1, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right results panel"""
        panel = QFrame()
        panel.setStyleSheet(create_card_style())
        layout = QVBoxLayout(panel)
        
        # Results tab widget
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background-color: #2B2B2B;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #4A4A4A;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
                color: #FFFFFF;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #0078D4;
            }
        """)
        
        # Analysis results tab
        self.results_widget = self.create_results_widget()
        self.results_tabs.addTab(self.results_widget, "📊 Results")
        
        # Visualizations tab
        self.visualizations_widget = ForensicVisualizationWidget()
        self.results_tabs.addTab(self.visualizations_widget, "📈 Visualizations")
        
        # Metadata tab
        self.metadata_widget = MetadataDisplayWidget()
        self.results_tabs.addTab(self.metadata_widget, "📄 Metadata")
        
        # Log tab
        self.log_widget = self.create_log_widget()
        self.results_tabs.addTab(self.log_widget, "📝 Logs")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def create_results_widget(self) -> QWidget:
        """Create the analysis results widget"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        
        # Results summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }}
        """)
        self.summary_text.setReadOnly(True)
        self.summary_text.setText("No analysis performed yet")
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Detailed results table
        results_group = QGroupBox("Detailed Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Analysis Type", "Status", "Score", "Details"])
        self.results_table.setStyleSheet(create_table_style())
        
        # Set column widths
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_json_btn = QPushButton("📄 Export JSON")
        self.export_json_btn.setStyleSheet(create_button_style('secondary'))
        self.export_json_btn.clicked.connect(self.export_results_json)
        self.export_json_btn.setEnabled(False)
        export_layout.addWidget(self.export_json_btn)
        
        self.export_report_btn = QPushButton("📋 Generate Report")
        self.export_report_btn.setStyleSheet(create_button_style('secondary'))
        self.export_report_btn.clicked.connect(self.generate_detailed_report)
        self.export_report_btn.setEnabled(False)
        export_layout.addWidget(self.export_report_btn)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        return widget
    
    def create_log_widget(self) -> QWidget:
        """Create the log display widget"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        
        # Log controls
        controls_layout = QHBoxLayout()
        
        clear_log_btn = QPushButton("🗑️ Clear Log")
        clear_log_btn.setStyleSheet(create_button_style('secondary'))
        clear_log_btn.clicked.connect(self.clear_log)
        controls_layout.addWidget(clear_log_btn)
        
        save_log_btn = QPushButton("💾 Save Log")
        save_log_btn.setStyleSheet(create_button_style('secondary'))
        save_log_btn.clicked.connect(self.save_log)
        controls_layout.addWidget(save_log_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Log display
        self.log_display = QPlainTextEdit()
        self.log_display.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }}
        """)
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        return widget
    
    def create_status_bar(self) -> QWidget:
        """Create status bar"""
        status_bar = QFrame()
        status_bar.setFixedHeight(30)
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {get_current_theme_color('surface')};
                border-top: 1px solid {get_current_theme_color('border')};
                padding: 5px 10px;
            }}
        """)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {get_current_theme_color('text_secondary')};")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Analysis time label
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {get_current_theme_color('text_secondary')};")
        layout.addWidget(self.time_label)
        
        return status_bar
    
    def setup_connections(self):
        """Setup signal connections"""
        pass
    
    def browse_image(self):
        """Browse for image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Analysis",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All Files (*)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path: str):
        """Load image for analysis"""
        try:
            self.current_image_path = file_path
            
            # Update image display
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = scale_pixmap_to_fit(pixmap, 200, 100)
                self.current_image_label.setPixmap(scaled_pixmap)
            else:
                self.current_image_label.setText(f"📄 {os.path.basename(file_path)}")
            
            # Enable start button
            self.start_analysis_btn.setEnabled(True)
            
            # Update status
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            
            # Add to log
            self.add_log_message(f"Image loaded: {file_path}")
            
        except Exception as e:
            show_error_message(self, "Error", f"Failed to load image: {str(e)}")
            self.add_log_message(f"Error loading image: {str(e)}")
    
    def start_analysis(self):
        """Start forensic analysis"""
        if not self.current_image_path:
            show_error_message(self, "Error", "No image selected for analysis")
            return
        
        # Get selected analysis types
        selected_analyses = []
        for analysis_type, checkbox in self.analysis_checkboxes.items():
            if checkbox.isChecked():
                selected_analyses.append(analysis_type)
        
        if not selected_analyses:
            show_error_message(self, "Error", "No analysis types selected")
            return
        
        # Get settings
        settings = {
            'sensitivity': self.sensitivity_slider.value(),
            'depth': self.depth_combo.currentText().lower()
        }
        
        # Update UI for analysis state
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.analysis_results = {}
        self.update_results_display()
        
        # Start analysis worker
        self.analysis_worker = AnalysisWorker(
            self.current_image_path,
            self.forensic_analyzer,
            selected_analyses,
            settings
        )
        
        # Connect signals
        self.analysis_worker.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_worker.analysis_finished.connect(self.on_analysis_finished)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.log_message.connect(self.add_log_message)
        self.analysis_worker.partial_results.connect(self.on_partial_results)
        
        # Start the worker
        self.analysis_worker.start()
        
        # Update status
        self.status_label.setText("Running analysis...")
        self.add_log_message("Analysis started")
        
        # Start timer for elapsed time
        self.analysis_start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        self.timer.start(1000)  # Update every second
        
        self.analysis_started.emit()
    
    def stop_analysis(self):
        """Stop current analysis"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.terminate()
            self.analysis_worker.wait()
        
        self.reset_analysis_ui()
        self.add_log_message("Analysis stopped by user")
        self.status_label.setText("Analysis stopped")
    
    def on_analysis_finished(self, results: Dict[str, Any]):
        """Handle analysis completion"""
        self.analysis_results = results
        self.reset_analysis_ui()
        
        # Update displays
        self.update_results_display()
        self.visualizations_widget.update_visualizations(results)
        
        # Update metadata if available
        if "metadata_extraction" in results:
            self.metadata_widget.update_metadata(results["metadata_extraction"])
        
        # Enable export buttons
        self.export_json_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)
        
        # Update status
        elapsed_time = time.time() - self.analysis_start_time
        self.status_label.setText(f"Analysis complete - {elapsed_time:.1f}s")
        self.add_log_message(f"Analysis completed in {elapsed_time:.1f} seconds")
        
        # Stop timer
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        self.analysis_finished.emit(results)
    
    def on_analysis_error(self, error_message: str):
        """Handle analysis error"""
        self.reset_analysis_ui()
        show_error_message(self, "Analysis Error", f"Analysis failed: {error_message}")
        self.add_log_message(f"Analysis error: {error_message}")
        self.status_label.setText("Analysis failed")
        
        if hasattr(self, 'timer'):
            self.timer.stop()
    
    def on_partial_results(self, analysis_type: str, results: Dict[str, Any]):
        """Handle partial analysis results"""
        self.analysis_results[analysis_type] = results
        self.update_results_display()
        self.add_log_message(f"Completed: {analysis_type}")
    
    def reset_analysis_ui(self):
        """Reset UI to non-analysis state"""
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self.time_label.setText("")
    
    def update_elapsed_time(self):
        """Update elapsed time display"""
        if hasattr(self, 'analysis_start_time'):
            elapsed = time.time() - self.analysis_start_time
            self.time_label.setText(f"Elapsed: {elapsed:.1f}s")
    
    def update_results_display(self):
        """Update the results display"""
        # Update summary
        if self.analysis_results:
            summary_text = f"Analysis Results for: {os.path.basename(self.current_image_path)}\n"
            summary_text += "=" * 50 + "\n"
            
            total_analyses = len(self.analysis_results)
            successful_analyses = sum(1 for r in self.analysis_results.values() if "error" not in r)
            
            summary_text += f"Total Analyses: {total_analyses}\n"
            summary_text += f"Successful: {successful_analyses}\n"
            summary_text += f"Failed: {total_analyses - successful_analyses}\n"
            
            # Overall assessment
            if successful_analyses > 0:
                summary_text += "\nOverall Assessment: Analysis completed with results available"
            else:
                summary_text += "\nOverall Assessment: Analysis failed or incomplete"
            
            self.summary_text.setText(summary_text)
        
        # Update results table
        self.results_table.setRowCount(len(self.analysis_results))
        
        for row, (analysis_type, results) in enumerate(self.analysis_results.items()):
            # Analysis type
            type_item = QTableWidgetItem(analysis_type.replace('_', ' ').title())
            self.results_table.setItem(row, 0, type_item)
            
            # Status
            if "error" in results:
                status_item = QTableWidgetItem("❌ Error")
                status_item.setBackground(Qt.red)
                score_item = QTableWidgetItem("N/A")
                details_item = QTableWidgetItem(results["error"])
            else:
                status_item = QTableWidgetItem("✅ Success")
                status_item.setBackground(Qt.green)
                
                # Try to extract score/confidence
                score = "N/A"
                if "confidence" in results:
                    score = f"{results['confidence']:.2f}"
                elif "score" in results:
                    score = f"{results['score']:.2f}"
                
                score_item = QTableWidgetItem(score)
                
                # Create summary of details
                details = []
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key not in ["error", "confidence", "score"] and not key.startswith("_"):
                            if isinstance(value, (int, float)):
                                details.append(f"{key}: {value:.2f}")
                            elif isinstance(value, str) and len(value) < 50:
                                details.append(f"{key}: {value}")
                
                details_text = "; ".join(details[:3])  # Show first 3 details
                if len(details) > 3:
                    details_text += "..."
                
                details_item = QTableWidgetItem(details_text)
            
            self.results_table.setItem(row, 1, status_item)
            self.results_table.setItem(row, 2, score_item)
            self.results_table.setItem(row, 3, details_item)
    
    def add_log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_display.appendPlainText(log_entry)
        self.log_message.emit(message)
    
    def clear_log(self):
        """Clear the log display"""
        self.log_display.clear()
    
    def save_log(self):
        """Save log to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log",
            f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.log_display.toPlainText())
                show_success_message(self, "Log Saved", f"Log saved to {file_path}")
            except Exception as e:
                show_error_message(self, "Save Error", f"Failed to save log: {str(e)}")
    
    def export_results_json(self):
        """Export results as JSON"""
        if not self.analysis_results:
            show_error_message(self, "Error", "No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                export_data = {
                    'image_path': self.current_image_path,
                    'export_timestamp': datetime.now().isoformat(),
                    'analysis_results': self.analysis_results
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                show_success_message(self, "Export Complete", f"Results exported to {file_path}")
                
            except Exception as e:
                show_error_message(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def generate_detailed_report(self):
        """Generate a detailed forensic report"""
        if not self.analysis_results:
            show_error_message(self, "Error", "No results to generate report from")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Generate Report",
            f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            try:
                # Generate HTML report
                html_content = self.generate_html_report()
                
                if file_path.endswith('.html'):
                    with open(file_path, 'w') as f:
                        f.write(html_content)
                elif file_path.endswith('.pdf'):
                    # Convert HTML to PDF (requires additional dependencies)
                    show_error_message(self, "Error", "PDF export not implemented yet. Please use HTML format.")
                    return
                else:
                    # Default to HTML
                    with open(file_path, 'w') as f:
                        f.write(html_content)
                
                show_success_message(self, "Report Generated", f"Report saved to {file_path}")
                
            except Exception as e:
                show_error_message(self, "Report Error", f"Failed to generate report: {str(e)}")
    
    def generate_html_report(self) -> str:
        """Generate HTML forensic report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
                .section { margin: 30px 0; }
                .section h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
                .result-item { margin: 15px 0; padding: 15px; border-left: 4px solid #007acc; background: #f8f9fa; }
                .error { border-left-color: #dc3545; }
                .success { border-left-color: #28a745; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metadata { font-family: monospace; white-space: pre-wrap; }
            </style>
        </head>
        <body>
        """
        
        # Header
        html += f"""
        <div class="header">
            <h1>Forensic Analysis Report</h1>
            <p><strong>Image:</strong> {os.path.basename(self.current_image_path)}</p>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        # Executive Summary
        html += """
        <div class="section">
            <h2>Executive Summary</h2>
        """
        
        total_analyses = len(self.analysis_results)
        successful_analyses = sum(1 for r in self.analysis_results.values() if "error" not in r)
        
        html += f"""
            <p>This report presents the results of a comprehensive forensic analysis performed on the image file.</p>
            <ul>
                <li>Total Analyses Performed: {total_analyses}</li>
                <li>Successful Analyses: {successful_analyses}</li>
                <li>Failed Analyses: {total_analyses - successful_analyses}</li>
            </ul>
        </div>
        """
        
        # Detailed Results
        html += """
        <div class="section">
            <h2>Detailed Analysis Results</h2>
        """
        
        for analysis_type, results in self.analysis_results.items():
            analysis_name = analysis_type.replace('_', ' ').title()
            
            if "error" in results:
                html += f"""
                <div class="result-item error">
                    <h3>❌ {analysis_name}</h3>
                    <p><strong>Status:</strong> Failed</p>
                    <p><strong>Error:</strong> {results['error']}</p>
                </div>
                """
            else:
                html += f"""
                <div class="result-item success">
                    <h3>✅ {analysis_name}</h3>
                    <p><strong>Status:</strong> Completed Successfully</p>
                """
                
                # Add specific details for each analysis type
                if analysis_type == "metadata_extraction" and "basic_info" in results:
                    basic_info = results["basic_info"]
                    html += f"""
                    <p><strong>File Size:</strong> {basic_info.get('file_size', 'Unknown')}</p>
                    <p><strong>Dimensions:</strong> {basic_info.get('width', '?')} × {basic_info.get('height', '?')}</p>
                    <p><strong>Format:</strong> {basic_info.get('format', 'Unknown')}</p>
                    """
                elif analysis_type == "histogram_analysis" and "channel_stats" in results:
                    html += "<p><strong>Channel Statistics:</strong></p><ul>"
                    for channel, stats in results["channel_stats"].items():
                        html += f"<li>{channel.upper()}: Mean={stats.get('mean', 0):.2f}, Std={stats.get('std', 0):.2f}</li>"
                    html += "</ul>"
                
                # Add raw data section
                html += f"""
                <details>
                    <summary>Raw Analysis Data</summary>
                    <pre class="metadata">{json.dumps(results, indent=2, default=str)}</pre>
                </details>
                """
                
                html += "</div>"
        
        html += "</div>"
        
        # Technical Details
        html += f"""
        <div class="section">
            <h2>Technical Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Image Path</td><td>{self.current_image_path}</td></tr>
                <tr><td>Analysis Timestamp</td><td>{datetime.now().isoformat()}</td></tr>
                <tr><td>Tool Version</td><td>StegAnalysis Suite v1.0</td></tr>
            </table>
        </div>
        """
        
        # Footer
        html += """
        <div class="section">
            <hr>
            <p><em>This report was generated automatically by the StegAnalysis Suite. 
            For questions about this analysis, please consult the documentation or contact support.</em></p>
        </div>
        </body>
        </html>
        """
        
        return html


class AnalysisSettingsDialog(QWidget):
    """Dialog for advanced analysis settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Analysis Settings")
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        """Initialize settings dialog UI"""
        layout = QVBoxLayout(self)
        
        # Algorithm-specific settings
        algorithms_group = QGroupBox("Algorithm Settings")
        algorithms_layout = QGridLayout(algorithms_group)
        
        # LSB Detection settings
        algorithms_layout.addWidget(QLabel("LSB Detection Threshold:"), 0, 0)
        self.lsb_threshold = QSpinBox()
        self.lsb_threshold.setRange(1, 100)
        self.lsb_threshold.setValue(10)
        algorithms_layout.addWidget(self.lsb_threshold, 0, 1)
        
        # Chi-Square settings
        algorithms_layout.addWidget(QLabel("Chi-Square Confidence:"), 1, 0)
        self.chi_confidence = QComboBox()
        self.chi_confidence.addItems(["90%", "95%", "99%"])
        self.chi_confidence.setCurrentText("95%")
        algorithms_layout.addWidget(self.chi_confidence, 1, 1)
        
        # Frequency analysis settings
        algorithms_layout.addWidget(QLabel("Frequency Bands:"), 2, 0)
        self.frequency_bands = QSpinBox()
        self.frequency_bands.setRange(8, 64)
        self.frequency_bands.setValue(32)
        algorithms_layout.addWidget(self.frequency_bands, 2, 1)
        
        layout.addWidget(algorithms_group)
        
        # Performance settings
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QGridLayout(performance_group)
        
        performance_layout.addWidget(QLabel("CPU Threads:"), 0, 0)
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, 16)
        self.cpu_threads.setValue(4)
        performance_layout.addWidget(self.cpu_threads, 0, 1)
        
        performance_layout.addWidget(QLabel("Memory Limit (MB):"), 1, 0)
        self.memory_limit = QSpinBox()
        self.memory_limit.setRange(256, 8192)
        self.memory_limit.setValue(2048)
        performance_layout.addWidget(self.memory_limit, 1, 1)
        
        layout.addWidget(performance_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)
        
        self.save_intermediate = QCheckBox("Save Intermediate Results")
        output_layout.addWidget(self.save_intermediate, 0, 0, 1, 2)
        
        self.generate_plots = QCheckBox("Generate Visualization Plots")
        self.generate_plots.setChecked(True)
        output_layout.addWidget(self.generate_plots, 1, 0, 1, 2)
        
        self.verbose_logging = QCheckBox("Verbose Logging")
        output_layout.addWidget(self.verbose_logging, 2, 0, 1, 2)
        
        layout.addWidget(output_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.setStyleSheet(create_button_style('primary'))
        buttons_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setStyleSheet(create_button_style('secondary'))
        buttons_layout.addWidget(self.reset_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet(create_button_style('secondary'))
        buttons_layout.addWidget(self.close_btn)
        
        layout.addLayout(buttons_layout)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings as dictionary"""
        return {
            'lsb_threshold': self.lsb_threshold.value(),
            'chi_confidence': self.chi_confidence.currentText(),
            'frequency_bands': self.frequency_bands.value(),
            'cpu_threads': self.cpu_threads.value(),
            'memory_limit': self.memory_limit.value(),
            'save_intermediate': self.save_intermediate.isChecked(),
            'generate_plots': self.generate_plots.isChecked(),
            'verbose_logging': self.verbose_logging.isChecked()
        }
    
    def set_settings(self, settings: Dict[str, Any]):
        """Apply settings from dictionary"""
        if 'lsb_threshold' in settings:
            self.lsb_threshold.setValue(settings['lsb_threshold'])
        if 'chi_confidence' in settings:
            self.chi_confidence.setCurrentText(settings['chi_confidence'])
        if 'frequency_bands' in settings:
            self.frequency_bands.setValue(settings['frequency_bands'])
        if 'cpu_threads' in settings:
            self.cpu_threads.setValue(settings['cpu_threads'])
        if 'memory_limit' in settings:
            self.memory_limit.setValue(settings['memory_limit'])
        if 'save_intermediate' in settings:
            self.save_intermediate.setChecked(settings['save_intermediate'])
        if 'generate_plots' in settings:
            self.generate_plots.setChecked(settings['generate_plots'])
        if 'verbose_logging' in settings:
            self.verbose_logging.setChecked(settings['verbose_logging'])


# Export classes for use in main application
__all__ = [
    'AnalysisTab',
    'AnalysisWorker', 
    'ForensicVisualizationWidget',
    'MetadataDisplayWidget',
    'AnalysisSettingsDialog'
]


if __name__ == "__main__":
    # Test code for standalone execution
    import sys
    from PyQt5.QtWidgets import QApplication
    
    class MockForensicAnalyzer:
        def extract_metadata(self, path):
            return {"basic_info": {"width": 800, "height": 600, "format": "JPEG"}}
        
        def histogram_analysis(self, path):
            return {"channel_stats": {"r": {"mean": 128, "std": 64}}}
        
        def noise_analysis(self, path):
            return {"noise_level": 0.05}
        
        def compression_analysis(self, path):
            return {"compression_ratio": 0.8}
        
        def pixel_correlation_analysis(self, path):
            return {"correlation_coefficients": {"horizontal": 0.95}}
        
        def frequency_domain_analysis(self, path):
            return {"dct_analysis": {"dc_component": 150}}
        
        def statistical_tests(self, path):
            return {"chi_square": {"p_value": 0.05}}
        
        def visual_inspection_analysis(self, path):
            return {"anomalies_detected": 2}
    
    class MockModelManager:
        pass
    
    app = QApplication(sys.argv)
    
    forensic_analyzer = MockForensicAnalyzer()
    model_manager = MockModelManager()
    
    analysis_tab = AnalysisTab(forensic_analyzer, model_manager)
    analysis_tab.show()
    
    sys.exit(app.exec_())