#!/usr/bin/env python3
"""
StegAnalysis Suite - Main GUI Window (Final)
Professional and modern interface for steganography detection with full functionality
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QMenuBar, QStatusBar, QToolBar, QAction, QLabel,
        QSplitter, QFrame, QProgressBar, QTextEdit, QGroupBox,
        QMessageBox, QFileDialog, QSystemTrayIcon, QStyle
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
    from PyQt5.QtGui import QIcon, QPixmap, QFont, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    raise ImportError("PyQt5 required for GUI. Install with: pip install PyQt5")

# Import the fixed detection tab
try:
    from detection_tab import DetectionTab
    DETECTION_TAB_AVAILABLE = True
except ImportError:
    print("Warning: Detection tab not found. Please ensure detection_tab.py is in the same directory.")
    DETECTION_TAB_AVAILABLE = False


class StegAnalysisMainWindow(QMainWindow):
    """Main window for StegAnalysis Suite with professional UI and enhanced functionality"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Settings
        self.settings = QSettings('StegAnalysis', 'Suite')
        
        # UI Components
        self.central_widget = None
        self.tab_widget = None
        self.status_bar = None
        self.progress_bar = None
        self.log_display = None
        self.log_panel = None
        
        # Tabs
        self.detection_tab = None
        self.analysis_tab = None
        self.reports_tab = None
        
        # Initialize UI
        self.init_ui()
        self.setup_theme()
        self.restore_settings()
        
        self.logger.info("StegAnalysis Suite GUI initialized successfully")
    
    def init_ui(self):
        """Initialize the user interface"""
        
        # Set window properties
        self.setWindowTitle("StegAnalysis Suite v1.0 - Professional Steganography Detection")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set window icon (if available)
        try:
            self.setWindowIcon(QIcon('icon.png'))
        except:
            pass
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for main content and log panel
        self.main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.main_splitter)
        
        # Create tab widget
        self.create_tab_widget()
        self.main_splitter.addWidget(self.tab_widget)
        
        # Create bottom panel (logs and status)
        self.log_panel = self.create_bottom_panel()
        self.main_splitter.addWidget(self.log_panel)
        
        # Set splitter proportions
        self.main_splitter.setSizes([600, 200])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
        
        # Create toolbar
        self.create_toolbar()
    
    def create_tab_widget(self):
        """Create the main tab widget with all detection interfaces"""
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(False)
        
        # Create detection tab
        if DETECTION_TAB_AVAILABLE:
            self.detection_tab = DetectionTab()
            # Connect log messages from detection tab to main window
            self.detection_tab.log_message.connect(self.add_log_message)
            self.detection_tab.detection_started.connect(lambda: self.status_label.setText("Running detection..."))
            self.detection_tab.detection_finished.connect(lambda r: self.status_label.setText("Detection completed"))
        else:
            self.detection_tab = self.create_basic_detection_tab()
        
        # Create analysis and reports tabs
        self.analysis_tab = self.create_analysis_tab()
        self.reports_tab = self.create_reports_tab()
        
        # Add tabs
        self.tab_widget.addTab(self.detection_tab, "🔍 Detection")
        self.tab_widget.addTab(self.analysis_tab, "📊 Analysis")
        self.tab_widget.addTab(self.reports_tab, "📋 Reports")
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def create_basic_detection_tab(self):
        """Create basic detection tab if enhanced version is not available"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Header
        header = QLabel("🔍 Steganography Detection")
        header.setFont(QFont("Arial", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #0078D4; margin-bottom: 30px;")
        layout.addWidget(header)
        
        # Error message
        error_group = QGroupBox("❌ Detection Module Not Found")
        error_layout = QVBoxLayout(error_group)
        
        message = QLabel("""
        <div style="font-size: 16px; line-height: 1.6;">
        <p><b>The detection module could not be loaded.</b></p>
        
        <p><b>To fix this issue:</b></p>
        <ol>
        <li>Ensure <code>detection_tab.py</code> is in the same directory as this file</li>
        <li>Install required dependencies: <code>pip install Pillow numpy</code></li>
        <li>Restart the application</li>
        </ol>
        
        <p><b>For command line detection:</b></p>
        <p><code>python main.py --detect --image your_image.jpg</code></p>
        </div>
        """)
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignLeft)
        message.setStyleSheet("color: #CCCCCC; padding: 20px; background-color: #3A3A3A; border-radius: 8px;")
        error_layout.addWidget(message)
        
        layout.addWidget(error_group)
        
        # Instructions for setup
        setup_group = QGroupBox("🔧 Setup Instructions")
        setup_layout = QVBoxLayout(setup_group)
        
        setup_text = QLabel("""
        <div style="font-size: 14px; line-height: 1.5;">
        <p><b>Required Files:</b></p>
        <ul>
        <li><code>main_gui.py</code> (this file)</li>
        <li><code>detection_tab.py</code> (detection interface)</li>
        </ul>
        
        <p><b>Required Packages:</b></p>
        <ul>
        <li>PyQt5: <code>pip install PyQt5</code></li>
        <li>Pillow: <code>pip install Pillow</code></li>
        <li>NumPy: <code>pip install numpy</code></li>
        </ul>
        </div>
        """)
        setup_text.setWordWrap(True)
        setup_text.setStyleSheet("color: #CCCCCC; padding: 15px;")
        setup_layout.addWidget(setup_text)
        
        layout.addWidget(setup_group)
        layout.addStretch()
        
        return widget
    
    def create_analysis_tab(self):
        """Create enhanced analysis tab"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("📊 Advanced Forensic Analysis")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #0078D4; margin-bottom: 20px;")
        layout.addWidget(header)
        
        # Analysis features
        features_group = QGroupBox("🔬 Available Analysis Features")
        features_layout = QVBoxLayout(features_group)
        
        features_text = """
        <div style="font-size: 14px; line-height: 1.6;">
        <h3>Current Detection Capabilities:</h3>
        <ul>
        <li><b>Chi-Square Test</b> - Statistical analysis for LSB steganography detection</li>
        <li><b>LSB Analysis</b> - Direct analysis of least significant bits</li>
        <li><b>Histogram Analysis</b> - Pixel distribution pattern analysis</li>
        <li><b>Pixel Analysis</b> - Adjacent pixel relationship analysis</li>
        <li><b>Statistical Analysis</b> - Comprehensive statistical measures</li>
        </ul>
        
        <h3>Advanced Features (Coming Soon):</h3>
        <ul>
        <li>Metadata extraction and analysis</li>
        <li>EXIF data forensic examination</li>
        <li>File signature verification</li>
        <li>Compression artifact detection</li>
        <li>Machine learning classification</li>
        <li>Batch processing capabilities</li>
        </ul>
        </div>
        """
        
        features_label = QLabel(features_text)
        features_label.setWordWrap(True)
        features_label.setStyleSheet("color: #CCCCCC; padding: 20px;")
        features_layout.addWidget(features_label)
        
        layout.addWidget(features_group)
        
        # Status and recommendations
        status_group = QGroupBox("📈 Current Status")
        status_layout = QVBoxLayout(status_group)
        
        status_label = QLabel("""
        <div style="font-size: 14px;">
        <p><b>✅ Ready for Use:</b></p>
        <ul>
        <li>Basic steganography detection algorithms</li>
        <li>Image loading and preview</li>
        <li>Results analysis and export</li>
        <li>Configuration management</li>
        </ul>
        
        <p><b>🚀 Usage Tips:</b></p>
        <ul>
        <li>Use the Detection tab to analyze individual images</li>
        <li>Try different detection method combinations</li>
        <li>Adjust sensitivity settings for better results</li>
        <li>Export results for documentation</li>
        </ul>
        </div>
        """)
        status_label.setWordWrap(True)
        status_label.setAlignment(Qt.AlignLeft)
        status_label.setStyleSheet("color: #44FF44; padding: 20px;")
        status_layout.addWidget(status_label)
        
        layout.addWidget(status_group)
        layout.addStretch()
        
        return widget
    
    def create_reports_tab(self):
        """Create enhanced reports tab"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("📋 Forensic Report Generation")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #0078D4; margin-bottom: 20px;")
        layout.addWidget(header)
        
        # Current capabilities
        current_group = QGroupBox("✅ Currently Available")
        current_layout = QVBoxLayout(current_group)
        
        current_text = """
        <div style="font-size: 14px; line-height: 1.6;">
        <h3>Export Options in Detection Tab:</h3>
        <ul>
        <li><b>JSON Format</b> - Machine-readable results with full details</li>
        <li><b>Detailed Analysis</b> - Human-readable comprehensive reports</li>
        <li><b>Configuration Export</b> - Save and share detection settings</li>
        <li><b>Session Archiving</b> - Complete analysis session data</li>
        </ul>
        
        <h3>How to Generate Reports:</h3>
        <ol>
        <li>Load an image in the Detection tab</li>
        <li>Run your desired detection methods</li>
        <li>Click "Save Results" to export findings</li>
        <li>Choose JSON format for technical reports</li>
        </ol>
        </div>
        """
        
        current_label = QLabel(current_text)
        current_label.setWordWrap(True)
        current_label.setStyleSheet("color: #44FF44; padding: 20px;")
        current_layout.addWidget(current_label)
        
        layout.addWidget(current_group)
        
        # Future features
        future_group = QGroupBox("🚀 Professional Features (Roadmap)")
        future_layout = QVBoxLayout(future_group)
        
        future_text = """
        <div style="font-size: 14px; line-height: 1.6;">
        <h3>Advanced Report Generation:</h3>
        <ul>
        <li>PDF report generation with charts and graphs</li>
        <li>Executive summary creation</li>
        <li>Legal-ready documentation templates</li>
        <li>Chain of custody tracking</li>
        <li>Multi-format export (HTML, PDF, DOCX)</li>
        </ul>
        
        <h3>Professional Features:</h3>
        <ul>
        <li>Batch analysis reporting</li>
        <li>Comparative analysis charts</li>
        <li>Timeline and metadata correlation</li>
        <li>Evidence presentation tools</li>
        <li>Custom report templates</li>
        </ul>
        </div>
        """
        
        future_label = QLabel(future_text)
        future_label.setWordWrap(True)
        future_label.setStyleSheet("color: #FFAA00; padding: 20px;")
        future_layout.addWidget(future_label)
        
        layout.addWidget(future_group)
        layout.addStretch()
        
        return widget
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Open file action
        open_action = QAction('&Open Image...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open an image for analysis')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Recent files (placeholder)
        recent_action = QAction('Recent Files', self)
        recent_action.setEnabled(False)
        file_menu.addAction(recent_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        # Quick detection
        quick_detect_action = QAction('&Quick Detection', self)
        quick_detect_action.setShortcut('Ctrl+D')
        quick_detect_action.setStatusTip('Run quick detection on loaded image')
        quick_detect_action.triggered.connect(self.quick_detection)
        tools_menu.addAction(quick_detect_action)
        
        tools_menu.addSeparator()
        
        # Settings
        settings_action = QAction('&Settings...', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Toggle log panel
        toggle_log_action = QAction('Toggle &Log Panel', self)
        toggle_log_action.setShortcut('F9')
        toggle_log_action.setStatusTip('Show/hide the log panel')
        toggle_log_action.triggered.connect(self.toggle_log_panel)
        view_menu.addAction(toggle_log_action)
        
        view_menu.addSeparator()
        
        # Full screen
        fullscreen_action = QAction('&Full Screen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # Documentation
        docs_action = QAction('&Documentation', self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        
        # About
        about_action = QAction('&About...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the application toolbar"""
        
        toolbar = self.addToolBar('Main')
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # Open image action
        open_action = QAction('📁 Open Image', self)
        open_action.setStatusTip('Open an image for analysis')
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Quick analysis action
        quick_action = QAction('⚡ Quick Analysis', self)
        quick_action.setStatusTip('Start quick analysis of loaded image')
        quick_action.triggered.connect(self.quick_detection)
        toolbar.addAction(quick_action)
        
        toolbar.addSeparator()
        
        # Clear action
        clear_action = QAction('🗑️ Clear', self)
        clear_action.setStatusTip('Clear current analysis')
        clear_action.triggered.connect(self.clear_analysis)
        toolbar.addAction(clear_action)
    
    def create_status_bar(self):
        """Create the status bar"""
        
        self.status_bar = self.statusBar()
        
        # Status label
        self.status_label = QLabel("Ready - Load an image to begin analysis")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Version info
        version_label = QLabel("v1.0")
        version_label.setStyleSheet("color: #888888;")
        self.status_bar.addPermanentWidget(version_label)
    
    def create_bottom_panel(self):
        """Create the bottom panel with logs"""
        
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Log display
        log_group = QGroupBox("Application Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier New", 9))
        
        # Add initial log messages
        welcome_message = """StegAnalysis Suite v1.0 - Professional Steganography Detection
==================================================================================
Application started successfully
Ready for steganography detection and analysis

Instructions:
1. Use File > Open Image or the toolbar button to load an image
2. Go to the Detection tab to run analysis
3. View results and export findings

For help, press F1 or visit the Help menu."""
        
        self.log_display.setPlainText(welcome_message)
        
        log_layout.addWidget(self.log_display)
        layout.addWidget(log_group)
        
        return panel
    
    def setup_theme(self):
        """Apply the dark theme"""
        
        app_style = """
            QMainWindow {
                background-color: #2B2B2B;
                color: #FFFFFF;
            }
            
            QWidget {
                color: #FFFFFF;
                background-color: #2B2B2B;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background-color: #2B2B2B;
                border-radius: 5px;
            }
            
            QTabBar::tab {
                background: #4A4A4A;
                border: 1px solid #555555;
                padding: 12px 20px;
                margin-right: 2px;
                color: #FFFFFF;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
            }
            
            QTabBar::tab:selected {
                background: #0078D4;
            }
            
            QTabBar::tab:hover {
                background: #5A5A5A;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                color: #FFFFFF;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #0078D4;
                font-weight: bold;
            }
            
            QTextEdit {
                background-color: #1E1E1E;
                color: #00FF00;
                border: 1px solid #333333;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                padding: 5px;
            }
            
            QLabel {
                color: #FFFFFF;
            }
            
            QMenuBar {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border-bottom: 1px solid #555555;
            }
            
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
            }
            
            QMenuBar::item:selected {
                background-color: #0078D4;
            }
            
            QMenu {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border: 1px solid #555555;
            }
            
            QMenu::item:selected {
                background-color: #0078D4;
            }
            
            QStatusBar {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border-top: 1px solid #555555;
            }
            
            QToolBar {
                background-color: #2B2B2B;
                border: 1px solid #555555;
                spacing: 3px;
                padding: 3px;
            }
            
            QToolBar::separator {
                background-color: #555555;
                width: 1px;
                margin: 2px;
            }
            
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
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
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #2B2B2B;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 3px;
            }
        """
        
        self.setStyleSheet(app_style)
    
    def restore_settings(self):
        """Restore application settings"""
        
        # Window geometry
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        # Window state
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
    
    def save_settings(self):
        """Save application settings"""
        
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
    
    # Event handlers
    def closeEvent(self, event):
        """Handle application close event"""
        
        self.save_settings()
        self.add_log_message("Application closing...")
        event.accept()
    
    def on_tab_changed(self, index):
        """Handle tab change"""
        
        tab_names = ["Detection", "Analysis", "Reports"]
        if 0 <= index < len(tab_names):
            self.status_label.setText(f"{tab_names[index]} tab active")
            self.add_log_message(f"Switched to {tab_names[index]} tab")
    
    def add_log_message(self, message):
        """Add message to log display"""
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.log_display.append(formatted_message)
        # Auto-scroll to bottom
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )
    
    # Action handlers
    def open_file(self):
        """Open file dialog and load into detection tab"""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image for Steganography Analysis", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All Files (*)"
        )
        
        if file_path:
            self.add_log_message(f"Loading image: {os.path.basename(file_path)}")
            
            # Load file into detection tab if available
            if DETECTION_TAB_AVAILABLE and hasattr(self.detection_tab, 'load_image'):
                self.detection_tab.load_image(file_path)
                # Switch to detection tab
                self.tab_widget.setCurrentIndex(0)
                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            else:
                QMessageBox.information(self, "Image Loaded", 
                                      f"Image loaded: {os.path.basename(file_path)}\n"
                                      "Detection features require the detection_tab module.")
                self.status_label.setText("Image loaded (detection module unavailable)")
    
    def quick_detection(self):
        """Start quick detection if image is loaded"""
        
        if DETECTION_TAB_AVAILABLE and hasattr(self.detection_tab, 'run_quick_detection'):
            self.detection_tab.run_quick_detection()
            self.tab_widget.setCurrentIndex(0)  # Switch to detection tab
        else:
            QMessageBox.information(self, "Feature Unavailable", 
                                  "Quick detection requires the detection_tab module.\n"
                                  "Please ensure detection_tab.py is available.")
    
    def clear_analysis(self):
        """Clear current analysis"""
        
        if DETECTION_TAB_AVAILABLE and hasattr(self.detection_tab, 'clear_all'):
            self.detection_tab.clear_all()
        
        self.status_label.setText("Analysis cleared - Ready for new image")
        self.add_log_message("Analysis cleared")
    
    def toggle_log_panel(self):
        """Toggle the log panel visibility"""
        
        if self.log_panel.isVisible():
            self.log_panel.hide()
            self.add_log_message("Log panel hidden")
        else:
            self.log_panel.show()
            self.add_log_message("Log panel shown")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        
        if self.isFullScreen():
            self.showNormal()
            self.add_log_message("Exited fullscreen mode")
        else:
            self.showFullScreen()
            self.add_log_message("Entered fullscreen mode")
    
    def show_settings(self):
        """Show settings dialog"""
        
        QMessageBox.information(self, "Settings", 
                              "Settings dialog will be available in a future update.\n\n"
                              "Current settings can be configured in the Detection tab.")
    
    def show_documentation(self):
        """Show documentation"""
        
        doc_text = """
        <h2>StegAnalysis Suite Documentation</h2>
        
        <h3>Quick Start Guide:</h3>
        <ol>
        <li><b>Load Image:</b> Use File > Open Image or drag & drop</li>
        <li><b>Select Methods:</b> Choose detection algorithms in the Detection tab</li>
        <li><b>Run Analysis:</b> Click "Start Detection" or use Quick Detection</li>
        <li><b>Review Results:</b> Check the results table and summary</li>
        <li><b>Export Findings:</b> Save results using the Save Results button</li>
        </ol>
        
        <h3>Detection Methods:</h3>
        <ul>
        <li><b>Chi-Square Test:</b> Statistical analysis for LSB steganography</li>
        <li><b>LSB Analysis:</b> Direct examination of least significant bits</li>
        <li><b>Histogram Analysis:</b> Pixel distribution pattern analysis</li>
        <li><b>Pixel Analysis:</b> Adjacent pixel relationship analysis</li>
        <li><b>Statistical Analysis:</b> Comprehensive statistical measures</li>
        </ul>
        
        <h3>Keyboard Shortcuts:</h3>
        <ul>
        <li><b>Ctrl+O:</b> Open image</li>
        <li><b>Ctrl+D:</b> Quick detection</li>
        <li><b>F9:</b> Toggle log panel</li>
        <li><b>F11:</b> Toggle fullscreen</li>
        <li><b>Ctrl+Q:</b> Exit application</li>
        </ul>
        """
        
        QMessageBox.about(self, "Documentation", doc_text)
    
    def show_about(self):
        """Show about dialog"""
        
        about_text = """
        <h2>StegAnalysis Suite v1.0</h2>
        <p><b>Professional Steganography Detection Tool</b></p>
        
        <p>Advanced statistical and machine learning algorithms for detecting hidden data in digital images.</p>
        
        <p><b>Key Features:</b></p>
        <ul>
        <li>Multiple detection algorithms (Chi-square, LSB, Histogram, etc.)</li>
        <li>Real-time analysis with progress tracking</li>
        <li>Professional results presentation</li>
        <li>Comprehensive export capabilities</li>
        <li>Intuitive drag-and-drop interface</li>
        <li>Configurable detection parameters</li>
        </ul>
        
        <p><b>Technical Specifications:</b></p>
        <ul>
        <li>Supports PNG, JPEG, BMP, TIFF, GIF formats</li>
        <li>Statistical and ML-based detection methods</li>
        <li>Batch processing capabilities (coming soon)</li>
        <li>Professional forensic reporting</li>
        </ul>
        
        <p><b>Developed by:</b> StegAnalysis Team<br>
        <b>License:</b> MIT License<br>
        <b>Version:</b> 1.0.0</p>
        
        <p><i>For support and updates, visit our documentation.</i></p>
        """
        
        QMessageBox.about(self, "About StegAnalysis Suite", about_text)


def main():
    """Main application entry point"""
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("StegAnalysis Suite")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("StegAnalysis")
    app.setOrganizationDomain("steganalysis.org")
    
    # Set application properties
    app.setQuitOnLastWindowClosed(True)
    
    # Create and show main window
    try:
        window = StegAnalysisMainWindow()
        window.show()
        
        # Add startup messages
        window.add_log_message("StegAnalysis Suite v1.0 initialized successfully")
        if DETECTION_TAB_AVAILABLE:
            window.add_log_message("Detection module loaded - Full functionality available")
        else:
            window.add_log_message("Warning: Detection module not found - Limited functionality")
        
        window.add_log_message("Ready for steganography detection and analysis")
        
        # Start event loop
        return app.exec_()
        
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", 
                           f"Failed to initialize application:\n{str(e)}")
        return 1


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('steganalysis.log'),
            logging.StreamHandler()
        ]
    )
    
    # Start application
    sys.exit(main())