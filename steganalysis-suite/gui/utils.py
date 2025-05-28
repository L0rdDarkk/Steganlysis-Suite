#!/usr/bin/env python3
"""
StegAnalysis Suite - GUI Utilities
Utilities for styling, icons, and GUI components
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import base64

try:
    from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QProgressDialog
    from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
    from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QLinearGradient, QPalette, QFont
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PySide2.QtWidgets import QApplication, QWidget, QMessageBox, QProgressDialog
        from PySide2.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
        from PySide2.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QLinearGradient, QPalette, QFont
        PYQT_AVAILABLE = True
    except ImportError:
        PYQT_AVAILABLE = False


# Color scheme for dark theme
DARK_THEME_COLORS = {
    'primary': '#0078D4',
    'primary_hover': '#106EBE',
    'primary_pressed': '#005FA3',
    'background': '#2B2B2B',
    'surface': '#3A3A3A',
    'surface_hover': '#4A4A4A',
    'text': '#FFFFFF',
    'text_secondary': '#CCCCCC',
    'border': '#555555',
    'success': '#10B981',
    'warning': '#F59E0B',
    'error': '#EF4444',
    'info': '#3B82F6'
}

# Icon definitions using base64 encoded SVGs
ICON_DATA = {
    'app_icon': """
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="32" height="32" rx="6" fill="#0078D4"/>
        <path d="M8 12h16v8H8z" fill="white" opacity="0.8"/>
        <circle cx="16" cy="16" r="3" fill="#0078D4"/>
        <path d="M10 14h12v4H10z" fill="white" opacity="0.4"/>
    </svg>
    """,
    'detection': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="11" cy="11" r="8" stroke="#0078D4" stroke-width="2"/>
        <path d="m21 21-4.35-4.35" stroke="#0078D4" stroke-width="2"/>
        <circle cx="11" cy="11" r="3" fill="#0078D4" opacity="0.5"/>
    </svg>
    """,
    'analysis': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 3v18h18" stroke="#0078D4" stroke-width="2"/>
        <path d="m19 9-5 5-4-4-5 5" stroke="#0078D4" stroke-width="2"/>
        <circle cx="19" cy="9" r="2" fill="#0078D4"/>
        <circle cx="14" cy="14" r="2" fill="#0078D4"/>
        <circle cx="10" cy="10" r="2" fill="#0078D4"/>
        <circle cx="5" cy="15" r="2" fill="#0078D4"/>
    </svg>
    """,
    'reports': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="#0078D4" stroke-width="2"/>
        <polyline points="14,2 14,8 20,8" stroke="#0078D4" stroke-width="2"/>
        <line x1="9" y1="15" x2="15" y2="15" stroke="#0078D4" stroke-width="2"/>
        <line x1="9" y1="18" x2="12" y2="18" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'open': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="#0078D4" stroke-width="2"/>
        <polyline points="14,2 14,8 20,8" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'folder': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'settings': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="3" stroke="#0078D4" stroke-width="2"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'models': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#0078D4" stroke-width="2"/>
        <path d="M2 17l10 5 10-5" stroke="#0078D4" stroke-width="2"/>
        <path d="M2 12l10 5 10-5" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'benchmark': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <line x1="12" y1="20" x2="12" y2="10" stroke="#0078D4" stroke-width="2"/>
        <line x1="18" y1="20" x2="18" y2="4" stroke="#0078D4" stroke-width="2"/>
        <line x1="6" y1="20" x2="6" y2="16" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'help': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" stroke="#0078D4" stroke-width="2"/>
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" stroke="#0078D4" stroke-width="2"/>
        <circle cx="12" cy="17" r="1" fill="#0078D4"/>
    </svg>
    """,
    'about': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" stroke="#0078D4" stroke-width="2"/>
        <line x1="12" y1="16" x2="12" y2="12" stroke="#0078D4" stroke-width="2"/>
        <circle cx="12" cy="8" r="1" fill="#0078D4"/>
    </svg>
    """,
    'recent': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" stroke="#0078D4" stroke-width="2"/>
        <polyline points="12,6 12,12 16,14" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """,
    'exit': """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" stroke="#0078D4" stroke-width="2"/>
        <polyline points="16,17 21,12 16,7" stroke="#0078D4" stroke-width="2"/>
        <line x1="21" y1="12" x2="9" y2="12" stroke="#0078D4" stroke-width="2"/>
    </svg>
    """
}


def create_icon(icon_name: str, size: Tuple[int, int] = (24, 24)) -> QIcon:
    """Create an icon from SVG data"""
    
    if not PYQT_AVAILABLE:
        return QIcon()
    
    if icon_name not in ICON_DATA:
        # Return a default icon if not found
        return QIcon()
    
    # Create pixmap from SVG
    svg_data = ICON_DATA[icon_name].encode('utf-8')
    pixmap = QPixmap()
    pixmap.loadFromData(svg_data)
    
    # Scale to desired size
    if pixmap.size() != size:
        pixmap = pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    return QIcon(pixmap)


def apply_dark_theme(widget_or_app):
    """Apply dark theme to widget or application"""
    
    if not PYQT_AVAILABLE:
        return
    
    # Define the dark theme stylesheet
    dark_stylesheet = f"""
    /* Base Widget Styling */
    QWidget {{
        background-color: {DARK_THEME_COLORS['background']};
        color: {DARK_THEME_COLORS['text']};
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12px;
        border: none;
    }}
    
    /* Main Window */
    QMainWindow {{
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                  stop: 0 #1E1E1E, stop: 1 {DARK_THEME_COLORS['background']});
    }}
    
    /* Buttons */
    QPushButton {{
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                  stop: 0 {DARK_THEME_COLORS['surface_hover']}, 
                                  stop: 1 {DARK_THEME_COLORS['surface']});
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        min-width: 80px;
        min-height: 30px;
    }}
    
    QPushButton:hover {{
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                  stop: 0 {DARK_THEME_COLORS['primary']}, 
                                  stop: 1 {DARK_THEME_COLORS['primary_hover']});
        border-color: {DARK_THEME_COLORS['primary']};
    }}
    
    QPushButton:pressed {{
        background: {DARK_THEME_COLORS['primary_pressed']};
    }}
    
    QPushButton:disabled {{
        background: #1A1A1A;
        color: #666666;
        border-color: #333333;
    }}
    
    /* Input Fields */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {DARK_THEME_COLORS['surface']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 4px;
        padding: 8px;
        color: {DARK_THEME_COLORS['text']};
    }}
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {DARK_THEME_COLORS['primary']};
    }}
    
    /* ComboBox */
    QComboBox {{
        background-color: {DARK_THEME_COLORS['surface']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 4px;
        padding: 8px;
        min-width: 100px;
    }}
    
    QComboBox:hover {{
        border-color: {DARK_THEME_COLORS['primary']};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid {DARK_THEME_COLORS['text']};
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {DARK_THEME_COLORS['surface']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        selection-background-color: {DARK_THEME_COLORS['primary']};
    }}
    
    /* Scroll Bars */
    QScrollBar:vertical {{
        background: {DARK_THEME_COLORS['surface']};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {DARK_THEME_COLORS['border']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {DARK_THEME_COLORS['primary']};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
    }}
    
    QScrollBar:horizontal {{
        background: {DARK_THEME_COLORS['surface']};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {DARK_THEME_COLORS['border']};
        border-radius: 6px;
        min-width: 20px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: {DARK_THEME_COLORS['primary']};
    }}
    
    /* Lists and Trees */
    QListWidget, QTreeWidget {{
        background-color: {DARK_THEME_COLORS['surface']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 4px;
        alternate-background-color: {DARK_THEME_COLORS['background']};
    }}
    
    QListWidget::item, QTreeWidget::item {{
        padding: 8px;
        border-bottom: 1px solid {DARK_THEME_COLORS['border']};
    }}
    
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {DARK_THEME_COLORS['primary']};
    }}
    
    QListWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {DARK_THEME_COLORS['surface_hover']};
    }}
    
    /* Tables */
    QTableWidget {{
        background-color: {DARK_THEME_COLORS['surface']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        gridline-color: {DARK_THEME_COLORS['border']};
        alternate-background-color: {DARK_THEME_COLORS['background']};
    }}
    
    QTableWidget::item {{
        padding: 8px;
    }}
    
    QTableWidget::item:selected {{
        background-color: {DARK_THEME_COLORS['primary']};
    }}
    
    QHeaderView::section {{
        background-color: {DARK_THEME_COLORS['surface_hover']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        padding: 8px;
        font-weight: bold;
    }}
    
    /* Splitter */
    QSplitter::handle {{
        background-color: {DARK_THEME_COLORS['border']};
    }}
    
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    
    /* Progress Bar */
    QProgressBar {{
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 4px;
        background-color: {DARK_THEME_COLORS['surface']};
        text-align: center;
        color: {DARK_THEME_COLORS['text']};
        font-weight: bold;
    }}
    
    QProgressBar::chunk {{
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                  stop: 0 {DARK_THEME_COLORS['primary']}, 
                                  stop: 1 #40E0D0);
        border-radius: 3px;
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        border: 1px solid {DARK_THEME_COLORS['border']};
        height: 6px;
        background: {DARK_THEME_COLORS['surface']};
        border-radius: 3px;
    }}
    
    QSlider::handle:horizontal {{
        background: {DARK_THEME_COLORS['primary']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {DARK_THEME_COLORS['primary_hover']};
    }}
    
    /* Check Boxes */
    QCheckBox {{
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 3px;
        background-color: {DARK_THEME_COLORS['surface']};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {DARK_THEME_COLORS['primary']};
        border-color: {DARK_THEME_COLORS['primary']};
    }}
    
    QCheckBox::indicator:hover {{
        border-color: {DARK_THEME_COLORS['primary']};
    }}
    
    /* Radio Buttons */
    QRadioButton {{
        spacing: 8px;
    }}
    
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 8px;
        background-color: {DARK_THEME_COLORS['surface']};
    }}
    
    QRadioButton::indicator:checked {{
        background-color: {DARK_THEME_COLORS['primary']};
        border-color: {DARK_THEME_COLORS['primary']};
    }}
    
    /* Tool Tips */
    QToolTip {{
        background-color: {DARK_THEME_COLORS['surface_hover']};
        color: {DARK_THEME_COLORS['text']};
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: 4px;
        padding: 5px;
    }}
    """
    
    widget_or_app.setStyleSheet(dark_stylesheet)


def create_gradient_background(colors: list, direction: str = 'vertical') -> str:
    """Create CSS gradient background"""
    
    if direction == 'horizontal':
        gradient_direction = "x1: 0, y1: 0, x2: 1, y2: 0"
    else:
        gradient_direction = "x1: 0, y1: 0, x2: 0, y2: 1"
    
    gradient = f"qlineargradient({gradient_direction},"
    
    for i, color in enumerate(colors):
        stop = i / (len(colors) - 1)
        gradient += f" stop: {stop} {color},"
    
    gradient = gradient.rstrip(',') + ")"
    
    return gradient


def show_loading_dialog(parent, title: str = "Processing", message: str = "Please wait..."):
    """Show a loading dialog"""
    
    if not PYQT_AVAILABLE:
        return None
    
    dialog = QProgressDialog(message, "Cancel", 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    
    # Apply dark theme
    dialog.setStyleSheet(f"""
        QProgressDialog {{
            background-color: {DARK_THEME_COLORS['background']};
            color: {DARK_THEME_COLORS['text']};
        }}
    """)
    
    return dialog


def show_error_message(parent, title: str, message: str):
    """Show an error message dialog"""
    
    if not PYQT_AVAILABLE:
        print(f"ERROR: {title} - {message}")
        return
    
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {DARK_THEME_COLORS['background']};
            color: {DARK_THEME_COLORS['text']};
        }}
        QMessageBox QPushButton {{
            min-width: 80px;
            padding: 6px 12px;
        }}
    """)
    msg_box.exec_()


def show_success_message(parent, title: str, message: str):
    """Show a success message dialog"""
    
    if not PYQT_AVAILABLE:
        print(f"SUCCESS: {title} - {message}")
        return
    
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Information)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {DARK_THEME_COLORS['background']};
            color: {DARK_THEME_COLORS['text']};
        }}
        QMessageBox QPushButton {{
            min-width: 80px;
            padding: 6px 12px;
        }}
    """)
    msg_box.exec_()


def show_confirmation_dialog(parent, title: str, message: str) -> bool:
    """Show a confirmation dialog"""
    
    if not PYQT_AVAILABLE:
        return True
    
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Question)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {DARK_THEME_COLORS['background']};
            color: {DARK_THEME_COLORS['text']};
        }}
        QMessageBox QPushButton {{
            min-width: 80px;
            padding: 6px 12px;
        }}
    """)
    
    return msg_box.exec_() == QMessageBox.Yes


class AnimatedWidget(QWidget):
    """Widget with animation capabilities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation = None
    
    def fade_in(self, duration: int = 300):
        """Fade in animation"""
        if not PYQT_AVAILABLE:
            return
        
        self.setWindowOpacity(0)
        self.show()
        
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(duration)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.OutQuart)
        self.animation.start()
    
    def fade_out(self, duration: int = 300):
        """Fade out animation"""
        if not PYQT_AVAILABLE:
            return
        
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(duration)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.setEasingCurve(QEasingCurve.OutQuart)
        self.animation.finished.connect(self.hide)
        self.animation.start()
    
    def slide_in_from_right(self, duration: int = 400):
        """Slide in from right animation"""
        if not PYQT_AVAILABLE:
            return
        
        parent = self.parent()
        if not parent:
            return
        
        start_pos = QRect(parent.width(), self.y(), self.width(), self.height())
        end_pos = QRect(self.x(), self.y(), self.width(), self.height())
        
        self.setGeometry(start_pos)
        self.show()
        
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(duration)
        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.setEasingCurve(QEasingCurve.OutQuart)
        self.animation.start()


def create_status_indicator(status: str, size: int = 12) -> QIcon:
    """Create a colored status indicator icon"""
    
    if not PYQT_AVAILABLE:
        return QIcon()
    
    # Color mapping for different statuses
    status_colors = {
        'success': DARK_THEME_COLORS['success'],
        'error': DARK_THEME_COLORS['error'],
        'warning': DARK_THEME_COLORS['warning'],
        'info': DARK_THEME_COLORS['info'],
        'processing': DARK_THEME_COLORS['primary'],
        'idle': DARK_THEME_COLORS['text_secondary']
    }
    
    color = status_colors.get(status.lower(), DARK_THEME_COLORS['text_secondary'])
    
    # Create a simple circle icon
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QBrush(QColor(color)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    
    return QIcon(pixmap)


def create_gradient_pixmap(width: int, height: int, colors: list, direction: str = 'vertical') -> QPixmap:
    """Create a gradient pixmap"""
    
    if not PYQT_AVAILABLE:
        return QPixmap()
    
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    gradient = QLinearGradient()
    
    if direction == 'horizontal':
        gradient.setStart(0, 0)
        gradient.setFinalStop(width, 0)
    else:
        gradient.setStart(0, 0)
        gradient.setFinalStop(0, height)
    
    for i, color in enumerate(colors):
        stop = i / (len(colors) - 1)
        gradient.setColorAt(stop, QColor(color))
    
    painter.fillRect(pixmap.rect(), QBrush(gradient))
    painter.end()
    
    return pixmap


def setup_font(family: str = "Segoe UI", size: int = 12, weight: int = QFont.Normal) -> QFont:
    """Setup custom font"""
    
    if not PYQT_AVAILABLE:
        return None
    
    font = QFont(family, size, weight)
    font.setStyleHint(QFont.SansSerif)
    font.setHintingPreference(QFont.PreferDefaultHinting)
    
    return font


def get_theme_color(color_name: str) -> str:
    """Get theme color by name"""
    return DARK_THEME_COLORS.get(color_name, DARK_THEME_COLORS['text'])


def create_card_style(background_color: str = None, border_radius: int = 8) -> str:
    """Create card-like styling"""
    
    bg_color = background_color or DARK_THEME_COLORS['surface']
    
    return f"""
        background-color: {bg_color};
        border: 1px solid {DARK_THEME_COLORS['border']};
        border-radius: {border_radius}px;
        padding: 12px;
        margin: 4px;
    """


def create_button_style(button_type: str = 'primary') -> str:
    """Create button styling based on type"""
    
    if button_type == 'primary':
        return f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {DARK_THEME_COLORS['primary']}, 
                                          stop: 1 {DARK_THEME_COLORS['primary_hover']});
                border: 1px solid {DARK_THEME_COLORS['primary']};
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                color: white;
                min-width: 100px;
                min-height: 35px;
            }}
            
            QPushButton:hover {{
                background: {DARK_THEME_COLORS['primary_hover']};
            }}
            
            QPushButton:pressed {{
                background: {DARK_THEME_COLORS['primary_pressed']};
            }}
        """
    
    elif button_type == 'secondary':
        return f"""
            QPushButton {{
                background: transparent;
                border: 2px solid {DARK_THEME_COLORS['border']};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: {DARK_THEME_COLORS['text']};
                min-width: 80px;
                min-height: 35px;
            }}
            
            QPushButton:hover {{
                border-color: {DARK_THEME_COLORS['primary']};
                color: {DARK_THEME_COLORS['primary']};
            }}
            
            QPushButton:pressed {{
                background: {DARK_THEME_COLORS['primary']};
                color: white;
            }}
        """
    
    elif button_type == 'danger':
        return f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {DARK_THEME_COLORS['error']}, 
                                          stop: 1 #DC2626);
                border: 1px solid {DARK_THEME_COLORS['error']};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
                min-width: 80px;
                min-height: 35px;
            }}
            
            QPushButton:hover {{
                background: #DC2626;
            }}
            
            QPushButton:pressed {{
                background: #B91C1C;
            }}
        """
    
    elif button_type == 'success':
        return f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {DARK_THEME_COLORS['success']}, 
                                          stop: 1 #059669);
                border: 1px solid {DARK_THEME_COLORS['success']};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
                min-width: 80px;
                min-height: 35px;
            }}
            
            QPushButton:hover {{
                background: #059669;
            }}
            
            QPushButton:pressed {{
                background: #047857;
            }}
        """
    
    else:
        # Default style
        return f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {DARK_THEME_COLORS['surface_hover']}, 
                                          stop: 1 {DARK_THEME_COLORS['surface']});
                border: 1px solid {DARK_THEME_COLORS['border']};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
                min-height: 35px;
            }}
            
            QPushButton:hover {{
                background: {DARK_THEME_COLORS['surface_hover']};
                border-color: {DARK_THEME_COLORS['primary']};
            }}
            
            QPushButton:pressed {{
                background: {DARK_THEME_COLORS['primary']};
            }}
        """


def create_input_style() -> str:
    """Create input field styling"""
    
    return f"""
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {DARK_THEME_COLORS['surface']};
            border: 2px solid {DARK_THEME_COLORS['border']};
            border-radius: 6px;
            padding: 10px;
            color: {DARK_THEME_COLORS['text']};
            font-size: 13px;
            selection-background-color: {DARK_THEME_COLORS['primary']};
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {DARK_THEME_COLORS['primary']};
            background-color: {DARK_THEME_COLORS['background']};
        }}
        
        QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {{
            border-color: {DARK_THEME_COLORS['primary']};
        }}
    """


def create_table_style() -> str:
    """Create table styling"""
    
    return f"""
        QTableWidget {{
            background-color: {DARK_THEME_COLORS['surface']};
            border: 1px solid {DARK_THEME_COLORS['border']};
            border-radius: 6px;
            gridline-color: {DARK_THEME_COLORS['border']};
            alternate-background-color: {DARK_THEME_COLORS['background']};
            selection-background-color: {DARK_THEME_COLORS['primary']};
        }}
        
        QTableWidget::item {{
            padding: 10px;
            border-bottom: 1px solid {DARK_THEME_COLORS['border']};
        }}
        
        QTableWidget::item:selected {{
            background-color: {DARK_THEME_COLORS['primary']};
            color: white;
        }}
        
        QTableWidget::item:hover {{
            background-color: {DARK_THEME_COLORS['surface_hover']};
        }}
        
        QHeaderView::section {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 {DARK_THEME_COLORS['surface_hover']}, 
                                      stop: 1 {DARK_THEME_COLORS['surface']});
            border: 1px solid {DARK_THEME_COLORS['border']};
            padding: 12px;
            font-weight: bold;
            color: {DARK_THEME_COLORS['text']};
        }}
        
        QHeaderView::section:hover {{
            background-color: {DARK_THEME_COLORS['primary']};
        }}
    """


class ThemeManager:
    """Centralized theme management"""
    
    def __init__(self):
        self.current_theme = 'dark'
        self.themes = {
            'dark': DARK_THEME_COLORS,
            'light': {
                'primary': '#0078D4',
                'primary_hover': '#106EBE',
                'primary_pressed': '#005FA3',
                'background': '#FFFFFF',
                'surface': '#F5F5F5',
                'surface_hover': '#E5E5E5',
                'text': '#000000',
                'text_secondary': '#666666',
                'border': '#CCCCCC',
                'success': '#10B981',
                'warning': '#F59E0B',
                'error': '#EF4444',
                'info': '#3B82F6'
            }
        }
    
    def get_color(self, color_name: str) -> str:
        """Get color from current theme"""
        return self.themes[self.current_theme].get(color_name, '#FFFFFF')
    
    def set_theme(self, theme_name: str):
        """Set current theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
    
    def apply_theme_to_widget(self, widget):
        """Apply current theme to widget"""
        if self.current_theme == 'dark':
            apply_dark_theme(widget)
        # Add light theme application here when needed


# Global theme manager instance
theme_manager = ThemeManager()


def get_current_theme_color(color_name: str) -> str:
    """Get color from current theme"""
    return theme_manager.get_color(color_name)


# Utility functions for common UI operations
def center_widget_on_screen(widget):
    """Center widget on screen"""
    if not PYQT_AVAILABLE:
        return
    
    screen = QApplication.desktop().screenGeometry()
    widget_rect = widget.geometry()
    
    x = (screen.width() - widget_rect.width()) // 2
    y = (screen.height() - widget_rect.height()) // 2
    
    widget.move(x, y)


def scale_pixmap_to_fit(pixmap: QPixmap, max_width: int, max_height: int) -> QPixmap:
    """Scale pixmap to fit within given dimensions while maintaining aspect ratio"""
    if not PYQT_AVAILABLE or pixmap.isNull():
        return QPixmap()
    
    return pixmap.scaled(
        max_width, max_height, 
        Qt.KeepAspectRatio, 
        Qt.SmoothTransformation
    )


def create_separator(orientation: str = 'horizontal') -> QWidget:
    """Create a visual separator"""
    if not PYQT_AVAILABLE:
        return QWidget()
    
    separator = QWidget()
    
    if orientation == 'horizontal':
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {DARK_THEME_COLORS['border']};")
    else:
        separator.setFixedWidth(1)
        separator.setStyleSheet(f"background-color: {DARK_THEME_COLORS['border']};")
    
    return separator


# Export commonly used functions and classes
__all__ = [
    'create_icon',
    'apply_dark_theme',
    'create_gradient_background',
    'show_loading_dialog',
    'show_error_message',
    'show_success_message',
    'show_confirmation_dialog',
    'AnimatedWidget',
    'create_status_indicator',
    'create_gradient_pixmap',
    'setup_font',
    'get_theme_color',
    'create_card_style',
    'create_button_style',
    'create_input_style',
    'create_table_style',
    'ThemeManager',
    'theme_manager',
    'get_current_theme_color',
    'center_widget_on_screen',
    'scale_pixmap_to_fit',
    'create_separator',
    'DARK_THEME_COLORS'
]


if __name__ == "__main__":
    # Test utility functions
    print("Testing GUI utilities...")
    
    if PYQT_AVAILABLE:
        app = QApplication([])
        
        # Test icon creation
        icon = create_icon('detection')
        print(f"Detection icon created: {not icon.isNull()}")
        
        # Test gradient creation
        gradient = create_gradient_background(['#0078D4', '#40E0D0'])
        print(f"Gradient created: {len(gradient) > 0}")
        
        # Test theme colors
        primary_color = get_theme_color('primary')
        print(f"Primary color: {primary_color}")
        
        print("GUI utilities test completed!")
    else:
        print("PyQt5/PySide2 not available - utilities will return defaults")