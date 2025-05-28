#!/usr/bin/env python3
"""
StegAnalysis Suite - GUI Module
Graphical User Interface components for the steganography analysis suite
"""

import sys
import logging
from pathlib import Path

# Check for GUI framework availability
PYQT_AVAILABLE = False
GUI_FRAMEWORK = None

try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    GUI_FRAMEWORK = "PyQt5"
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PySide2.QtWidgets import QApplication
        from PySide2.QtCore import Qt
        GUI_FRAMEWORK = "PySide2"
        PYQT_AVAILABLE = True
    except ImportError:
        pass

if not PYQT_AVAILABLE:
    logging.warning("No GUI framework available. Please install PyQt5 or PySide2.")

# Version information
__version__ = "1.0.0"
__author__ = "StegAnalysis Suite Team"
__email__ = "support@steganalysis.com"

# Module exports
__all__ = [
    'MainWindow',
    'DetectionTab',
    'AnalysisTab', 
    'ReportsTab',
    'GUI_FRAMEWORK',
    'PYQT_AVAILABLE',
    'create_application',
    'setup_gui_logging',
    'get_gui_info',
    'check_gui_dependencies',
    'initialize_gui_module',
    'create_main_window',
    'run_gui_application'
]

# Import main components if GUI framework is available
if PYQT_AVAILABLE:
    try:
        from .main_window import MainWindow
        from .detection_tab import DetectionTab
        from .analysis_tab import AnalysisTab
        from .reports_tab import ReportsTab
        from .utils import (
            create_icon, create_card_style, create_button_style, 
            create_table_style, show_error_message, show_success_message,
            scale_pixmap_to_fit, create_status_indicator, get_current_theme_color
        )
    except ImportError as e:
        logging.error(f"Failed to import GUI components: {e}")
        MainWindow = None
        DetectionTab = None
        AnalysisTab = None
        ReportsTab = None


def create_application(app_name: str = "StegAnalysis Suite", 
                      organization: str = "StegAnalysis", 
                      version: str = "1.0.0") -> 'QApplication':
    """
    Create and configure the main Qt application
    
    Args:
        app_name: Application name
        organization: Organization name
        version: Application version
        
    Returns:
        Configured QApplication instance
        
    Raises:
        RuntimeError: If no GUI framework is available
    """
    if not PYQT_AVAILABLE:
        raise RuntimeError("No GUI framework available. Please install PyQt5 or PySide2.")
    
    # Create application instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(app_name)
    app.setOrganizationName(organization)
    app.setApplicationVersion(version)
    
    # Set application icon if available
    try:
        from .utils import create_icon
        app_icon = create_icon("🔬")
        if app_icon:
            app.setWindowIcon(app_icon)
    except:
        pass
    
    # Configure application style
    configure_application_style(app)
    
    return app


def configure_application_style(app: 'QApplication'):
    """
    Configure the application's visual style and theme
    
    Args:
        app: QApplication instance to configure
    """
    try:
        # Set application stylesheet for dark theme
        dark_stylesheet = """
        QMainWindow {
            background-color: #2B2B2B;
            color: #FFFFFF;
        }
        
        QWidget {
            background-color: #2B2B2B;
            color: #FFFFFF;
            selection-background-color: #0078D4;
        }
        
        QMenuBar {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            color: #FFFFFF;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
        }
        
        QMenuBar::item:selected {
            background-color: #0078D4;
        }
        
        QMenu {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            color: #FFFFFF;
        }
        
        QMenu::item {
            padding: 4px 16px;
        }
        
        QMenu::item:selected {
            background-color: #0078D4;
        }
        
        QStatusBar {
            background-color: #3C3C3C;
            border-top: 1px solid #555555;
            color: #CCCCCC;
        }
        
        QToolBar {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            spacing: 2px;
            padding: 2px;
        }
        
        QScrollBar:vertical {
            background: #3C3C3C;
            width: 16px;
            border: 1px solid #555555;
        }
        
        QScrollBar::handle:vertical {
            background: #666666;
            min-height: 20px;
            border-radius: 8px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #777777;
        }
        
        QScrollBar:horizontal {
            background: #3C3C3C;
            height: 16px;
            border: 1px solid #555555;
        }
        
        QScrollBar::handle:horizontal {
            background: #666666;
            min-width: 20px;
            border-radius: 8px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #777777;
        }
        
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
        
        QTabBar::tab:hover {
            background: #5A5A5A;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QPushButton {
            background-color: #4A4A4A;
            border: 1px solid #666666;
            color: #FFFFFF;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: #5A5A5A;
            border-color: #777777;
        }
        
        QPushButton:pressed {
            background-color: #3A3A3A;
        }
        
        QPushButton:disabled {
            background-color: #333333;
            color: #666666;
            border-color: #444444;
        }
        
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #3C3C3C;
            border: 1px solid #666666;
            color: #FFFFFF;
            padding: 4px;
            border-radius: 4px;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #0078D4;
        }
        
        QComboBox {
            background-color: #3C3C3C;
            border: 1px solid #666666;
            color: #FFFFFF;
            padding: 4px;
            border-radius: 4px;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #666666;
        }
        
        QComboBox::down-arrow {
            image: none;
            border: 1px solid #666666;
        }
        
        QCheckBox {
            color: #FFFFFF;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 2px solid #666666;
            background-color: #3C3C3C;
        }
        
        QCheckBox::indicator:checked {
            background-color: #0078D4;
            border-color: #0078D4;
        }
        
        QProgressBar {
            border: 1px solid #666666;
            border-radius: 5px;
            text-align: center;
            background-color: #3C3C3C;
            color: #FFFFFF;
        }
        
        QProgressBar::chunk {
            background-color: #0078D4;
            border-radius: 4px;
        }
        
        QTableWidget {
            background-color: #3C3C3C;
            alternate-background-color: #424242;
            border: 1px solid #666666;
            gridline-color: #555555;
            color: #FFFFFF;
        }
        
        QHeaderView::section {
            background-color: #4A4A4A;
            border: 1px solid #666666;
            padding: 4px;
            color: #FFFFFF;
            font-weight: bold;
        }
        
        QTreeWidget {
            background-color: #3C3C3C;
            alternate-background-color: #424242;
            border: 1px solid #666666;
            color: #FFFFFF;
        }
        
        QTreeWidget::item {
            padding: 4px;
            border-bottom: 1px solid #555555;
        }
        
        QTreeWidget::item:selected {
            background-color: #0078D4;
        }
        
        QListWidget {
            background-color: #3C3C3C;
            border: 1px solid #666666;
            color: #FFFFFF;
        }
        
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #555555;
            border-radius: 3px;
            margin: 2px;
        }
        
        QListWidget::item:selected {
            background-color: #0078D4;
        }
        
        QSplitter::handle {
            background-color: #555555;
        }
        
        QSplitter::handle:horizontal {
            width: 3px;
        }
        
        QSplitter::handle:vertical {
            height: 3px;
        }
        """
        
        app.setStyleSheet(dark_stylesheet)
        
    except Exception as e:
        logging.warning(f"Failed to set application style: {e}")


def setup_gui_logging(log_level: int = logging.INFO, 
                      log_file: str = "gui.log") -> logging.Logger:
    """
    Setup logging for GUI components
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Log file path (default: "gui.log")
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure GUI logger
    gui_logger = logging.getLogger("gui")
    gui_logger.setLevel(log_level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not gui_logger.handlers:  # Avoid duplicate handlers
        gui_logger.addHandler(file_handler)
        gui_logger.addHandler(console_handler)
    
    return gui_logger


def get_gui_info() -> dict:
    """
    Get information about the GUI framework and capabilities
    
    Returns:
        Dictionary with GUI information
    """
    info = {
        'framework': GUI_FRAMEWORK,
        'available': PYQT_AVAILABLE,
        'version': __version__,
        'components': []
    }
    
    if PYQT_AVAILABLE:
        # Check which components are available
        try:
            from .main_window import MainWindow
            info['components'].append('MainWindow')
        except ImportError:
            pass
        
        try:
            from .detection_tab import DetectionTab
            info['components'].append('DetectionTab')
        except ImportError:
            pass
        
        try:
            from .analysis_tab import AnalysisTab
            info['components'].append('AnalysisTab')
        except ImportError:
            pass
        
        try:
            from .reports_tab import ReportsTab
            info['components'].append('ReportsTab')
        except ImportError:
            pass
        
        # Get framework version
        try:
            if GUI_FRAMEWORK == "PyQt5":
                from PyQt5.QtCore import QT_VERSION_STR
                info['framework_version'] = QT_VERSION_STR
            elif GUI_FRAMEWORK == "PySide2":
                from PySide2.QtCore import __version__ as pyside_version
                info['framework_version'] = pyside_version
        except ImportError:
            info['framework_version'] = "Unknown"
    
    return info


def check_gui_dependencies() -> dict:
    """
    Check for required GUI dependencies
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'qt_framework': PYQT_AVAILABLE,
        'framework_name': GUI_FRAMEWORK,
        'pillow': False,
        'matplotlib': False,
        'numpy': False,
        'opencv': False,
        'scipy': False
    }
    
    # Check for Pillow (PIL)
    try:
        import PIL
        dependencies['pillow'] = True
        dependencies['pillow_version'] = PIL.__version__
    except ImportError:
        pass
    
    # Check for matplotlib
    try:
        import matplotlib
        dependencies['matplotlib'] = True
        dependencies['matplotlib_version'] = matplotlib.__version__
    except ImportError:
        pass
    
    # Check for numpy
    try:
        import numpy
        dependencies['numpy'] = True
        dependencies['numpy_version'] = numpy.__version__
    except ImportError:
        pass
    
    # Check for OpenCV
    try:
        import cv2
        dependencies['opencv'] = True
        dependencies['opencv_version'] = cv2.__version__
    except ImportError:
        pass
    
    # Check for scipy
    try:
        import scipy
        dependencies['scipy'] = True
        dependencies['scipy_version'] = scipy.__version__
    except ImportError:
        pass
    
    return dependencies


def initialize_gui_module() -> bool:
    """
    Initialize the GUI module and check dependencies
    
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        # Setup logging
        gui_logger = setup_gui_logging()
        gui_logger.info(f"Initializing GUI module with {GUI_FRAMEWORK}")
        
        # Check dependencies
        deps = check_gui_dependencies()
        
        missing_deps = []
        if not deps['qt_framework']:
            missing_deps.append('PyQt5 or PySide2')
        if not deps['pillow']:
            missing_deps.append('Pillow')
        if not deps['numpy']:
            missing_deps.append('NumPy')
        
        if missing_deps:
            gui_logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            # Don't return False for missing optional dependencies
            if 'PyQt5 or PySide2' in missing_deps:
                return False
        
        gui_logger.info("GUI module initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize GUI module: {e}")
        return False


def create_main_window(*args, **kwargs):
    """
    Factory function to create the main window
    
    Returns:
        MainWindow instance or None if not available
    """
    if not PYQT_AVAILABLE or MainWindow is None:
        logging.error("MainWindow not available - GUI framework not installed")
        return None
    
    try:
        return MainWindow(*args, **kwargs)
    except Exception as e:
        logging.error(f"Failed to create main window: {e}")
        return None


def run_gui_application(main_window=None, app_args=None):
    """
    Run the GUI application
    
    Args:
        main_window: MainWindow instance (optional)
        app_args: Command line arguments for QApplication
        
    Returns:
        Application exit code
    """
    if not PYQT_AVAILABLE:
        print("Error: No GUI framework available. Please install PyQt5 or PySide2.")
        return 1
    
    try:
        # Create application
        if app_args is None:
            app_args = sys.argv
        
        app = create_application()
        
        # Create main window if not provided
        if main_window is None:
            main_window = create_main_window()
        
        if main_window is None:
            print("Error: Failed to create main window")
            return 1
        
        # Show main window
        main_window.show()
        
        # Handle high DPI displays
        try:
            if hasattr(app, 'setAttribute'):
                if GUI_FRAMEWORK == "PyQt5":
                    from PyQt5.QtCore import Qt
                    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
                    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
                elif GUI_FRAMEWORK == "PySide2":
                    from PySide2.QtCore import Qt
                    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
                    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        except:
            pass
        
        # Setup exception handling
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            print(f"Critical error: {exc_value}")
        
        sys.excepthook = handle_exception
        
        # Run application
        return app.exec_()
        
    except Exception as e:
        print(f"Error running GUI application: {e}")
        logging.error(f"Failed to run GUI application: {e}")
        return 1


def get_theme_colors() -> dict:
    """
    Get the current theme color palette
    
    Returns:
        Dictionary with theme colors
    """
    return {
        'primary': '#0078D4',
        'secondary': '#4A4A4A',
        'success': '#28A745',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'info': '#17A2B8',
        'light': '#F8F9FA',
        'dark': '#343A40',
        'background': '#2B2B2B',
        'surface': '#3C3C3C',
        'border': '#666666',
        'text': '#FFFFFF',
        'text_secondary': '#CCCCCC',
        'text_muted': '#999999'
    }


def apply_custom_theme(app: 'QApplication', theme_colors: dict = None):
    """
    Apply a custom theme to the application
    
    Args:
        app: QApplication instance
        theme_colors: Dictionary with custom colors (optional)
    """
    if theme_colors is None:
        theme_colors = get_theme_colors()
    
    try:
        # Generate custom stylesheet with provided colors
        custom_stylesheet = f"""
        QMainWindow {{
            background-color: {theme_colors['background']};
            color: {theme_colors['text']};
        }}
        
        QWidget {{
            background-color: {theme_colors['background']};
            color: {theme_colors['text']};
            selection-background-color: {theme_colors['primary']};
        }}
        
        QPushButton {{
            background-color: {theme_colors['secondary']};
            border: 1px solid {theme_colors['border']};
            color: {theme_colors['text']};
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        QPushButton:hover {{
            background-color: {theme_colors['primary']};
        }}
        
        QPushButton:pressed {{
            background-color: {theme_colors['dark']};
        }}
        
        QProgressBar::chunk {{
            background-color: {theme_colors['success']};
        }}
        """
        
        app.setStyleSheet(custom_stylesheet)
        logging.info("Custom theme applied successfully")
        
    except Exception as e:
        logging.warning(f"Failed to apply custom theme: {e}")


def create_splash_screen(app: 'QApplication', image_path: str = None) -> 'QSplashScreen':
    """
    Create a splash screen for the application
    
    Args:
        app: QApplication instance
        image_path: Path to splash screen image (optional)
        
    Returns:
        QSplashScreen instance or None
    """
    if not PYQT_AVAILABLE:
        return None
    
    try:
        if GUI_FRAMEWORK == "PyQt5":
            from PyQt5.QtWidgets import QSplashScreen
            from PyQt5.QtGui import QPixmap, QPainter, QFont
            from PyQt5.QtCore import Qt
        else:
            from PySide2.QtWidgets import QSplashScreen
            from PySide2.QtGui import QPixmap, QPainter, QFont
            from PySide2.QtCore import Qt
        
        # Create splash screen pixmap
        if image_path and Path(image_path).exists():
            pixmap = QPixmap(image_path)
        else:
            # Create default splash screen
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.darkBlue)
            
            painter = QPainter(pixmap)
            painter.setPen(Qt.white)
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "StegAnalysis Suite")
            painter.end()
        
        splash = QSplashScreen(pixmap)
        splash.setMask(pixmap.mask())
        
        return splash
        
    except Exception as e:
        logging.warning(f"Failed to create splash screen: {e}")
        return None


def show_splash_message(splash: 'QSplashScreen', message: str, color=None):
    """
    Show a message on the splash screen
    
    Args:
        splash: QSplashScreen instance
        message: Message to display
        color: Text color (optional)
    """
    if splash is None or not PYQT_AVAILABLE:
        return
    
    try:
        if GUI_FRAMEWORK == "PyQt5":
            from PyQt5.QtCore import Qt
        else:
            from PySide2.QtCore import Qt
        
        if color is None:
            color = Qt.white
        
        splash.showMessage(message, Qt.AlignBottom | Qt.AlignCenter, color)
        QApplication.processEvents()
        
    except Exception as e:
        logging.warning(f"Failed to show splash message: {e}")


def save_window_state(window, settings_file: str = "window_state.ini"):
    """
    Save window state to settings file
    
    Args:
        window: QMainWindow instance
        settings_file: Settings file path
    """
    if not PYQT_AVAILABLE:
        return
    
    try:
        if GUI_FRAMEWORK == "PyQt5":
            from PyQt5.QtCore import QSettings
        else:
            from PySide2.QtCore import QSettings
        
        settings = QSettings(settings_file, QSettings.IniFormat)
        settings.setValue("geometry", window.saveGeometry())
        settings.setValue("windowState", window.saveState())
        
        logging.info(f"Window state saved to {settings_file}")
        
    except Exception as e:
        logging.warning(f"Failed to save window state: {e}")


def restore_window_state(window, settings_file: str = "window_state.ini"):
    """
    Restore window state from settings file
    
    Args:
        window: QMainWindow instance
        settings_file: Settings file path
    """
    if not PYQT_AVAILABLE or not Path(settings_file).exists():
        return
    
    try:
        if GUI_FRAMEWORK == "PyQt5":
            from PyQt5.QtCore import QSettings
        else:
            from PySide2.QtCore import QSettings
        
        settings = QSettings(settings_file, QSettings.IniFormat)
        geometry = settings.value("geometry")
        window_state = settings.value("windowState")
        
        if geometry:
            window.restoreGeometry(geometry)
        if window_state:
            window.restoreState(window_state)
        
        logging.info(f"Window state restored from {settings_file}")
        
    except Exception as e:
        logging.warning(f"Failed to restore window state: {e}")


# Initialize module on import
_module_initialized = initialize_gui_module()

# Print initialization status
if __name__ != "__main__":
    if _module_initialized:
        logging.info(f"GUI module loaded successfully with {GUI_FRAMEWORK}")
    else:
        logging.warning("GUI module loaded with missing dependencies")


# Main execution for testing
if __name__ == "__main__":
    print("StegAnalysis Suite - GUI Module")
    print("=" * 50)
    
    # Print GUI information
    gui_info = get_gui_info()
    print(f"Framework: {gui_info['framework'] or 'Not Available'}")
    print(f"Available: {gui_info['available']}")
    print(f"Module Version: {gui_info['version']}")
    
    if gui_info['framework']:
        print(f"Framework Version: {gui_info.get('framework_version', 'Unknown')}")
        print(f"Available Components: {', '.join(gui_info['components']) if gui_info['components'] else 'None'}")
    
    print(f"\nAuthor: {__author__}")
    print(f"Contact: {__email__}")
    
    print("\nDependency Check:")
    print("-" * 30)
    deps = check_gui_dependencies()
    for dep, status in deps.items():
        if dep.endswith('_version'):
            continue
        status_str = "✓" if status else "✗"
        version = deps.get(f"{dep}_version", "")
        version_str = f" ({version})" if version else ""
        dep_name = dep.replace('_', ' ').title()
        print(f"{status_str} {dep_name:<15} {version_str}")
    
    print(f"\nModule Initialized: {'✓' if _module_initialized else '✗'}")
    
    # Test theme colors
    print("\nTheme Colors:")
    print("-" * 20)
    theme = get_theme_colors()
    for color_name, color_value in theme.items():
        print(f"{color_name:<15}: {color_value}")
    
    # Test run if GUI is available
    if gui_info['available'] and MainWindow is not None:
        print("\nGUI Components Available - Ready to launch")
        print("Run with: python -m gui to start the application")
        
        # Optional: Start test GUI
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            print("\nStarting test GUI...")
            try:
                exit_code = run_gui_application()
                print(f"GUI application exited with code: {exit_code}")
            except KeyboardInterrupt:
                print("\nGUI application interrupted by user")
            except Exception as e:
                print(f"Error running test GUI: {e}")
    else:
        print("\nGUI not available - install required dependencies:")
        missing = []
        if not deps['qt_framework']:
            missing.append("PyQt5 or PySide2")
        if not deps['pillow']:
            missing.append("Pillow")
        if not deps['numpy']:
            missing.append("NumPy")
        
        if missing:
            print("Missing dependencies:")
            for dep in missing:
                print(f"  - {dep}")
            print("\nInstall with: pip install PyQt5 Pillow numpy matplotlib opencv-python scipy")
        
    print("\n" + "=" * 50)