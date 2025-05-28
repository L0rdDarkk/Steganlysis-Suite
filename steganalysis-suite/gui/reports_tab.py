#!/usr/bin/env python3
"""
StegAnalysis Suite - Reports Tab
Forensic report generation and management interface
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import threading
import time
from datetime import datetime, timedelta
import webbrowser
import tempfile

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
        QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
        QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
        QHeaderView, QFileDialog, QCheckBox, QComboBox, QSpinBox,
        QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QPlainTextEdit,
        QSizePolicy, QListWidget, QListWidgetItem, QDateEdit, QLineEdit,
        QTextBrowser, QDialog, QDialogButtonBox, QFormLayout
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot, QDate
    from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QTextCursor, QTextDocument
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
            QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
            QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
            QHeaderView, QFileDialog, QCheckBox, QComboBox, QSpinBox,
            QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QPlainTextEdit,
            QSizePolicy, QListWidget, QListWidgetItem, QDateEdit, QLineEdit,
            QTextBrowser, QDialog, QDialogButtonBox, QFormLayout
        )
        from PySide2.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer, Slot as pyqtSlot, QDate
        from PySide2.QtGui import QPixmap, QIcon, QFont, QPalette, QTextCursor, QTextDocument
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
from modules.reports import ForensicReportGenerator, ReportTemplate


class ReportGenerationWorker(QThread):
    """Worker thread for generating forensic reports"""
    
    progress_updated = pyqtSignal(int)
    report_generated = pyqtSignal(str)  # file_path
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, report_data: Dict[str, Any], output_path: str, report_type: str, template: str):
        super().__init__()
        self.report_data = report_data
        self.output_path = output_path
        self.report_type = report_type
        self.template = template
        
    def run(self):
        """Generate the forensic report"""
        try:
            self.log_message.emit("Starting report generation...")
            self.progress_updated.emit(10)
            
            # Initialize report generator
            generator = ForensicReportGenerator()
            self.progress_updated.emit(25)
            
            self.log_message.emit(f"Generating {self.report_type} report...")
            
            if self.report_type == "html":
                generator.generate_html_report(self.report_data, self.output_path, self.template)
            elif self.report_type == "pdf":
                generator.generate_pdf_report(self.report_data, self.output_path, self.template)
            elif self.report_type == "json":
                generator.generate_json_report(self.report_data, self.output_path)
            elif self.report_type == "xml":
                generator.generate_xml_report(self.report_data, self.output_path)
            else:
                raise ValueError(f"Unsupported report type: {self.report_type}")
            
            self.progress_updated.emit(90)
            self.log_message.emit("Report generation completed")
            self.progress_updated.emit(100)
            
            self.report_generated.emit(self.output_path)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ReportTemplateEditor(QDialog):
    """Dialog for editing report templates"""
    
    def __init__(self, template_content: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report Template Editor")
        self.setModal(True)
        self.resize(800, 600)
        self.template_content = template_content
        self.init_ui()
    
    def init_ui(self):
        """Initialize template editor UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("📝 Report Template Editor")
        title.setFont(QFont("", 14, QFont.Bold))
        header_layout.addWidget(title)
        
        # Template type selector
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Template Type:"))
        self.template_type = QComboBox()
        self.template_type.addItems(["HTML", "JSON", "XML", "Custom"])
        header_layout.addWidget(self.template_type)
        
        layout.addLayout(header_layout)
        
        # Template editor
        editor_group = QGroupBox("Template Content")
        editor_layout = QVBoxLayout(editor_group)
        
        self.template_editor = QTextEdit()
        self.template_editor.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 10px;
            }}
        """)
        self.template_editor.setPlainText(self.template_content)
        editor_layout.addWidget(self.template_editor)
        
        # Template variables help
        help_text = """
Available Template Variables:
{{title}} - Report title
{{date}} - Analysis date
{{image_path}} - Path to analyzed image
{{analysis_results}} - Dictionary of analysis results
{{metadata}} - Image metadata
{{summary}} - Analysis summary
{{timestamp}} - Report generation timestamp
        """
        
        help_label = QLabel(help_text)
        help_label.setStyleSheet(f"""
            QLabel {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }}
        """)
        editor_layout.addWidget(help_label)
        
        layout.addWidget(editor_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("👁️ Preview")
        self.preview_btn.setStyleSheet(create_button_style('secondary'))
        self.preview_btn.clicked.connect(self.preview_template)
        button_layout.addWidget(self.preview_btn)
        
        self.save_btn = QPushButton("💾 Save Template")
        self.save_btn.setStyleSheet(create_button_style('secondary'))
        self.save_btn.clicked.connect(self.save_template)
        button_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("📁 Load Template")
        self.load_btn.setStyleSheet(create_button_style('secondary'))
        self.load_btn.clicked.connect(self.load_template)
        button_layout.addWidget(self.load_btn)
        
        button_layout.addStretch()
        
        # Dialog buttons
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        button_layout.addWidget(dialog_buttons)
        
        layout.addLayout(button_layout)
    
    def preview_template(self):
        """Preview the template with sample data"""
        template_content = self.template_editor.toPlainText()
        
        # Create sample data for preview
        sample_data = {
            'title': 'Sample Forensic Analysis Report',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'image_path': '/path/to/sample/image.jpg',
            'analysis_results': {
                'metadata_extraction': {'status': 'success'},
                'histogram_analysis': {'status': 'success'}
            },
            'metadata': {'format': 'JPEG', 'size': '1920x1080'},
            'summary': 'Sample analysis completed successfully',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Simple template substitution for preview
            preview_content = template_content
            for key, value in sample_data.items():
                placeholder = f"{{{{{key}}}}}"
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                preview_content = preview_content.replace(placeholder, str(value))
            
            # Show preview in new window
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Template Preview")
            preview_dialog.resize(700, 500)
            
            layout = QVBoxLayout(preview_dialog)
            
            preview_text = QTextBrowser()
            if self.template_type.currentText() == "HTML":
                preview_text.setHtml(preview_content)
            else:
                preview_text.setPlainText(preview_content)
            
            layout.addWidget(preview_text)
            
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)
            
            preview_dialog.exec_()
            
        except Exception as e:
            show_error_message(self, "Preview Error", f"Failed to preview template: {str(e)}")
    
    def save_template(self):
        """Save template to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Template",
            f"report_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html);;JSON Files (*.json);;XML Files (*.xml);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.template_editor.toPlainText())
                show_success_message(self, "Template Saved", f"Template saved to {file_path}")
            except Exception as e:
                show_error_message(self, "Save Error", f"Failed to save template: {str(e)}")
    
    def load_template(self):
        """Load template from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Template",
            "",
            "HTML Files (*.html);;JSON Files (*.json);;XML Files (*.xml);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                self.template_editor.setPlainText(content)
            except Exception as e:
                show_error_message(self, "Load Error", f"Failed to load template: {str(e)}")
    
    def get_template_content(self) -> str:
        """Get the current template content"""
        return self.template_editor.toPlainText()


class ReportHistoryWidget(QWidget):
    """Widget for managing report history"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reports_data = []
        self.init_ui()
        self.load_report_history()
    
    def init_ui(self):
        """Initialize report history UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("📚 Report History")
        title.setFont(QFont("", 12, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet(create_button_style('secondary'))
        refresh_btn.clicked.connect(self.load_report_history)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Reports table
        self.reports_table = QTableWidget()
        self.reports_table.setColumnCount(6)
        self.reports_table.setHorizontalHeaderLabels([
            "Date", "Image", "Type", "Template", "Size", "Actions"
        ])
        self.reports_table.setStyleSheet(create_table_style())
        
        # Set column widths
        header = self.reports_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.reports_table)
        
        # Filter controls
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)
        
        filter_layout.addWidget(QLabel("Date Range:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        self.start_date.setCalendarPopup(True)
        filter_layout.addWidget(self.start_date)
        
        filter_layout.addWidget(QLabel("to"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        filter_layout.addWidget(self.end_date)
        
        filter_layout.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "HTML", "PDF", "JSON", "XML"])
        filter_layout.addWidget(self.type_filter)
        
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.setStyleSheet(create_button_style('primary'))
        apply_filter_btn.clicked.connect(self.apply_filters)
        filter_layout.addWidget(apply_filter_btn)
        
        filter_layout.addStretch()
        
        layout.addWidget(filter_group)
    
    def load_report_history(self):
        """Load report history from disk"""
        try:
            reports_dir = Path("reports/generated")
            if not reports_dir.exists():
                self.reports_table.setRowCount(0)
                return
            
            self.reports_data = []
            
            # Scan for report files
            for report_file in reports_dir.iterdir():
                if report_file.is_file() and report_file.suffix in ['.html', '.pdf', '.json', '.xml']:
                    try:
                        stat = report_file.stat()
                        report_info = {
                            'path': str(report_file),
                            'name': report_file.name,
                            'type': report_file.suffix[1:].upper(),
                            'size': stat.st_size,
                            'date': datetime.fromtimestamp(stat.st_mtime),
                            'template': self.extract_template_info(report_file)
                        }
                        self.reports_data.append(report_info)
                    except Exception as e:
                        continue
            
            # Sort by date (newest first)
            self.reports_data.sort(key=lambda x: x['date'], reverse=True)
            
            self.update_table()
            
        except Exception as e:
            show_error_message(self, "Load Error", f"Failed to load report history: {str(e)}")
    
    def extract_template_info(self, report_file: Path) -> str:
        """Extract template information from report file"""
        try:
            if report_file.suffix == '.html':
                with open(report_file, 'r') as f:
                    content = f.read()
                    if 'forensic_template' in content:
                        return 'Forensic'
                    elif 'summary_template' in content:
                        return 'Summary'
                    else:
                        return 'Custom'
            else:
                return 'Standard'
        except:
            return 'Unknown'
    
    def update_table(self):
        """Update the reports table display"""
        self.reports_table.setRowCount(len(self.reports_data))
        
        for row, report in enumerate(self.reports_data):
            # Date
            date_item = QTableWidgetItem(report['date'].strftime('%Y-%m-%d %H:%M'))
            self.reports_table.setItem(row, 0, date_item)
            
            # Image name (extract from filename)
            image_name = self.extract_image_name(report['name'])
            image_item = QTableWidgetItem(image_name)
            self.reports_table.setItem(row, 1, image_item)
            
            # Type
            type_item = QTableWidgetItem(report['type'])
            self.reports_table.setItem(row, 2, type_item)
            
            # Template
            template_item = QTableWidgetItem(report['template'])
            self.reports_table.setItem(row, 3, template_item)
            
            # Size
            size_str = self.format_file_size(report['size'])
            size_item = QTableWidgetItem(size_str)
            self.reports_table.setItem(row, 4, size_item)
            
            # Actions - create action buttons
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            view_btn = QPushButton("👁️")
            view_btn.setToolTip("View Report")
            view_btn.setMaximumSize(30, 25)
            view_btn.clicked.connect(lambda checked, path=report['path']: self.view_report(path))
            actions_layout.addWidget(view_btn)
            
            delete_btn = QPushButton("🗑️")
            delete_btn.setToolTip("Delete Report")
            delete_btn.setMaximumSize(30, 25)
            delete_btn.clicked.connect(lambda checked, path=report['path']: self.delete_report(path))
            actions_layout.addWidget(delete_btn)
            
            self.reports_table.setCellWidget(row, 5, actions_widget)
    
    def extract_image_name(self, filename: str) -> str:
        """Extract image name from report filename"""
        # Expected format: analysis_IMAGENAME_YYYYMMDD_HHMMSS.ext
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == 'analysis':
            return parts[1]
        return "Unknown"
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def apply_filters(self):
        """Apply date and type filters to report list"""
        start_date = self.start_date.date().toPython()
        end_date = self.end_date.date().toPython()
        type_filter = self.type_filter.currentText()
        
        filtered_data = []
        for report in self.reports_data:
            report_date = report['date'].date()
            
            # Check date range
            if not (start_date <= report_date <= end_date):
                continue
            
            # Check type filter
            if type_filter != "All" and report['type'] != type_filter:
                continue
            
            filtered_data.append(report)
        
        # Update table with filtered data
        self.reports_table.setRowCount(len(filtered_data))
        
        for row, report in enumerate(filtered_data):
            # Update table items (same as update_table but with filtered data)
            date_item = QTableWidgetItem(report['date'].strftime('%Y-%m-%d %H:%M'))
            self.reports_table.setItem(row, 0, date_item)
            
            image_name = self.extract_image_name(report['name'])
            image_item = QTableWidgetItem(image_name)
            self.reports_table.setItem(row, 1, image_item)
            
            type_item = QTableWidgetItem(report['type'])
            self.reports_table.setItem(row, 2, type_item)
            
            template_item = QTableWidgetItem(report['template'])
            self.reports_table.setItem(row, 3, template_item)
            
            size_str = self.format_file_size(report['size'])
            size_item = QTableWidgetItem(size_str)
            self.reports_table.setItem(row, 4, size_item)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            view_btn = QPushButton("👁️")
            view_btn.setToolTip("View Report")
            view_btn.setMaximumSize(30, 25)
            view_btn.clicked.connect(lambda checked, path=report['path']: self.view_report(path))
            actions_layout.addWidget(view_btn)
            
            delete_btn = QPushButton("🗑️")
            delete_btn.setToolTip("Delete Report")
            delete_btn.setMaximumSize(30, 25)
            delete_btn.clicked.connect(lambda checked, path=report['path']: self.delete_report(path))
            actions_layout.addWidget(delete_btn)
            
            self.reports_table.setCellWidget(row, 5, actions_widget)
    
    def view_report(self, report_path: str):
        """View the selected report"""
        try:
            report_file = Path(report_path)
            
            if report_file.suffix == '.html':
                # Open HTML reports in browser
                webbrowser.open(f"file://{report_file.absolute()}")
            elif report_file.suffix == '.pdf':
                # Open PDF with default system viewer
                os.startfile(str(report_file)) if os.name == 'nt' else os.system(f'open "{report_file}"')
            else:
                # For JSON/XML, show in text viewer
                with open(report_file, 'r') as f:
                    content = f.read()
                
                viewer_dialog = QDialog(self)
                viewer_dialog.setWindowTitle(f"Report Viewer - {report_file.name}")
                viewer_dialog.resize(800, 600)
                
                layout = QVBoxLayout(viewer_dialog)
                
                text_browser = QTextBrowser()
                if report_file.suffix == '.json':
                    # Pretty print JSON
                    try:
                        json_data = json.loads(content)
                        content = json.dumps(json_data, indent=2)
                    except:
                        pass
                
                text_browser.setPlainText(content)
                text_browser.setStyleSheet(f"""
                    QTextBrowser {{
                        background-color: {get_current_theme_color('surface')};
                        border: 1px solid {get_current_theme_color('border')};
                        border-radius: 4px;
                        font-family: 'Courier New', monospace;
                        font-size: 11px;
                        padding: 10px;
                    }}
                """)
                layout.addWidget(text_browser)
                
                close_btn = QPushButton("Close")
                close_btn.clicked.connect(viewer_dialog.close)
                layout.addWidget(close_btn)
                
                viewer_dialog.exec_()
                
        except Exception as e:
            show_error_message(self, "View Error", f"Failed to view report: {str(e)}")
    
    def delete_report(self, report_path: str):
        """Delete the selected report"""
        try:
            report_file = Path(report_path)
            
            # Confirm deletion
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete the report:\n{report_file.name}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                report_file.unlink()
                show_success_message(self, "Report Deleted", f"Report {report_file.name} has been deleted")
                self.load_report_history()  # Refresh the list
                
        except Exception as e:
            show_error_message(self, "Delete Error", f"Failed to delete report: {str(e)}")


class ReportsTab(QWidget):
    """Main reports tab widget for forensic report generation and management"""
    
    report_generated = pyqtSignal(str)  # file_path
    log_message = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_analysis_data = None
        self.report_worker = None
        self.templates = {}
        
        self.init_ui()
        self.setup_connections()
        self.load_templates()
    
    def init_ui(self):
        """Initialize the reports tab UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setSizes([400, 600])
        main_layout.addWidget(main_splitter)
        
        # Left panel - Report generation controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Report history and preview
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Status bar
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)
    
    def create_left_panel(self) -> QWidget:
        """Create the left control panel"""
        panel = QFrame()
        panel.setStyleSheet(create_card_style())
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("📋 Report Generation")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Analysis data section
        data_group = QGroupBox("Analysis Data")
        data_layout = QVBoxLayout(data_group)
        
        self.data_status_label = QLabel("No analysis data loaded")
        self.data_status_label.setStyleSheet(f"""
            QLabel {{
                padding: 10px;
                border: 2px dashed {get_current_theme_color('border')};
                border-radius: 8px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('text_secondary')};
                text-align: center;
            }}
        """)
        data_layout.addWidget(self.data_status_label)
        
        # Load analysis data button
        self.load_data_btn = QPushButton("📁 Load Analysis Data")
        self.load_data_btn.setStyleSheet(create_button_style('secondary'))
        self.load_data_btn.clicked.connect(self.load_analysis_data)
        data_layout.addWidget(self.load_data_btn)
        
        layout.addWidget(data_group)
        
        # Report configuration
        config_group = QGroupBox("Report Configuration")
        config_layout = QFormLayout(config_group)
        
        # Report title
        self.report_title = QLineEdit("Forensic Analysis Report")
        config_layout.addRow("Title:", self.report_title)
        
        # Report type
        self.report_type = QComboBox()
        self.report_type.addItems(["HTML", "PDF", "JSON", "XML"])
        self.report_type.currentTextChanged.connect(self.on_report_type_changed)
        config_layout.addRow("Format:", self.report_type)
        
        # Template selection
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Forensic Template", "Summary Template", "Custom Template"])
        config_layout.addRow("Template:", self.template_combo)
        
        # Edit template button
        self.edit_template_btn = QPushButton("✏️ Edit Template")
        self.edit_template_btn.setStyleSheet(create_button_style('secondary'))
        self.edit_template_btn.clicked.connect(self.edit_template)
        config_layout.addRow("", self.edit_template_btn)
        
        # Include options
        self.include_metadata = QCheckBox("Include Metadata")
        self.include_metadata.setChecked(True)
        config_layout.addRow("", self.include_metadata)
        
        self.include_visualizations = QCheckBox("Include Visualizations")
        self.include_visualizations.setChecked(True)
        config_layout.addRow("", self.include_visualizations)
        
        self.include_raw_data = QCheckBox("Include Raw Analysis Data")
        self.include_raw_data.setChecked(False)
        config_layout.addRow("", self.include_raw_data)
        
        layout.addWidget(config_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit("reports/generated")
        output_dir_layout.addWidget(self.output_dir)
        
        browse_dir_btn = QPushButton("📁")
        browse_dir_btn.setMaximumWidth(40)
        browse_dir_btn.clicked.connect(self.browse_output_directory)
        output_dir_layout.addWidget(browse_dir_btn)
        
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        # Filename pattern
        self.filename_pattern = QLineEdit("analysis_{image}_{timestamp}")
        self.filename_pattern.setToolTip("Use {image}, {timestamp}, {date} as placeholders")
        output_layout.addRow("Filename Pattern:", self.filename_pattern)
        
        layout.addWidget(output_group)
        
        # Generation controls
        controls_group = QGroupBox("Generation Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Generate report button
        self.generate_btn = QPushButton("🚀 Generate Report")
        self.generate_btn.setStyleSheet(create_button_style('success'))
        self.generate_btn.clicked.connect(self.generate_report)
        self.generate_btn.setEnabled(False)
        controls_layout.addWidget(self.generate_btn)
        
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
        
        # Cancel button
        self.cancel_btn = QPushButton("⏹️ Cancel")
        self.cancel_btn.setStyleSheet(create_button_style('danger'))
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.setVisible(False)
        controls_layout.addWidget(self.cancel_btn)
        
        layout.addWidget(controls_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with history and preview"""
        panel = QFrame()
        panel.setStyleSheet(create_card_style())
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
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
        
        # Report history tab
        self.history_widget = ReportHistoryWidget()
        self.right_tabs.addTab(self.history_widget, "📚 History")
        
        # Preview tab
        self.preview_widget = self.create_preview_widget()
        self.right_tabs.addTab(self.preview_widget, "👁️ Preview")
        
        # Templates tab
        self.templates_widget = self.create_templates_widget()
        self.right_tabs.addTab(self.templates_widget, "📄 Templates")
        
        layout.addWidget(self.right_tabs)
        
        return panel
    
    def create_preview_widget(self) -> QWidget:
        """Create the preview widget"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        
        # Preview controls
        controls_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("🔄 Update Preview")
        self.preview_btn.setStyleSheet(create_button_style('primary'))
        self.preview_btn.clicked.connect(self.update_preview)
        controls_layout.addWidget(self.preview_btn)
        
        controls_layout.addStretch()
        
        self.auto_preview = QCheckBox("Auto Preview")
        self.auto_preview.setChecked(True)
        controls_layout.addWidget(self.auto_preview)
        
        layout.addLayout(controls_layout)
        
        # Preview display
        self.preview_display = QTextBrowser()
        self.preview_display.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                padding: 10px;
                font-size: 12px;
            }}
        """)
        self.preview_display.setHtml("<p>No preview available. Load analysis data and configure report settings.</p>")
        layout.addWidget(self.preview_display)
        
        return widget
    
    def create_templates_widget(self) -> QWidget:
        """Create the templates management widget"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        
        # Templates controls
        controls_layout = QHBoxLayout()
        
        title = QLabel("📄 Report Templates")
        title.setFont(QFont("", 12, QFont.Bold))
        controls_layout.addWidget(title)
        
        controls_layout.addStretch()
        
        new_template_btn = QPushButton("➕ New Template")
        new_template_btn.setStyleSheet(create_button_style('primary'))
        new_template_btn.clicked.connect(self.create_new_template)
        controls_layout.addWidget(new_template_btn)
        
        import_template_btn = QPushButton("📥 Import")
        import_template_btn.setStyleSheet(create_button_style('secondary'))
        import_template_btn.clicked.connect(self.import_template)
        controls_layout.addWidget(import_template_btn)
        
        layout.addLayout(controls_layout)
        
        # Templates list
        self.templates_list = QListWidget()
        self.templates_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {get_current_theme_color('surface')};
                border: 1px solid {get_current_theme_color('border')};
                border-radius: 4px;
                padding: 5px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {get_current_theme_color('border')};
                border-radius: 3px;
                margin: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {get_current_theme_color('primary')};
            }}
        """)
        layout.addWidget(self.templates_list)
        
        # Template actions
        actions_layout = QHBoxLayout()
        
        edit_btn = QPushButton("✏️ Edit")
        edit_btn.setStyleSheet(create_button_style('secondary'))
        edit_btn.clicked.connect(self.edit_selected_template)
        actions_layout.addWidget(edit_btn)
        
        duplicate_btn = QPushButton("📋 Duplicate")
        duplicate_btn.setStyleSheet(create_button_style('secondary'))
        duplicate_btn.clicked.connect(self.duplicate_template)
        actions_layout.addWidget(duplicate_btn)
        
        export_btn = QPushButton("📤 Export")
        export_btn.setStyleSheet(create_button_style('secondary'))
        export_btn.clicked.connect(self.export_template)
        actions_layout.addWidget(export_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.setStyleSheet(create_button_style('danger'))
        delete_btn.clicked.connect(self.delete_template)
        actions_layout.addWidget(delete_btn)
        
        layout.addLayout(actions_layout)
        
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
        
        self.status_label = QLabel("Ready to generate reports")
        self.status_label.setStyleSheet(f"color: {get_current_theme_color('text_secondary')};")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Reports count
        self.reports_count_label = QLabel("")
        self.reports_count_label.setStyleSheet(f"color: {get_current_theme_color('text_secondary')};")
        layout.addWidget(self.reports_count_label)
        
        return status_bar
    
    def setup_connections(self):
        """Setup signal connections"""
        # Connect auto-preview
        self.report_title.textChanged.connect(self.on_settings_changed)
        self.template_combo.currentTextChanged.connect(self.on_settings_changed)
        self.include_metadata.toggled.connect(self.on_settings_changed)
        self.include_visualizations.toggled.connect(self.on_settings_changed)
        self.include_raw_data.toggled.connect(self.on_settings_changed)
    
    def load_templates(self):
        """Load available report templates"""
        try:
            templates_dir = Path("reports/templates")
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Load default templates
            self.templates = {
                "Forensic Template": self.get_default_forensic_template(),
                "Summary Template": self.get_default_summary_template(),
                "Custom Template": self.get_default_custom_template()
            }
            
            # Load custom templates from disk
            for template_file in templates_dir.glob("*.html"):
                try:
                    with open(template_file, 'r') as f:
                        content = f.read()
                    template_name = template_file.stem
                    self.templates[template_name] = content
                except Exception as e:
                    continue
            
            # Update templates list
            self.update_templates_list()
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {str(e)}")
    
    def update_templates_list(self):
        """Update the templates list widget"""
        self.templates_list.clear()
        for template_name in self.templates.keys():
            item = QListWidgetItem(template_name)
            self.templates_list.addItem(item)
        
        # Update template combo
        current_selection = self.template_combo.currentText()
        self.template_combo.clear()
        self.template_combo.addItems(list(self.templates.keys()))
        
        # Restore selection if it still exists
        index = self.template_combo.findText(current_selection)
        if index >= 0:
            self.template_combo.setCurrentIndex(index)
    
    def get_default_forensic_template(self) -> str:
        """Get the default forensic template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{{title}}</title>
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
    <div class="header">
        <h1>{{title}}</h1>
        <p><strong>Image:</strong> {{image_path}}</p>
        <p><strong>Analysis Date:</strong> {{date}}</p>
        <p><strong>Report Generated:</strong> {{timestamp}}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{{summary}}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Results</h2>
        {{analysis_results}}
    </div>
    
    <div class="section">
        <h2>Metadata Information</h2>
        <pre class="metadata">{{metadata}}</pre>
    </div>
</body>
</html>
        """
    
    def get_default_summary_template(self) -> str:
        """Get the default summary template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{{title}} - Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary-box { border: 2px solid #007acc; border-radius: 10px; padding: 20px; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px; padding: 15px; border-radius: 5px; background: #f0f8ff; }
        .metric h3 { margin: 0; color: #007acc; }
        .metric p { margin: 5px 0; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>{{title}}</h1>
    <p><strong>Date:</strong> {{date}} | <strong>Image:</strong> {{image_path}}</p>
    
    <div class="summary-box">
        <h2>Quick Summary</h2>
        <p>{{summary}}</p>
    </div>
    
    <div class="summary-box">
        <h2>Key Metrics</h2>
        {{analysis_results}}
    </div>
</body>
</html>
        """
    
    def get_default_custom_template(self) -> str:
        """Get the default custom template"""
        return """
{
    "report_title": "{{title}}",
    "generation_date": "{{timestamp}}",
    "image_path": "{{image_path}}",
    "analysis_date": "{{date}}",
    "summary": "{{summary}}",
    "metadata": {{metadata}},
    "analysis_results": {{analysis_results}}
}
        """
    
    def load_analysis_data(self):
        """Load analysis data from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Analysis Data",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.current_analysis_data = json.load(f)
                
                # Update status
                image_name = os.path.basename(self.current_analysis_data.get('image_path', 'Unknown'))
                self.data_status_label.setText(f"✅ Loaded analysis for: {image_name}")
                self.data_status_label.setStyleSheet(f"""
                    QLabel {{
                        padding: 10px;
                        border: 2px solid {get_current_theme_color('success')};
                        border-radius: 8px;
                        background-color: {get_current_theme_color('surface')};
                        color: {get_current_theme_color('success')};
                        text-align: center;
                    }}
                """)
                
                # Enable generate button
                self.generate_btn.setEnabled(True)
                
                # Update preview if auto-preview is enabled
                if self.auto_preview.isChecked():
                    self.update_preview()
                
                self.add_log_message(f"Analysis data loaded from: {file_path}")
                
            except Exception as e:
                show_error_message(self, "Load Error", f"Failed to load analysis data: {str(e)}")
                self.add_log_message(f"Error loading analysis data: {str(e)}")
    
    def set_analysis_data(self, analysis_data: Dict[str, Any]):
        """Set analysis data directly (called from analysis tab)"""
        self.current_analysis_data = analysis_data
        
        # Update status
        image_name = os.path.basename(analysis_data.get('image_path', 'Unknown'))
        self.data_status_label.setText(f"✅ Analysis data ready: {image_name}")
        self.data_status_label.setStyleSheet(f"""
            QLabel {{
                padding: 10px;
                border: 2px solid {get_current_theme_color('success')};
                border-radius: 8px;
                background-color: {get_current_theme_color('surface')};
                color: {get_current_theme_color('success')};
                text-align: center;
            }}
        """)
        
        # Enable generate button
        self.generate_btn.setEnabled(True)
        
        # Update preview if auto-preview is enabled
        if self.auto_preview.isChecked():
            self.update_preview()
        
        self.add_log_message("Analysis data set from current analysis")
    
    def browse_output_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir.text()
        )
        if directory:
            self.output_dir.setText(directory)
    
    def on_report_type_changed(self, report_type: str):
        """Handle report type change to adjust template options"""
        if report_type in ["JSON", "XML"]:
            self.template_combo.setEnabled(False)
            self.edit_template_btn.setEnabled(False)
        else:
            self.template_combo.setEnabled(True)
            self.edit_template_btn.setEnabled(True)
    
    def edit_template(self):
        """Open template editor dialog"""
        current_template = self.templates.get(self.template_combo.currentText(), "")
        editor = ReportTemplateEditor(current_template, self)
        if editor.exec_() == QDialog.Accepted:
            new_content = editor.get_template_content()
            self.templates[self.template_combo.currentText()] = new_content
            self.update_templates_list()
            if self.auto_preview.isChecked():
                self.update_preview()
    
    def on_settings_changed(self):
        """Update preview when settings change if auto-preview is enabled"""
        if self.auto_preview.isChecked() and self.current_analysis_data:
            self.update_preview()
    
    def generate_report(self):
        """Start the report generation process"""
        if not self.current_analysis_data:
            show_error_message(self, "Error", "No analysis data loaded. Please load or set analysis data first.")
            return
        
        # Prepare report data based on include options
        report_data = self.current_analysis_data.copy()
        report_data['title'] = self.report_title.text()
        report_data['date'] = datetime.now().strftime('%Y-%m-%d')
        report_data['timestamp'] = datetime.now().isoformat()
        
        if not self.include_metadata.isChecked():
            report_data.pop('metadata', None)
        if not self.include_visualizations.isChecked():
            report_data.pop('visualizations', None)
        if not self.include_raw_data.isChecked():
            report_data.pop('raw_data', None)
        
        # Generate output filename
        image_name = os.path.basename(report_data.get('image_path', 'unknown'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.filename_pattern.text().format(image=image_name, timestamp=timestamp, date=datetime.now().strftime('%Y-%m-%d'))
        output_path = os.path.join(self.output_dir.text(), f"{filename}.{self.report_type.lower()}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Start worker thread
        self.report_worker = ReportGenerationWorker(report_data, output_path, self.report_type.currentText().lower(), self.templates[self.template_combo.currentText()])
        self.report_worker.progress_updated.connect(self.progress_bar.setValue)
        self.report_worker.report_generated.connect(self.on_report_generated)
        self.report_worker.error_occurred.connect(self.on_generation_error)
        self.report_worker.log_message.connect(self.add_log_message)
        
        self.progress_bar.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Generating report...")
        
        self.report_worker.start()
    
    def cancel_generation(self):
        """Cancel the current report generation"""
        if self.report_worker and self.report_worker.isRunning():
            self.report_worker.terminate()
            self.progress_bar.setVisible(False)
            self.cancel_btn.setVisible(False)
            self.generate_btn.setEnabled(True)
            self.status_label.setText("Report generation cancelled")
            self.add_log_message("Report generation cancelled by user")
    
    def on_report_generated(self, file_path: str):
        """Handle successful report generation"""
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Report generated successfully")
        self.report_generated.emit(file_path)
        self.history_widget.load_report_history()
        self.add_log_message(f"Report generated: {file_path}")
        self.update_reports_count()
        
        # Update preview if auto-preview is enabled
        if self.auto_preview.isChecked():
            self.update_preview()
    
    def on_generation_error(self, error_msg: str):
        """Handle errors during report generation"""
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Error generating report")
        show_error_message(self, "Generation Error", f"Report generation failed: {error_msg}")
        self.add_log_message(f"Error generating report: {error_msg}")
    
    def update_preview(self):
        """Update the preview display with current settings"""
        if not self.current_analysis_data:
            self.preview_display.setHtml("<p>No preview available. Load analysis data and configure report settings.</p>")
            return
        
        report_data = self.current_analysis_data.copy()
        report_data['title'] = self.report_title.text()
        report_data['date'] = datetime.now().strftime('%Y-%m-%d')
        report_data['timestamp'] = datetime.now().isoformat()
        
        if not self.include_metadata.isChecked():
            report_data.pop('metadata', None)
        if not self.include_visualizations.isChecked():
            report_data.pop('visualizations', None)
        if not self.include_raw_data.isChecked():
            report_data.pop('raw_data', None)
        
        template = self.templates[self.template_combo.currentText()]
        preview_content = template
        
        for key, value in report_data.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, dict):
                value = json.dumps(value, indent=2)
            preview_content = preview_content.replace(placeholder, str(value))
        
        if self.report_type.currentText() == "HTML":
            self.preview_display.setHtml(preview_content)
        else:
            self.preview_display.setPlainText(preview_content)
    
    def create_new_template(self):
        """Create a new template"""
        editor = ReportTemplateEditor("", self)
        if editor.exec_() == QDialog.Accepted:
            new_content = editor.get_template_content()
            new_name = f"Custom Template {len(self.templates)}"
            self.templates[new_name] = new_content
            self.update_templates_list()
    
    def edit_selected_template(self):
        """Edit the selected template"""
        selected_items = self.templates_list.selectedItems()
        if selected_items:
            template_name = selected_items[0].text()
            editor = ReportTemplateEditor(self.templates[template_name], self)
            if editor.exec_() == QDialog.Accepted:
                self.templates[template_name] = editor.get_template_content()
                self.update_templates_list()
                if self.auto_preview.isChecked():
                    self.update_preview()
    
    def duplicate_template(self):
        """Duplicate the selected template"""
        selected_items = self.templates_list.selectedItems()
        if selected_items:
            template_name = selected_items[0].text()
            new_name = f"{template_name} Copy"
            self.templates[new_name] = self.templates[template_name]
            self.update_templates_list()
    
    def export_template(self):
        """Export the selected template"""
        selected_items = self.templates_list.selectedItems()
        if selected_items:
            template_name = selected_items[0].text()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Template",
                f"template_{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "HTML Files (*.html);;JSON Files (*.json);;XML Files (*.xml);;All Files (*)"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(self.templates[template_name])
                    show_success_message(self, "Export Success", f"Template exported to {file_path}")
                except Exception as e:
                    show_error_message(self, "Export Error", f"Failed to export template: {str(e)}")
    
    def import_template(self):
        """Import a template from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Template",
            "",
            "HTML Files (*.html);;JSON Files (*.json);;XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                template_name = Path(file_path).stem
                self.templates[template_name] = content
                self.update_templates_list()
            except Exception as e:
                show_error_message(self, "Import Error", f"Failed to import template: {str(e)}")
    
    def delete_template(self):
        """Delete the selected template"""
        selected_items = self.templates_list.selectedItems()
        if selected_items:
            template_name = selected_items[0].text()
            reply = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete the template '{template_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                del self.templates[template_name]
                self.update_templates_list()
                if self.auto_preview.isChecked():
                    self.update_preview()
    
    def add_log_message(self, message: str):
        """Add a message to the status log"""
        self.log_message.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.status_label.setText(message)
    
    def update_reports_count(self):
        """Update the reports count in the status bar"""
        reports_dir = Path("reports/generated")
        count = len(list(reports_dir.glob("*.*"))) if reports_dir.exists() else 0
        self.reports_count_label.setText(f"Reports: {count}")