#!/usr/bin/env python3
"""
StegAnalysis Suite - Report Generation Module
Professional PDF, JSON, and HTML report generation for steganography analysis
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import base64

# Report generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import Color, black, blue, red, green
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab not available. PDF generation will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Chart generation will be limited.")

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available. HTML template rendering will be limited.")


class ReportGenerator:
    """Professional report generation for steganography analysis results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reports_config = config.get('reports', {})
        
        # Configuration
        self.output_dir = Path(self.reports_config.get('output_dir', 'reports/generated'))
        self.template_dir = Path(self.reports_config.get('template_dir', 'reports/templates'))
        self.formats = self.reports_config.get('formats', ['pdf', 'json'])
        self.include_visualizations = self.reports_config.get('include_visualizations', True)
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF settings
        self.pdf_settings = self.reports_config.get('pdf_settings', {})
        self.page_size = A4 if self.pdf_settings.get('page_size', 'A4') == 'A4' else letter
        
        # Visualization settings
        self.viz_settings = self.reports_config.get('visualization_settings', {})
        self.dpi = self.viz_settings.get('dpi', 300)
        self.figure_size = tuple(self.viz_settings.get('figure_size', [10, 8]))
        
    def generate_single_report(self, analysis_results: Dict[str, Any], output_path: str) -> Dict[str, str]:
        """
        Generate comprehensive report for single image analysis
        
        Args:
            analysis_results: Complete analysis results dictionary
            output_path: Base path for output files (without extension)
            
        Returns:
            Dictionary mapping format to generated file path
        """
        generated_files = {}
        
        try:
            # Prepare report data
            report_data = self._prepare_single_report_data(analysis_results)
            
            # Generate visualizations if requested
            if self.include_visualizations and MATPLOTLIB_AVAILABLE:
                chart_paths = self._generate_analysis_charts(analysis_results, output_path)
                report_data['charts'] = chart_paths
            
            # Generate reports in requested formats
            for format_type in self.formats:
                if format_type == 'pdf' and REPORTLAB_AVAILABLE:
                    pdf_path = f"{output_path}.pdf"
                    self._generate_pdf_report(report_data, pdf_path)
                    generated_files['pdf'] = pdf_path
                
                elif format_type == 'json':
                    json_path = f"{output_path}.json"
                    self._generate_json_report(report_data, json_path)
                    generated_files['json'] = json_path
                
                elif format_type == 'html' and JINJA2_AVAILABLE:
                    html_path = f"{output_path}.html"
                    self._generate_html_report(report_data, html_path)
                    generated_files['html'] = html_path
            
            self.logger.info(f"Single analysis report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate single report: {str(e)}")
        
        return generated_files
    
    def generate_batch_report(self, batch_results: List[Dict[str, Any]], output_path: str) -> Dict[str, str]:
        """
        Generate comprehensive report for batch analysis
        
        Args:
            batch_results: List of analysis results dictionaries
            output_path: Base path for output files (without extension)
            
        Returns:
            Dictionary mapping format to generated file path
        """
        generated_files = {}
        
        try:
            # Prepare batch report data
            report_data = self._prepare_batch_report_data(batch_results)
            
            # Generate batch visualizations
            if self.include_visualizations and MATPLOTLIB_AVAILABLE:
                chart_paths = self._generate_batch_charts(batch_results, output_path)
                report_data['charts'] = chart_paths
            
            # Generate reports in requested formats
            for format_type in self.formats:
                if format_type == 'pdf' and REPORTLAB_AVAILABLE:
                    pdf_path = f"{output_path}_batch.pdf"
                    self._generate_batch_pdf_report(report_data, pdf_path)
                    generated_files['pdf'] = pdf_path
                
                elif format_type == 'json':
                    json_path = f"{output_path}_batch.json"
                    self._generate_json_report(report_data, json_path)
                    generated_files['json'] = json_path
                
                elif format_type == 'html' and JINJA2_AVAILABLE:
                    html_path = f"{output_path}_batch.html"
                    self._generate_batch_html_report(report_data, html_path)
                    generated_files['html'] = html_path
            
            self.logger.info(f"Batch analysis report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch report: {str(e)}")
        
        return generated_files
    
    def _prepare_single_report_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data structure for single image report"""
        
        # Extract key information
        image_path = results.get('image_path', 'Unknown')
        timestamp = results.get('timestamp', datetime.now().isoformat())
        overall_verdict = results.get('overall_verdict', 'unknown')
        confidence_score = results.get('confidence_score', 0.0)
        
        # Detection results summary
        detection_results = results.get('detection_results', {})
        algorithms_used = results.get('algorithms_used', [])
        
        # Statistical analysis summary
        statistical_analysis = results.get('statistical_analysis', {})
        
        # Forensic analysis summary
        forensic_analysis = results.get('forensic_analysis', {})
        
        report_data = {
            'metadata': {
                'report_type': 'single_image_analysis',
                'generation_timestamp': datetime.now().isoformat(),
                'analysis_timestamp': timestamp,
                'image_path': image_path,
                'suite_version': '1.0.0'
            },
            'executive_summary': {
                'verdict': overall_verdict,
                'confidence': confidence_score,
                'algorithms_used': algorithms_used,
                'key_findings': self._extract_key_findings(results),
                'risk_assessment': self._assess_risk_level(overall_verdict, confidence_score)
            },
            'detection_analysis': {
                'summary': self._summarize_detection_results(detection_results),
                'detailed_results': detection_results
            },
            'statistical_analysis': {
                'summary': self._summarize_statistical_analysis(statistical_analysis),
                'detailed_results': statistical_analysis
            },
            'forensic_analysis': {
                'summary': self._summarize_forensic_analysis(forensic_analysis),
                'detailed_results': forensic_analysis
            },
            'recommendations': self._generate_recommendations(results),
            'technical_details': {
                'configuration': self._get_analysis_configuration(),
                'processing_info': self._get_processing_info(results)
            }
        }
        
        return report_data
    
    def _prepare_batch_report_data(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data structure for batch analysis report"""
        
        # Calculate batch statistics
        total_images = len(batch_results)
        steganographic_count = sum(1 for r in batch_results if r.get('overall_verdict') == 'steganographic')
        suspicious_count = sum(1 for r in batch_results if r.get('overall_verdict') == 'suspicious')
        clean_count = sum(1 for r in batch_results if r.get('overall_verdict') == 'clean')
        
        # Average confidence
        confidences = [r.get('confidence_score', 0.0) for r in batch_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Algorithm usage statistics
        algorithm_stats = self._calculate_algorithm_statistics(batch_results)
        
        report_data = {
            'metadata': {
                'report_type': 'batch_analysis',
                'generation_timestamp': datetime.now().isoformat(),
                'total_images_analyzed': total_images,
                'suite_version': '1.0.0'
            },
            'executive_summary': {
                'batch_statistics': {
                    'total_images': total_images,
                    'steganographic': steganographic_count,
                    'suspicious': suspicious_count,
                    'clean': clean_count,
                    'average_confidence': avg_confidence
                },
                'detection_rate': steganographic_count / total_images if total_images > 0 else 0.0,
                'key_patterns': self._identify_batch_patterns(batch_results),
                'risk_distribution': self._calculate_risk_distribution(batch_results)
            },
            'algorithm_performance': algorithm_stats,
            'detailed_results': batch_results,
            'individual_summaries': [self._create_individual_summary(r) for r in batch_results],
            'recommendations': self._generate_batch_recommendations(batch_results),
            'statistical_overview': self._create_statistical_overview(batch_results)
        }
        
        return report_data
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate professional PDF report"""
        if not REPORTLAB_AVAILABLE:
            self.logger.error("ReportLab not available for PDF generation")
            return
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.HexColor('#2E4B8A')
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.HexColor('#1F3A6B')
            )
            
            story = []
            
            # Title page
            story.append(Paragraph("StegAnalysis Suite", title_style))
            story.append(Paragraph("Steganography Detection Report", styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Metadata table
            metadata = report_data['metadata']
            metadata_data = [
                ['Analysis Date:', metadata.get('analysis_timestamp', 'Unknown')],
                ['Image Path:', metadata.get('image_path', 'Unknown')],
                ['Report Generated:', metadata.get('generation_timestamp', 'Unknown')],
                ['Suite Version:', metadata.get('suite_version', '1.0.0')]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 30))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            exec_summary = report_data['executive_summary']
            verdict = exec_summary.get('verdict', 'unknown').upper()
            confidence = exec_summary.get('confidence', 0.0)
            
            # Verdict styling based on result
            if verdict == 'STEGANOGRAPHIC':
                verdict_color = colors.red
            elif verdict == 'SUSPICIOUS':
                verdict_color = colors.orange
            else:
                verdict_color = colors.green
            
            story.append(Paragraph(f"<b>Analysis Verdict:</b> <font color='{verdict_color.hexval()}'>{verdict}</font>", styles['Normal']))
            story.append(Paragraph(f"<b>Confidence Score:</b> {confidence:.2f}", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Key findings
            key_findings = exec_summary.get('key_findings', [])
            if key_findings:
                story.append(Paragraph("<b>Key Findings:</b>", styles['Normal']))
                for finding in key_findings:
                    story.append(Paragraph(f"• {finding}", styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Detection Analysis
            story.append(Paragraph("Detection Analysis", heading_style))
            
            detection_summary = report_data['detection_analysis']['summary']
            if detection_summary:
                # Create detection results table
                detection_data = [['Algorithm', 'Result', 'Confidence']]
                
                for algo, result in detection_summary.items():
                    if isinstance(result, dict):
                        detected = "DETECTED" if result.get('detected', False) else "CLEAN"
                        confidence = f"{result.get('confidence', 0.0):.2f}"
                    else:
                        detected = "DETECTED" if result else "CLEAN"
                        confidence = "N/A"
                    
                    detection_data.append([algo.upper(), detected, confidence])
                
                detection_table = Table(detection_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                detection_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6E6E6')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                
                story.append(detection_table)
                story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommendations", heading_style))
            
            recommendations = report_data.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
            else:
                story.append(Paragraph("No specific recommendations at this time.", styles['Normal']))
            
            # Add charts if available
            if 'charts' in report_data:
                for chart_type, chart_path in report_data['charts'].items():
                    if os.path.exists(chart_path):
                        story.append(Spacer(1, 20))
                        story.append(Paragraph(f"{chart_type.replace('_', ' ').title()}", heading_style))
                        story.append(RLImage(chart_path, width=5*inch, height=4*inch))
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"PDF report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {str(e)}")
    
    def _generate_json_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate detailed JSON report"""
        try:
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"JSON report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {str(e)}")
    
    def _generate_html_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate interactive HTML report"""
        try:
            html_content = self._generate_simple_html(report_data)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {str(e)}")
    
    def _generate_batch_html_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate HTML report for batch analysis"""
        try:
            metadata = report_data['metadata']
            exec_summary = report_data['executive_summary']
            batch_stats = exec_summary['batch_statistics']
            
            # Generate individual results table
            individual_results_html = ""
            for summary in report_data.get('individual_summaries', []):
                verdict_class = f"verdict-{summary['verdict'].lower()}"
                individual_results_html += f"""
                <tr>
                    <td>{summary['image_path']}</td>
                    <td class="{verdict_class}">{summary['verdict'].upper()}</td>
                    <td>{summary['confidence']:.2f}</td>
                    <td>{', '.join(summary['algorithms_detected'])}</td>
                </tr>
                """
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>StegAnalysis Batch Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2E4B8A; color: white; padding: 20px; text-align: center; }}
        .summary {{ background-color: #f4f4f4; padding: 15px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .verdict-steganographic {{ color: #c62828; }}
        .verdict-suspicious {{ color: #ef6c00; }}
        .verdict-clean {{ color: #2e7d32; }}
        .verdict-unknown {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>StegAnalysis Suite - Batch Report</h1>
    </div>
    
    <div class="summary">
        <h2>Batch Analysis Summary</h2>
        <p><strong>Generated:</strong> {metadata.get('generation_timestamp', 'Unknown')}</p>
        <p><strong>Total Images:</strong> {batch_stats['total_images']}</p>
        <p><strong>Detection Rate:</strong> {exec_summary['detection_rate']:.1%}</p>
        
        <table>
            <tr><th>Verdict</th><th>Count</th><th>Percentage</th></tr>
            <tr><td>Steganographic</td><td>{batch_stats['steganographic']}</td><td>{batch_stats['steganographic']/batch_stats['total_images']*100:.1f}%</td></tr>
            <tr><td>Suspicious</td><td>{batch_stats['suspicious']}</td><td>{batch_stats['suspicious']/batch_stats['total_images']*100:.1f}%</td></tr>
            <tr><td>Clean</td><td>{batch_stats['clean']}</td><td>{batch_stats['clean']/batch_stats['total_images']*100:.1f}%</td></tr>
        </table>
        
        <h3>Individual Results</h3>
        <table>
            <tr><th>Image Path</th><th>Verdict</th><th>Confidence</th><th>Detecting Algorithms</th></tr>
            {individual_results_html}
        </table>
    </div>
    
    <div class="charts">
        {f'<img src="{report_data["charts"]["batch_summary"]}" style="max-width: 100%;">' if report_data.get('charts', {}).get('batch_summary') else ''}
    </div>
</body>
</html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            self.logger.info(f"Batch HTML report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch HTML report: {str(e)}")
    
    def _generate_simple_html(self, report_data: Dict[str, Any]) -> str:
        """Generate simple HTML report"""
        metadata = report_data.get('metadata', {})
        exec_summary = report_data.get('executive_summary', {})
        
        verdict = exec_summary.get('verdict', 'unknown').upper()
        confidence = exec_summary.get('confidence', 0.0)
        
        # Determine verdict class
        verdict_class = f"verdict-{verdict.lower()}"
        
        # Generate algorithm results
        detection_analysis = report_data.get('detection_analysis', {})
        algorithm_results_html = ""
        
        for algo, result in detection_analysis.get('summary', {}).items():
            if isinstance(result, dict):
                detected = result.get('detected', False)
                algo_confidence = result.get('confidence', 0.0)
            else:
                detected = bool(result)
                algo_confidence = 0.5 if result else 0.0
            
            result_class = "result-detected" if detected else "result-clean"
            result_text = "DETECTED" if detected else "CLEAN"
            
            algorithm_results_html += f"""
            <div class="algorithm-card">
                <div class="algorithm-name">{algo.upper()}</div>
                <div class="algorithm-result {result_class}">{result_text}</div>
                <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    Confidence: {algo_confidence:.2f}
                </div>
            </div>
            """
        
        # Generate key findings
        key_findings_html = ""
        for finding in exec_summary.get('key_findings', []):
            key_findings_html += f"<li>{finding}</li>"
        
        # Generate recommendations
        recommendations_html = ""
        for rec in report_data.get('recommendations', []):
            recommendations_html += f"<li>{rec}</li>"
        
        # Add charts if available
        charts_html = ""
        if 'charts' in report_data:
            for chart_type, chart_path in report_data['charts'].items():
                if os.path.exists(chart_path):
                    charts_html += f'<img src="{chart_path}" style="max-width: 100%; margin: 20px 0;">'
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StegAnalysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #2E4B8A, #4A6BB8); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 30px; }}
        .section-title {{ color: #2E4B8A; font-size: 1.8em; margin-bottom: 20px; border-left: 4px solid #4A6BB8; padding-left: 15px; }}
        .verdict-box {{ padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; font-size: 1.3em; font-weight: bold; }}
        .verdict-steganographic {{ background-color: #ffebee; color: #c62828; border: 2px solid #ef5350; }}
        .verdict-suspicious {{ background-color: #fff3e0; color: #ef6c00; border: 2px solid #ff9800; }}
        .verdict-clean {{ background-color: #e8f5e8; color: #2e7d32; border: 2px solid #4caf50; }}
        .verdict-unknown {{ background-color: #f5f5f5; color: #666; border: 2px solid #999; }}
        .algorithm-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .algorithm-card {{ background-color: #f8f9fa; border-radius: 8px; padding: 15px; border: 1px solid #dee2e6; }}
        .algorithm-name {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
        .algorithm-result {{ padding: 5px 10px; border-radius: 20px; font-size: 0.9em; font-weight: bold; text-align: center; }}
        .result-detected {{ background-color: #ffcdd2; color: #c62828; }}
        .result-clean {{ background-color: #c8e6c9; color: #2e7d32; }}
        .findings-list {{ list-style: none; padding: 0; }}
        .findings-list li {{ background-color: #f8f9fa; margin: 10px 0; padding: 15px; border-left: 4px solid #4A6BB8; border-radius: 0 8px 8px 0; }}
        .recommendations-list {{ list-style: none; padding: 0; }}
        .recommendations-list li {{ background-color: #e3f2fd; margin: 10px 0; padding: 15px; border-left: 4px solid #2196f3; border-radius: 0 8px 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>StegAnalysis Suite</h1>
            <div>Comprehensive Steganography Detection Report</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">Analysis Overview</h2>
                <p><strong>Image:</strong> {metadata.get('image_path', 'Unknown')}</p>
                <p><strong>Analysis Date:</strong> {metadata.get('analysis_timestamp', 'Unknown')}</p>
                <p><strong>Report Generated:</strong> {metadata.get('generation_timestamp', 'Unknown')}</p>
                
                <div class="verdict-box {verdict_class}">
                    Analysis Verdict: {verdict}
                </div>
                
                <p><strong>Confidence Level:</strong> {confidence:.1%}</p>
                <p><strong>Risk Assessment:</strong> {exec_summary.get('risk_assessment', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Detection Results</h2>
                <div class="algorithm-grid">
                    {algorithm_results_html}
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Key Findings</h2>
                <ul class="findings-list">
                    {key_findings_html}
                </ul>
            </div>
            
            <div class="section">
                <h2 class="section-title">Recommendations</h2>
                <ul class="recommendations-list">
                    {recommendations_html}
                </ul>
            </div>
            
            <div class="section">
                <h2 class="section-title">Visualizations</h2>
                {charts_html}
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_analysis_charts(self, results: Dict[str, Any], base_path: str) -> Dict[str, str]:
        """Generate visualization charts for analysis results"""
        chart_paths = {}
        
        if not MATPLOTLIB_AVAILABLE:
            return chart_paths
        
        try:
            plt.style.use('default')
            
            # Detection results chart
            detection_results = results.get('detection_results', {})
            if detection_results:
                chart_path = f"{base_path}_detection_chart.png"
                self._create_detection_chart(detection_results, chart_path)
                chart_paths['detection_results'] = chart_path
            
            # Confidence chart
            chart_path = f"{base_path}_confidence_chart.png"
            self._create_confidence_chart(results, chart_path)
            chart_paths['confidence_breakdown'] = chart_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis charts: {str(e)}")
        
        return chart_paths
    
    def _create_detection_chart(self, detection_results: Dict[str, Any], output_path: str):
        """Create detection results visualization"""
        try:
            algorithms = []
            confidences = []
            detected = []
            
            for algo, result in detection_results.items():
                algorithms.append(algo.upper())
                if isinstance(result, dict):
                    confidences.append(result.get('confidence', 0.0))
                    detected.append(result.get('detected', False))
                else:
                    confidences.append(0.5 if result else 0.0)
                    detected.append(result)
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Confidence scores bar chart
            colors = ['red' if d else 'green' for d in detected]
            bars = ax.bar(algorithms, confidences, color=colors, alpha=0.7)
            ax.set_title('Detection Confidence by Algorithm')
            ax.set_ylabel('Confidence Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create detection chart: {str(e)}")
    
    def _create_confidence_chart(self, results: Dict[str, Any], output_path: str):
        """Create confidence breakdown visualization"""
        try:
            overall_confidence = results.get('confidence_score', 0.0)
            verdict = results.get('overall_verdict', 'unknown')
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Prepare data
            categories = ['Overall Confidence']
            values = [overall_confidence]
            
            # Color based on confidence level
            if overall_confidence >= 0.8:
                bar_color = 'red'
            elif overall_confidence >= 0.6:
                bar_color = 'orange'
            elif overall_confidence >= 0.4:
                bar_color = 'yellow'
            else:
                bar_color = 'green'
            
            # Add category-specific confidence
            if verdict == 'steganographic':
                categories.append('Steganographic Confidence')
                values.append(overall_confidence * 0.8)
            elif verdict == 'suspicious':
                categories.append('Suspicious Confidence')
                values.append(overall_confidence * 0.6)
            else:
                categories.append('Clean Confidence')
                values.append(overall_confidence * 0.4)
            
            # Create bar chart
            bars = ax.bar(categories, values, color=bar_color, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Confidence Score')
            ax.set_title(f'Analysis Confidence: {overall_confidence:.1%}\nVerdict: {verdict.upper()}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Confidence chart generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create confidence chart: {str(e)}")
    
    def _generate_batch_charts(self, batch_results: List[Dict[str, Any]], base_path: str) -> Dict[str, str]:
        """Generate charts for batch analysis results"""
        chart_paths = {}
        
        if not MATPLOTLIB_AVAILABLE:
            return chart_paths
        
        try:
            # Batch summary chart
            chart_path = f"{base_path}_batch_summary.png"
            self._create_batch_summary_chart(batch_results, chart_path)
            chart_paths['batch_summary'] = chart_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch charts: {str(e)}")
        
        return chart_paths
    
    def _create_batch_summary_chart(self, batch_results: List[Dict[str, Any]], output_path: str):
        """Create batch analysis summary chart"""
        try:
            # Count verdicts
            verdicts = [r.get('overall_verdict', 'unknown') for r in batch_results]
            verdict_counts = {}
            for verdict in ['steganographic', 'suspicious', 'clean', 'unknown']:
                verdict_counts[verdict] = verdicts.count(verdict)
            
            # Remove empty categories
            verdict_counts = {k: v for k, v in verdict_counts.items() if v > 0}
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Pie chart
            colors = {'steganographic': 'red', 'suspicious': 'orange', 'clean': 'green', 'unknown': 'gray'}
            pie_colors = [colors.get(k, 'blue') for k in verdict_counts.keys()]
            
            ax.pie(verdict_counts.values(), labels=verdict_counts.keys(), autopct='%1.1f%%', 
                   colors=pie_colors, alpha=0.8)
            ax.set_title('Analysis Results Distribution')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create batch summary chart: {str(e)}")
    
    def _generate_batch_pdf_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate PDF report for batch analysis"""
        if not REPORTLAB_AVAILABLE:
            self.logger.error("ReportLab not available for PDF generation")
            return
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
            styles = getSampleStyleSheet()
            
            # Custom styles
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.HexColor('#1F3A6B')
            )
            
            story = []
            
            # Title
            story.append(Paragraph("StegAnalysis Suite - Batch Analysis Report", styles['Title']))
            story.append(Spacer(1, 20))
            
            # Metadata
            metadata = report_data['metadata']
            story.append(Paragraph(f"<b>Report Generated:</b> {metadata.get('generation_timestamp', 'Unknown')}", styles['Normal']))
            story.append(Paragraph(f"<b>Total Images Analyzed:</b> {metadata.get('total_images_analyzed', 0)}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            exec_summary = report_data['executive_summary']
            batch_stats = exec_summary['batch_statistics']
            
            story.append(Paragraph("Batch Analysis Summary", heading_style))
            
            # Statistics table
            stats_data = [
                ['Metric', 'Count', 'Percentage'],
                ['Total Images', batch_stats['total_images'], '100%'],
                ['Steganographic', batch_stats['steganographic'], f"{batch_stats['steganographic']/batch_stats['total_images']*100:.1f}%" if batch_stats['total_images'] > 0 else '0%'],
                ['Suspicious', batch_stats['suspicious'], f"{batch_stats['suspicious']/batch_stats['total_images']*100:.1f}%" if batch_stats['total_images'] > 0 else '0%'],
                ['Clean', batch_stats['clean'], f"{batch_stats['clean']/batch_stats['total_images']*100:.1f}%" if batch_stats['total_images'] > 0 else '0%']
            ]
            
            stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6E6E6')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 20))
            
            # Individual Results
            story.append(Paragraph("Individual Analysis Results", heading_style))
            individual_data = [['Image Path', 'Verdict', 'Confidence', 'Detecting Algorithms']]
            
            for summary in report_data.get('individual_summaries', []):
                individual_data.append([
                    summary['image_path'],
                    summary['verdict'].upper(),
                    f"{summary['confidence']:.2f}",
                    ', '.join(summary['algorithms_detected'])
                ])
            
            individual_table = Table(individual_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.5*inch])
            individual_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6E6E6')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            story.append(individual_table)
            story.append(Spacer(1, 20))
            
            # Add batch summary chart if available
            if 'charts' in report_data and 'batch_summary' in report_data['charts']:
                chart_path = report_data['charts']['batch_summary']
                if os.path.exists(chart_path):
                    story.append(Paragraph("Batch Results Distribution", heading_style))
                    story.append(RLImage(chart_path, width=5*inch, height=4*inch))
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"Batch PDF report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch PDF report: {str(e)}")
    
    # Helper methods for data processing
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results"""
        findings = []
        
        # Detection findings
        detection_results = results.get('detection_results', {})
        detected_by = [algo for algo, result in detection_results.items() 
                      if (isinstance(result, dict) and result.get('detected', False)) or 
                         (isinstance(result, bool) and result)]
        
        if detected_by:
            findings.append(f"Steganography detected by: {', '.join(detected_by)}")
        
        # Statistical findings
        statistical_analysis = results.get('statistical_analysis', {})
        if statistical_analysis.get('anomaly_scores', {}).get('overall_anomaly', 0) > 0.5:
            findings.append("High statistical anomaly score detected")
        
        if not findings:
            findings.append("No significant steganographic indicators detected")
        
        return findings
    
    def _assess_risk_level(self, verdict: str, confidence: float) -> str:
        """Assess overall risk level"""
        if verdict == 'steganographic' and confidence >= 0.8:
            return 'HIGH'
        elif verdict == 'steganographic' and confidence >= 0.6:
            return 'MEDIUM-HIGH'
        elif verdict == 'suspicious' or (verdict == 'steganographic' and confidence < 0.6):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _summarize_detection_results(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize detection results"""
        summary = {}
        for algo, result in detection_results.items():
            if isinstance(result, dict):
                summary[algo] = {
                    'detected': result.get('detected', False),
                    'confidence': result.get('confidence', 0.0)
                }
            else:
                summary[algo] = {'detected': bool(result), 'confidence': 0.5 if result else 0.0}
        return summary
    
    def _summarize_statistical_analysis(self, statistical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize statistical analysis results"""
        if not statistical_analysis:
            return {}
        
        summary = {
            'anomaly_scores': statistical_analysis.get('anomaly_scores', {}),
            'entropy_analysis': {}
        }
        
        # Entropy summary
        entropy_data = statistical_analysis.get('entropy_analysis', {})
        if entropy_data:
            for channel, data in entropy_data.items():
                if isinstance(data, dict):
                    summary['entropy_analysis'][channel] = {
                        'global_entropy': data.get('global_entropy', 0),
                        'lsb_entropy': data.get('lsb_entropy', 0)
                    }
        
        return summary
    
    def _summarize_forensic_analysis(self, forensic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize forensic analysis results"""
        if not forensic_analysis:
            return {}
        
        return forensic_analysis.get('forensic_summary', {})
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        verdict = results.get('overall_verdict', 'unknown')
        confidence = results.get('confidence_score', 0.0)
        
        if verdict == 'steganographic':
            if confidence >= 0.8:
                recommendations.extend([
                    "High confidence steganography detection - immediate investigation recommended",
                    "Attempt data extraction using appropriate tools",
                    "Preserve original file for forensic analysis"
                ])
            else:
                recommendations.extend([
                    "Possible steganography detected - further analysis recommended",
                    "Consider using additional detection methods",
                    "Verify results with alternative tools"
                ])
        
        elif verdict == 'suspicious':
            recommendations.extend([
                "Suspicious patterns detected - monitor closely",
                "Consider additional forensic analysis",
                "Compare with known clean samples if available"
            ])
        
        else:
            recommendations.extend([
                "No steganographic content detected in current analysis",
                "File appears to be clean based on applied algorithms"
            ])
        
        return recommendations
    
    def _get_analysis_configuration(self) -> Dict[str, Any]:
        """Get current analysis configuration"""
        return {
            'detection_algorithms': self.config.get('detection', {}).get('algorithms', []),
            'ml_models_enabled': bool(self.config.get('ml_models', {})),
            'gpu_acceleration': self.config.get('gpu', {}).get('enabled', False),
            'statistical_analysis': bool(self.config.get('statistical_analysis', {})),
            'forensic_analysis': bool(self.config.get('forensic_analysis', {}))
        }
    
    def _get_processing_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing information"""
        return {
            'timestamp': results.get('timestamp', 'Unknown'),
            'algorithms_used': results.get('algorithms_used', []),
            'processing_time': results.get('processing_time', 'Not measured'),
            'errors_encountered': 'error' in results
        }
    
    def _calculate_algorithm_statistics(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate algorithm performance statistics"""
        stats = {}
        
        for result in batch_results:
            detection_results = result.get('detection_results', {})
            for algo, algo_result in detection_results.items():
                if algo not in stats:
                    stats[algo] = {
                        'total_runs': 0,
                        'detections': 0,
                        'avg_confidence': 0.0,
                        'confidences': []
                    }
                
                stats[algo]['total_runs'] += 1
                
                if isinstance(algo_result, dict):
                    if algo_result.get('detected', False):
                        stats[algo]['detections'] += 1
                    confidence = algo_result.get('confidence', 0.0)
                    stats[algo]['confidences'].append(confidence)
                elif algo_result:
                    stats[algo]['detections'] += 1
                    stats[algo]['confidences'].append(1.0)
                else:
                    stats[algo]['confidences'].append(0.0)
        
        # Calculate averages
        for algo in stats:
            if stats[algo]['confidences']:
                stats[algo]['avg_confidence'] = sum(stats[algo]['confidences']) / len(stats[algo]['confidences'])
            stats[algo]['detection_rate'] = stats[algo]['detections'] / stats[algo]['total_runs'] if stats[algo]['total_runs'] > 0 else 0.0
            # Remove raw confidences for cleaner output
            del stats[algo]['confidences']
        
        return stats
    
    def _identify_batch_patterns(self, batch_results: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in batch analysis results"""
        patterns = []
        
        total_images = len(batch_results)
        if total_images == 0:
            return patterns
        
        # High detection rate pattern
        steganographic_count = sum(1 for r in batch_results if r.get('overall_verdict') == 'steganographic')
        detection_rate = steganographic_count / total_images
        
        if detection_rate > 0.5:
            patterns.append(f"High steganography detection rate: {detection_rate:.1%}")
        elif detection_rate > 0.2:
            patterns.append(f"Moderate steganography detection rate: {detection_rate:.1%}")
        
        # Confidence patterns
        confidences = [r.get('confidence_score', 0.0) for r in batch_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        if avg_confidence > 0.8:
            patterns.append("High average confidence in detections")
        elif avg_confidence < 0.3:
            patterns.append("Low average confidence - results may be uncertain")
        
        return patterns
    
    def _calculate_risk_distribution(self, batch_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate risk level distribution"""
        risk_counts = {'HIGH': 0, 'MEDIUM-HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for result in batch_results:
            verdict = result.get('overall_verdict', 'unknown')
            confidence = result.get('confidence_score', 0.0)
            risk_level = self._assess_risk_level(verdict, confidence)
            risk_counts[risk_level] += 1
        
        return risk_counts
    
    def _create_individual_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for individual result in batch"""
        return {
            'image_path': result.get('image_path', 'Unknown'),
            'verdict': result.get('overall_verdict', 'unknown'),
            'confidence': result.get('confidence_score', 0.0),
            'algorithms_detected': [
                algo for algo, algo_result in result.get('detection_results', {}).items()
                if (isinstance(algo_result, dict) and algo_result.get('detected', False)) or
                   (isinstance(algo_result, bool) and algo_result)
            ]
        }
    
    def _generate_batch_recommendations(self, batch_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for batch analysis"""
        recommendations = []
        
        total_images = len(batch_results)
        steganographic_count = sum(1 for r in batch_results if r.get('overall_verdict') == 'steganographic')
        
        if steganographic_count > total_images * 0.1:  # More than 10%
            recommendations.append("High proportion of steganographic content detected - comprehensive investigation recommended")
        
        if steganographic_count > 0:
            recommendations.append("Isolate and further analyze detected steganographic images")
            recommendations.append("Implement additional monitoring for similar content")
        else:
            recommendations.append("No steganographic content detected in batch analysis")
        
        return recommendations
    
    def _create_statistical_overview(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create statistical overview of batch results"""
        if not batch_results:
            return {}
        
        # Basic statistics
        total_images = len(batch_results)
        verdicts = [r.get('overall_verdict', 'unknown') for r in batch_results]
        confidences = [r.get('confidence_score', 0.0) for r in batch_results]
        
        # Verdict statistics
        verdict_stats = {
            'steganographic': verdicts.count('steganographic'),
            'suspicious': verdicts.count('suspicious'),
            'clean': verdicts.count('clean'),
            'unknown': verdicts.count('unknown')
        }
        
        # Confidence statistics
        confidence_stats = {
            'mean': sum(confidences) / len(confidences) if confidences else 0.0,
            'min': min(confidences) if confidences else 0.0,
            'max': max(confidences) if confidences else 0.0
        }
        
        return {
            'total_images': total_images,
            'verdict_distribution': verdict_stats,
            'confidence_statistics': confidence_stats,
            'detection_rate': verdict_stats['steganographic'] / total_images if total_images > 0 else 0.0
        }


# Example usage and testing functions
def create_sample_config() -> Dict[str, Any]:
    """Create sample configuration for testing"""
    return {
        'reports': {
            'output_dir': 'reports/generated',
            'template_dir': 'reports/templates',
            'formats': ['pdf', 'json', 'html'],
            'include_visualizations': True,
            'pdf_settings': {
                'page_size': 'A4'
            },
            'visualization_settings': {
                'dpi': 300,
                'figure_size': [10, 8]
            }
        },
        'detection': {
            'algorithms': ['lsb', 'dct', 'dwt']
        },
        'ml_models': {},
        'gpu': {'enabled': False},
        'statistical_analysis': {},
        'forensic_analysis': {}
    }


def create_sample_results() -> Dict[str, Any]:
    """Create sample analysis results for testing"""
    return {
        'image_path': '/path/to/test_image.jpg',
        'timestamp': datetime.now().isoformat(),
        'overall_verdict': 'steganographic',
        'confidence_score': 0.85,
        'algorithms_used': ['lsb', 'dct', 'dwt'],
        'detection_results': {
            'lsb': {'detected': True, 'confidence': 0.9},
            'dct': {'detected': True, 'confidence': 0.8},
            'dwt': {'detected': False, 'confidence': 0.2}
        },
        'statistical_analysis': {
            'anomaly_scores': {'overall_anomaly': 0.7},
            'entropy_analysis': {
                'red': {'global_entropy': 7.2, 'lsb_entropy': 0.8},
                'green': {'global_entropy': 7.1, 'lsb_entropy': 0.9},
                'blue': {'global_entropy': 7.0, 'lsb_entropy': 0.7}
            }
        },
        'forensic_analysis': {
            'forensic_summary': {'overall_assessment': 'suspicious'}
        }
    }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config = create_sample_config()
    generator = ReportGenerator(config)
    
    # Test single report generation
    sample_results = create_sample_results()
    output_path = "test_report"
    
    generated_files = generator.generate_single_report(sample_results, output_path)
    print(f"Generated reports: {generated_files}")
    
    # Test batch report generation
    batch_results = [create_sample_results() for _ in range(3)]
    batch_output_path = "test_batch_report"
    
    batch_files = generator.generate_batch_report(batch_results, batch_output_path)
    print(f"Generated batch reports: {batch_files}")