"""
Report generation utilities for plagiarism detection results.
Handles HTML, CSV, JSON, and console output formatting.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO

from core.detector import DetectionReport, DetectionResult


class ReportGenerator:
    def __init__(self):
        """Initialize report generator"""
        self.template_dir = Path(__file__).parent / "templates"
    
    def generate_html_report(self, report: DetectionReport, output_path: str) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            report: Detection report
            output_path: Output file path
            
        Returns:
            Path to generated HTML file
        """
        html_content = self._create_html_content(report)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def generate_csv_report(self, report: DetectionReport, output_path: str) -> str:
        """
        Generate CSV report with plagiarism cases.
        
        Args:
            report: Detection report
            output_path: Output file path
            
        Returns:
            Path to generated CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'File1', 'File2', 'Similarity_Score', 'Is_Plagiarism',
                'File1_Tokens', 'File2_Tokens', 'Common_Tokens',
                'File1_Name', 'File2_Name'
            ])
            
            # Write detection results
            for result in report.plagiarism_cases:
                file1_name = Path(result.file1_path).name
                file2_name = Path(result.file2_path).name
                
                writer.writerow([
                    result.file1_path,
                    result.file2_path,
                    f"{result.similarity_score:.4f}",
                    result.is_plagiarism,
                    result.file1_tokens,
                    result.file2_tokens,
                    result.common_tokens,
                    file1_name,
                    file2_name
                ])
        
        return str(output_path)
    
    def generate_json_report(self, report: DetectionReport, output_path: str) -> str:
        """
        Generate JSON report.
        
        Args:
            report: Detection report
            output_path: Output file path
            
        Returns:
            Path to generated JSON file
        """
        # Convert report to JSON-serializable format
        json_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_files': int(report.total_files),
                'total_comparisons': int(report.total_comparisons),
                'processing_time': float(report.processing_time),
                'threshold': float(report.threshold),
                'plagiarism_cases_count': len(report.plagiarism_cases)
            },
            'statistics': {k: float(v) for k, v in report.statistics.items()},
            'file_paths': report.file_paths,
            'plagiarism_cases': []
        }
        
        # Add plagiarism cases
        for result in report.plagiarism_cases:
            case_data = {
                'file1_path': result.file1_path,
                'file2_path': result.file2_path,
                'similarity_score': float(result.similarity_score),
                'is_plagiarism': bool(result.is_plagiarism),
                'file1_tokens': int(result.file1_tokens),
                'file2_tokens': int(result.file2_tokens),
                'common_tokens': int(result.common_tokens)
            }
            
            if result.explanation:
                case_data['explanation'] = result.explanation
            
            json_data['plagiarism_cases'].append(case_data)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def generate_similarity_heatmap(self, report: DetectionReport, 
                                   output_path: str, figsize: tuple = (12, 10)) -> str:
        """
        Generate similarity matrix heatmap.
        
        Args:
            report: Detection report
            output_path: Output image path
            figsize: Figure size tuple
            
        Returns:
            Path to generated image
        """
        plt.figure(figsize=figsize)
        
        # Create file name labels (shortened for readability)
        file_labels = [Path(fp).name[:20] + '...' if len(Path(fp).name) > 20 
                      else Path(fp).name for fp in report.file_paths]
        
        # Create heatmap
        sns.heatmap(report.similarity_matrix, 
                   annot=False,
                   cmap='RdYlBu_r',
                   vmin=0, vmax=1,
                   xticklabels=file_labels,
                   yticklabels=file_labels)
        
        plt.title(f'Code Similarity Matrix\n(Threshold: {report.threshold})')
        plt.xlabel('Files')
        plt.ylabel('Files')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def print_console_report(self, report: DetectionReport, 
                           detailed: bool = False) -> None:
        """
        Print report to console.
        
        Args:
            report: Detection report
            detailed: Whether to show detailed information
        """
        print("=" * 60)
        print("PLAGIARISM DETECTION REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Files analyzed: {report.total_files}")
        print(f"Total comparisons: {report.total_comparisons}")
        print(f"Processing time: {report.processing_time:.2f} seconds")
        print(f"Similarity threshold: {report.threshold}")
        print(f"Plagiarism cases found: {len(report.plagiarism_cases)}")
        print()
        
        # Statistics
        if report.statistics:
            print("SIMILARITY STATISTICS:")
            print("-" * 30)
            stats = report.statistics
            print(f"Mean similarity: {stats.get('mean', 0):.4f}")
            print(f"Median similarity: {stats.get('median', 0):.4f}")
            print(f"Standard deviation: {stats.get('std', 0):.4f}")
            print(f"Min similarity: {stats.get('min', 0):.4f}")
            print(f"Max similarity: {stats.get('max', 0):.4f}")
            print()
        
        # Plagiarism cases
        if report.plagiarism_cases:
            print("PLAGIARISM CASES:")
            print("-" * 40)
            
            for i, result in enumerate(report.plagiarism_cases, 1):
                file1_name = Path(result.file1_path).name
                file2_name = Path(result.file2_path).name
                
                print(f"{i}. {file1_name} ↔ {file2_name}")
                print(f"   Similarity: {result.similarity_score:.4f}")
                print(f"   Tokens: {result.file1_tokens} / {result.file2_tokens}")
                print(f"   Common tokens: {result.common_tokens}")
                
                if detailed and result.explanation:
                    exp = result.explanation
                    if 'contributing_terms' in exp:
                        terms = exp['contributing_terms'][:3]  # Top 3 terms
                        if terms:
                            term_names = [t['term'] for t in terms]
                            print(f"   Key terms: {', '.join(term_names)}")
                
                print()
        else:
            print("No plagiarism cases detected above the threshold.")
        
        print("=" * 60)
    
    def generate_detailed_comparison(self, result: DetectionResult, 
                                   output_path: str) -> str:
        """
        Generate detailed comparison report for a specific case.
        
        Args:
            result: Detection result
            output_path: Output HTML file path
            
        Returns:
            Path to generated HTML file
        """
        html_content = self._create_comparison_html(result)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _create_html_content(self, report: DetectionReport) -> str:
        """Create HTML content for the main report"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #e9e9e9; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; }}
        .cases {{ margin-top: 30px; }}
        .case {{ background: #fff; border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .case.high {{ border-left: 5px solid #ff4444; }}
        .case.medium {{ border-left: 5px solid #ffaa44; }}
        .case.low {{ border-left: 5px solid #44ff44; }}
        .similarity-score {{ font-size: 1.2em; font-weight: bold; }}
        .file-info {{ margin: 10px 0; }}
        .contributing-terms {{ margin-top: 10px; font-size: 0.9em; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Plagiarism Detection Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Files Analyzed:</strong> {report.total_files}</p>
        <p><strong>Total Comparisons:</strong> {report.total_comparisons}</p>
        <p><strong>Processing Time:</strong> {report.processing_time:.2f} seconds</p>
        <p><strong>Similarity Threshold:</strong> {report.threshold}</p>
        <p><strong>Plagiarism Cases Found:</strong> {len(report.plagiarism_cases)}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Mean Similarity</h3>
            <p>{report.statistics.get('mean', 0):.4f}</p>
        </div>
        <div class="stat-card">
            <h3>Median Similarity</h3>
            <p>{report.statistics.get('median', 0):.4f}</p>
        </div>
        <div class="stat-card">
            <h3>Max Similarity</h3>
            <p>{report.statistics.get('max', 0):.4f}</p>
        </div>
        <div class="stat-card">
            <h3>Standard Deviation</h3>
            <p>{report.statistics.get('std', 0):.4f}</p>
        </div>
    </div>
    
    <div class="cases">
        <h2>Plagiarism Cases</h2>
        """
        
        if not report.plagiarism_cases:
            html += "<p>No plagiarism cases detected above the threshold.</p>"
        else:
            for i, result in enumerate(report.plagiarism_cases, 1):
                # Determine severity class
                if result.similarity_score >= 0.9:
                    severity_class = "high"
                elif result.similarity_score >= 0.8:
                    severity_class = "medium"
                else:
                    severity_class = "low"
                
                file1_name = Path(result.file1_path).name
                file2_name = Path(result.file2_path).name
                
                html += f"""
        <div class="case {severity_class}">
            <h3>Case #{i}</h3>
            <div class="similarity-score">Similarity: {result.similarity_score:.4f}</div>
            <div class="file-info">
                <p><strong>File 1:</strong> {file1_name} ({result.file1_tokens} tokens)</p>
                <p><strong>File 2:</strong> {file2_name} ({result.file2_tokens} tokens)</p>
                <p><strong>Common Tokens:</strong> {result.common_tokens}</p>
            </div>
                """
                
                # Add contributing terms if available
                if result.explanation and 'contributing_terms' in result.explanation:
                    terms = result.explanation['contributing_terms'][:5]
                    if terms:
                        term_list = ', '.join([t['term'] for t in terms])
                        html += f'<div class="contributing-terms"><strong>Key Contributing Terms:</strong> {term_list}</div>'
                
                html += "</div>"
        
        # File list
        html += """
    </div>
    
    <h2>Analyzed Files</h2>
    <table>
        <tr><th>#</th><th>File Path</th><th>File Name</th></tr>
        """
        
        for i, file_path in enumerate(report.file_paths, 1):
            file_name = Path(file_path).name
            html += f"<tr><td>{i}</td><td>{file_path}</td><td>{file_name}</td></tr>"
        
        html += """
    </table>
</body>
</html>
        """
        
        return html
    
    def _create_comparison_html(self, result: DetectionResult) -> str:
        """Create detailed comparison HTML for a specific case"""
        file1_name = Path(result.file1_path).name
        file2_name = Path(result.file2_path).name
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detailed Comparison: {file1_name} vs {file2_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .comparison {{ display: flex; gap: 20px; margin-top: 20px; }}
        .file-section {{ flex: 1; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .file-section h3 {{ margin-top: 0; }}
        pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        .stats {{ background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .contributing-terms {{ margin-top: 20px; }}
        .term {{ display: inline-block; background-color: #007bff; color: white; 
                 padding: 5px 10px; margin: 3px; border-radius: 3px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Detailed Comparison</h1>
        <h2>{file1_name} ↔ {file2_name}</h2>
        <p><strong>Similarity Score:</strong> {result.similarity_score:.4f}</p>
        <p><strong>Classification:</strong> {'Plagiarism Detected' if result.is_plagiarism else 'No Plagiarism'}</p>
    </div>
    
    <div class="stats">
        <h3>File Statistics</h3>
        <p><strong>{file1_name}:</strong> {result.file1_tokens} tokens</p>
        <p><strong>{file2_name}:</strong> {result.file2_tokens} tokens</p>
        <p><strong>Common Tokens:</strong> {result.common_tokens}</p>
        <p><strong>Token Overlap:</strong> {(result.common_tokens / max(result.file1_tokens, result.file2_tokens) * 100):.1f}%</p>
    </div>
        """
        
        # Add contributing terms if available
        if result.explanation and 'contributing_terms' in result.explanation:
            terms = result.explanation['contributing_terms']
            if terms:
                html += '<div class="contributing-terms"><h3>Key Contributing Terms</h3>'
                for term_info in terms[:10]:  # Show top 10 terms
                    html += f'<span class="term">{term_info["term"]} ({term_info["contribution"]:.3f})</span>'
                html += '</div>'
        
        html += """
</body>
</html>
        """
        
        return html