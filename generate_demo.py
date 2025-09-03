#!/usr/bin/env python3
"""
Generate comprehensive demo data and reports for CodeGuard.
This script creates sample analysis results for demonstration.
"""

import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.detector import PlagiarismDetector
from utils.reporter import ReportGenerator
from utils.file_handler import FileHandler


def generate_sample_reports():
    """Generate comprehensive sample reports"""
    
    print("🛡️  CodeGuard Demo Generation")
    print("=" * 50)
    
    # Initialize components
    detector = PlagiarismDetector(similarity_threshold=0.7, min_tokens=5)
    reporter = ReportGenerator()
    file_handler = FileHandler()
    
    # Sample data directory
    sample_dir = Path("data/sample_codes")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    if not sample_dir.exists():
        print("❌ Sample data directory not found!")
        return
    
    print("📁 Analyzing sample code files...")
    
    try:
        # Run detection on sample files
        report = detector.detect_in_directory(
            str(sample_dir),
            file_extensions=['.py', '.js'],
            recursive=False
        )
        
        print(f"✅ Analysis completed:")
        print(f"   - Files analyzed: {report.total_files}")
        print(f"   - Comparisons made: {report.total_comparisons}")
        print(f"   - Plagiarism cases: {len(report.plagiarism_cases)}")
        print(f"   - Processing time: {report.processing_time:.2f}s")
        
        # Generate all report formats
        timestamp = "demo"
        
        # HTML Report
        html_path = reports_dir / f"demo_report_{timestamp}.html"
        reporter.generate_html_report(report, str(html_path))
        print(f"📄 HTML report: {html_path}")
        
        # CSV Report
        csv_path = reports_dir / f"demo_report_{timestamp}.csv"
        reporter.generate_csv_report(report, str(csv_path))
        print(f"📊 CSV report: {csv_path}")
        
        # JSON Report
        json_path = reports_dir / f"demo_report_{timestamp}.json"
        reporter.generate_json_report(report, str(json_path))
        print(f"🔧 JSON report: {json_path}")
        
        # Generate similarity heatmap
        if report.total_files > 1:
            heatmap_path = reports_dir / f"similarity_heatmap_{timestamp}.png"
            reporter.generate_similarity_heatmap(report, str(heatmap_path), figsize=(10, 8))
            print(f"🔥 Similarity heatmap: {heatmap_path}")
        
        # Generate detailed comparisons
        if report.plagiarism_cases:
            comparison_dir = reports_dir / f"detailed_comparisons_{timestamp}"
            comparison_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(report.plagiarism_cases[:3], 1):
                file1_name = Path(result.file1_path).stem
                file2_name = Path(result.file2_path).stem
                comparison_path = comparison_dir / f"comparison_{i}_{file1_name}_vs_{file2_name}.html"
                
                reporter.generate_detailed_comparison(result, str(comparison_path))
                print(f"🔍 Detailed comparison {i}: {comparison_path}")
        
        # Generate demo data for web interface
        generate_web_demo_data(report, reports_dir)
        
        print("\n🎉 Demo generation completed successfully!")
        print(f"📂 All reports saved in: {reports_dir.absolute()}")
        print("\nTo view the demo:")
        print("1. Run: python demo_server.py")
        print("2. Open: http://localhost:8080/index.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_web_demo_data(report, output_dir):
    """Generate JSON data for web interface demo"""
    
    # Create demo data structure
    demo_data = {
        "analysis_results": {
            "files_analyzed": int(report.total_files),
            "total_comparisons": int(report.total_comparisons),
            "plagiarism_cases": len(report.plagiarism_cases),
            "processing_time": round(float(report.processing_time), 2),
            "threshold": float(report.threshold)
        },
        "similarity_statistics": {k: float(v) for k, v in report.statistics.items()},
        "plagiarism_cases": [],
        "similarity_matrix": report.similarity_matrix.tolist() if report.similarity_matrix.size > 0 else [],
        "file_names": [Path(fp).name for fp in report.file_paths],
        "timestamp": "2024-demo"
    }
    
    # Add plagiarism cases
    for i, case in enumerate(report.plagiarism_cases, 1):
        case_data = {
            "case_number": i,
            "file1_name": Path(case.file1_path).name,
            "file2_name": Path(case.file2_path).name,
            "similarity_score": round(float(case.similarity_score), 4),
            "is_plagiarism": bool(case.is_plagiarism),
            "file1_tokens": int(case.file1_tokens),
            "file2_tokens": int(case.file2_tokens),
            "common_tokens": int(case.common_tokens),
            "overlap_percentage": round((float(case.common_tokens) / max(float(case.file1_tokens), float(case.file2_tokens))) * 100, 1)
        }
        
        # Add contributing terms if available
        if case.explanation and 'contributing_terms' in case.explanation:
            terms = case.explanation['contributing_terms'][:5]
            case_data['contributing_terms'] = [
                {
                    'term': term['term'],
                    'contribution': round(term['contribution'], 4)
                }
                for term in terms
            ]
        
        demo_data['plagiarism_cases'].append(case_data)
    
    # Save demo data
    demo_json_path = output_dir / "demo_data.json"
    with open(demo_json_path, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"🌐 Web demo data: {demo_json_path}")


def create_additional_samples():
    """Create additional sample files for better demo"""
    
    sample_dir = Path("data/sample_codes")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a plagiarized version with more differences
    plagiarized_code = '''
def calculate_factorial(number):
    """Calculate factorial using recursion."""
    if number <= 1:
        return 1
    else:
        return number * calculate_factorial(number - 1)

def get_fibonacci_number(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    
    previous, current = 0, 1
    for i in range(2, n + 1):
        previous, current = current, previous + current
    return current

class MathOperations:
    def __init__(self):
        self.last_result = 0
    
    def add_numbers(self, a, b):
        result = a + b
        self.last_result = result
        return result
    
    def multiply_numbers(self, a, b):
        result = a * b
        self.last_result = result
        return result
    
    def square_number(self, num):
        result = num * num
        self.last_result = result
        return result

def main():
    math_ops = MathOperations()
    print(f"Addition: {math_ops.add_numbers(5, 3)}")
    print(f"Factorial: {calculate_factorial(5)}")
    print(f"Fibonacci: {get_fibonacci_number(10)}")

if __name__ == "__main__":
    main()
    '''
    
    with open(sample_dir / "plagiarized_sample.py", 'w') as f:
        f.write(plagiarized_code)
    
    print("📝 Created additional sample file: plagiarized_sample.py")


if __name__ == "__main__":
    print("Starting demo generation...")
    
    # Create additional samples
    create_additional_samples()
    
    # Generate reports
    success = generate_sample_reports()
    
    if success:
        print("\n🎯 Demo ready! Next steps:")
        print("1. python demo_server.py")
        print("2. Open browser to http://localhost:8080")
        print("3. Click on the generated reports to see results")
    else:
        print("\n❌ Demo generation failed. Check the error messages above.")
        sys.exit(1)