#!/usr/bin/env python3
"""
CodeGuard - Source Code Plagiarism Detection System
Main entry point for the plagiarism detection application.

Usage:
    python main.py --directory /path/to/source/files
    python main.py --files file1.py file2.py file3.js
    python main.py --help
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.detector import PlagiarismDetector
from utils.file_handler import FileHandler
from utils.reporter import ReportGenerator


def main():
    """Main application entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.directory and not args.files:
        parser.error("Either --directory or --files must be specified")
    
    try:
        # Initialize components
        detector = PlagiarismDetector(
            similarity_threshold=args.threshold,
            min_tokens=args.min_tokens,
            min_df=args.min_df,
            max_df=args.max_df
        )
        
        file_handler = FileHandler()
        reporter = ReportGenerator()
        
        # Get list of files to analyze
        if args.directory:
            print(f"Scanning directory: {args.directory}")
            
            if not Path(args.directory).exists():
                print(f"Error: Directory does not exist: {args.directory}")
                sys.exit(1)
            
            # Get directory statistics
            stats = file_handler.get_directory_stats(args.directory)
            print(f"Found {stats['total_files']} source files")
            print(f"Total size: {stats['total_size']} bytes")
            print(f"Total lines: {stats['total_lines']}")
            
            if stats['total_files'] < 2:
                print("Error: At least 2 source files are required for comparison")
                sys.exit(1)
            
            # Detect plagiarism in directory
            report = detector.detect_in_directory(
                args.directory,
                file_extensions=args.extensions,
                recursive=args.recursive,
                progress_callback=print_progress if args.verbose else None
            )
        
        else:
            print(f"Analyzing {len(args.files)} files")
            
            # Validate file paths
            valid_files, invalid_files = file_handler.validate_file_paths(args.files)
            
            if invalid_files:
                print(f"Warning: {len(invalid_files)} invalid files will be skipped:")
                for file_path in invalid_files:
                    print(f"  - {file_path}")
            
            if len(valid_files) < 2:
                print("Error: At least 2 valid source files are required for comparison")
                sys.exit(1)
            
            # Detect plagiarism between files
            report = detector.detect_between_files(
                valid_files,
                progress_callback=print_progress if args.verbose else None
            )
        
        # Generate and save reports
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating reports in: {output_dir}")
        
        # Console report (always generated)
        print("\n" + "="*60)
        reporter.print_console_report(report, detailed=args.detailed)
        
        # Generate file reports based on format options
        timestamp = int(time.time())
        
        if args.format in ['html', 'all']:
            html_path = output_dir / f"plagiarism_report_{timestamp}.html"
            reporter.generate_html_report(report, str(html_path))
            print(f"HTML report saved: {html_path}")
        
        if args.format in ['csv', 'all']:
            csv_path = output_dir / f"plagiarism_report_{timestamp}.csv"
            reporter.generate_csv_report(report, str(csv_path))
            print(f"CSV report saved: {csv_path}")
        
        if args.format in ['json', 'all']:
            json_path = output_dir / f"plagiarism_report_{timestamp}.json"
            reporter.generate_json_report(report, str(json_path))
            print(f"JSON report saved: {json_path}")
        
        # Generate similarity heatmap if requested
        if args.heatmap:
            heatmap_path = output_dir / f"similarity_heatmap_{timestamp}.png"
            reporter.generate_similarity_heatmap(report, str(heatmap_path))
            print(f"Similarity heatmap saved: {heatmap_path}")
        
        # Generate detailed comparisons for high similarity cases
        if args.detailed_comparisons:
            comparison_dir = output_dir / f"detailed_comparisons_{timestamp}"
            comparison_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(report.plagiarism_cases[:args.max_comparisons], 1):
                if result.similarity_score >= args.comparison_threshold:
                    file1_name = Path(result.file1_path).stem
                    file2_name = Path(result.file2_path).stem
                    comparison_path = comparison_dir / f"comparison_{i}_{file1_name}_vs_{file2_name}.html"
                    
                    reporter.generate_detailed_comparison(result, str(comparison_path))
                    print(f"Detailed comparison saved: {comparison_path}")
        
        # Save model if requested
        if args.save_model:
            model_path = output_dir / f"plagiarism_model_{timestamp}.pkl"
            detector.save_model(str(model_path))
            print(f"Model saved: {model_path}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Files analyzed: {report.total_files}")
        print(f"  Plagiarism cases: {len(report.plagiarism_cases)}")
        print(f"  Processing time: {report.processing_time:.2f} seconds")
        
        if report.plagiarism_cases:
            highest_similarity = max(case.similarity_score for case in report.plagiarism_cases)
            print(f"  Highest similarity: {highest_similarity:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="CodeGuard - Source Code Plagiarism Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --directory ./student_submissions --threshold 0.8
  python main.py --files file1.py file2.py file3.js --format html
  python main.py --directory ./code --heatmap --detailed-comparisons
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory containing source code files to analyze'
    )
    input_group.add_argument(
        '--files', '-f',
        nargs='+',
        help='List of specific files to analyze'
    )
    
    # Detection parameters
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.7,
        help='Similarity threshold for plagiarism detection (default: 0.7)'
    )
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=10,
        help='Minimum number of tokens required for analysis (default: 10)'
    )
    parser.add_argument(
        '--min-df',
        type=int,
        default=1,
        help='Minimum document frequency for TF-IDF (default: 1)'
    )
    parser.add_argument(
        '--max-df',
        type=float,
        default=0.95,
        help='Maximum document frequency for TF-IDF (default: 0.95)'
    )
    
    # File filtering options
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.py', '.js'],
        help='File extensions to include (default: .py .js)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Search directories recursively (default: True)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='Disable recursive directory search'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./reports',
        help='Output directory for reports (default: ./reports)'
    )
    parser.add_argument(
        '--format',
        choices=['html', 'csv', 'json', 'all'],
        default='html',
        help='Output format for reports (default: html)'
    )
    
    # Visualization options
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Generate similarity heatmap'
    )
    parser.add_argument(
        '--detailed-comparisons',
        action='store_true',
        help='Generate detailed comparison reports for high similarity cases'
    )
    parser.add_argument(
        '--comparison-threshold',
        type=float,
        default=0.8,
        help='Minimum similarity for detailed comparisons (default: 0.8)'
    )
    parser.add_argument(
        '--max-comparisons',
        type=int,
        default=10,
        help='Maximum number of detailed comparisons to generate (default: 10)'
    )
    
    # Model options
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model for later use'
    )
    parser.add_argument(
        '--load-model',
        type=str,
        help='Load a previously saved model'
    )
    
    # Display options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information in console report'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser


def print_progress(message: str):
    """Progress callback function"""
    print(f"Progress: {message}")


if __name__ == "__main__":
    sys.exit(main())