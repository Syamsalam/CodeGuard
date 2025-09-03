#!/usr/bin/env python3
"""
Web interface for CodeGuard using Streamlit.
Provides interactive dashboard to visualize plagiarism detection process.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import json
import sys
import io
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.detector import PlagiarismDetector
from core.ast_tokenizer import ASTTokenizer
from utils.file_handler import FileHandler
from utils.reporter import ReportGenerator


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="CodeGuard - Plagiarism Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ CodeGuard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Source Code Plagiarism Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
    min_tokens = st.sidebar.number_input("Minimum Tokens", 1, 1000, 10)
    min_df = st.sidebar.number_input("Min Document Frequency", 1, 10, 1)
    max_df = st.sidebar.slider("Max Document Frequency", 0.1, 1.0, 0.95, 0.05)
    
    # File upload options
    st.sidebar.subheader("Input Options")
    input_method = st.sidebar.radio("Choose input method:", 
                                   ["Upload Files", "Use Sample Data", "Enter Code Manually"])
    
    # Main content area
    if input_method == "Upload Files":
        show_file_upload_interface(threshold, min_tokens, min_df, max_df)
    elif input_method == "Use Sample Data":
        show_sample_data_interface(threshold, min_tokens, min_df, max_df)
    else:
        show_manual_input_interface(threshold, min_tokens, min_df, max_df)


def show_file_upload_interface(threshold, min_tokens, min_df, max_df):
    """Show file upload interface"""
    st.markdown('<div class="step-header">📁 Step 1: Upload Source Code Files</div>', 
                unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose source code files",
        accept_multiple_files=True,
        type=['py', 'js', 'jsx', 'ts', 'tsx']
    )
    
    if uploaded_files and len(uploaded_files) >= 2:
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for uploaded_file in uploaded_files:
            temp_path = Path(temp_dir) / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(temp_path))
        
        # Show file info
        st.success(f"Uploaded {len(uploaded_files)} files successfully!")
        
        # Display file info
        file_info = []
        file_handler = FileHandler()
        
        for file_path in file_paths:
            try:
                info = file_handler.get_file_info(file_path)
                file_info.append({
                    'File': info['name'],
                    'Size (bytes)': info['size_bytes'],
                    'Lines': info['lines'],
                    'Extension': info['extension']
                })
            except:
                pass
        
        if file_info:
            df = pd.DataFrame(file_info)
            st.dataframe(df, use_container_width=True)
        
        # Run analysis button
        if st.button("🔍 Start Plagiarism Detection", type="primary"):
            run_plagiarism_analysis(file_paths, threshold, min_tokens, min_df, max_df)


def show_sample_data_interface(threshold, min_tokens, min_df, max_df):
    """Show sample data interface"""
    st.markdown('<div class="step-header">📝 Using Sample Data</div>', 
                unsafe_allow_html=True)
    
    sample_dir = Path("data/sample_codes")
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.py")) + list(sample_dir.glob("*.js"))
        
        st.info(f"Found {len(sample_files)} sample files")
        
        # Show sample files
        for file_path in sample_files:
            with st.expander(f"📄 {file_path.name}"):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    st.code(content, language=file_path.suffix[1:])
                except:
                    st.error("Could not read file")
        
        if st.button("🔍 Analyze Sample Files", type="primary"):
            file_paths = [str(f) for f in sample_files]
            run_plagiarism_analysis(file_paths, threshold, min_tokens, min_df, max_df)
    else:
        st.error("Sample data directory not found!")


def show_manual_input_interface(threshold, min_tokens, min_df, max_df):
    """Show manual code input interface"""
    st.markdown('<div class="step-header">✍️ Manual Code Input</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Code File 1")
        lang1 = st.selectbox("Language 1", ["python", "javascript"], key="lang1")
        code1 = st.text_area("Enter code 1:", height=300, key="code1")
        filename1 = st.text_input("Filename 1:", value=f"code1.{'py' if lang1=='python' else 'js'}")
    
    with col2:
        st.subheader("Code File 2")
        lang2 = st.selectbox("Language 2", ["python", "javascript"], key="lang2")
        code2 = st.text_area("Enter code 2:", height=300, key="code2")
        filename2 = st.text_input("Filename 2:", value=f"code2.{'py' if lang2=='python' else 'js'}")
    
    if code1.strip() and code2.strip() and st.button("🔍 Compare Codes", type="primary"):
        # Save codes to temporary files
        temp_dir = tempfile.mkdtemp()
        
        file_paths = []
        for i, (code, filename) in enumerate([(code1, filename1), (code2, filename2)], 1):
            temp_path = Path(temp_dir) / filename
            with open(temp_path, 'w') as f:
                f.write(code)
            file_paths.append(str(temp_path))
        
        run_plagiarism_analysis(file_paths, threshold, min_tokens, min_df, max_df)


def run_plagiarism_analysis(file_paths, threshold, min_tokens, min_df, max_df):
    """Run the complete plagiarism analysis workflow"""
    
    with st.spinner("🔄 Running plagiarism detection..."):
        # Initialize detector
        detector = PlagiarismDetector(
            similarity_threshold=threshold,
            min_tokens=min_tokens,
            min_df=min_df,
            max_df=max_df
        )
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Tokenization
        status_text.text("Step 1/4: Tokenizing source code...")
        progress_bar.progress(25)
        show_tokenization_process(file_paths)
        
        # Step 2: TF-IDF Vectorization
        status_text.text("Step 2/4: Computing TF-IDF vectors...")
        progress_bar.progress(50)
        
        # Step 3: Similarity Calculation
        status_text.text("Step 3/4: Calculating similarities...")
        progress_bar.progress(75)
        
        # Step 4: Generate Report
        status_text.text("Step 4/4: Generating report...")
        progress_bar.progress(90)
        
        # Run detection
        try:
            report = detector.detect_between_files(file_paths)
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed!")
            
            # Show results
            show_analysis_results(report, detector)
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")


def show_tokenization_process(file_paths):
    """Show AST tokenization process"""
    st.markdown('<div class="step-header">🔧 Step 2: AST Tokenization Process</div>', 
                unsafe_allow_html=True)
    
    tokenizer = ASTTokenizer()
    
    # Show tokenization for each file
    cols = st.columns(min(len(file_paths), 3))
    
    for i, file_path in enumerate(file_paths[:3]):  # Show max 3 files
        with cols[i % 3]:
            file_name = Path(file_path).name
            st.subheader(f"📄 {file_name}")
            
            try:
                tokens = tokenizer.tokenize_file(file_path)
                
                # Show token statistics
                token_stats = tokenizer.get_token_statistics(tokens)
                
                st.metric("Total Tokens", len(tokens))
                st.metric("Unique Tokens", len(token_stats))
                
                # Show top tokens
                if token_stats:
                    top_tokens = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    st.write("**Top Tokens:**")
                    for token, count in top_tokens:
                        st.write(f"• {token}: {count}")
                
            except Exception as e:
                st.error(f"Error tokenizing {file_name}: {str(e)}")


def show_analysis_results(report, detector):
    """Show comprehensive analysis results"""
    st.markdown('<div class="step-header">📊 Analysis Results</div>', 
                unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Analyzed", report.total_files)
    
    with col2:
        st.metric("Total Comparisons", report.total_comparisons)
    
    with col3:
        st.metric("Plagiarism Cases", len(report.plagiarism_cases))
    
    with col4:
        st.metric("Processing Time", f"{report.processing_time:.2f}s")
    
    # Similarity statistics
    if report.statistics:
        st.subheader("📈 Similarity Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Similarity", f"{report.statistics.get('mean', 0):.4f}")
        with col2:
            st.metric("Median Similarity", f"{report.statistics.get('median', 0):.4f}")
        with col3:
            st.metric("Max Similarity", f"{report.statistics.get('max', 0):.4f}")
    
    # Similarity matrix heatmap
    st.subheader("🔥 Similarity Matrix Heatmap")
    
    if report.similarity_matrix.size > 0:
        # Create plotly heatmap
        file_names = [Path(fp).name for fp in report.file_paths]
        
        fig = px.imshow(
            report.similarity_matrix,
            x=file_names,
            y=file_names,
            color_continuous_scale='RdYlBu_r',
            title="Code Similarity Matrix"
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Files",
            yaxis_title="Files"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add threshold line visualization
        st.subheader("📏 Similarity Distribution")
        
        # Get upper triangle values (excluding diagonal)
        upper_triangle = []
        n = report.similarity_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                upper_triangle.append(report.similarity_matrix[i][j])
        
        if upper_triangle:
            fig_hist = px.histogram(
                x=upper_triangle,
                nbins=20,
                title="Distribution of Similarity Scores",
                labels={'x': 'Similarity Score', 'y': 'Frequency'}
            )
            
            # Add threshold line
            fig_hist.add_vline(
                x=report.threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {report.threshold}"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Plagiarism cases
    if report.plagiarism_cases:
        st.subheader("🚨 Detected Plagiarism Cases")
        
        for i, case in enumerate(report.plagiarism_cases, 1):
            with st.expander(f"Case #{i}: {Path(case.file1_path).name} ↔ {Path(case.file2_path).name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Similarity Score", f"{case.similarity_score:.4f}")
                    st.metric("File 1 Tokens", case.file1_tokens)
                    st.metric("Common Tokens", case.common_tokens)
                
                with col2:
                    st.metric("Classification", "PLAGIARISM" if case.is_plagiarism else "SIMILAR")
                    st.metric("File 2 Tokens", case.file2_tokens)
                    overlap_pct = (case.common_tokens / max(case.file1_tokens, case.file2_tokens)) * 100
                    st.metric("Token Overlap", f"{overlap_pct:.1f}%")
                
                # Show contributing terms if available
                if case.explanation and 'contributing_terms' in case.explanation:
                    terms = case.explanation['contributing_terms'][:5]
                    if terms:
                        st.write("**Key Contributing Terms:**")
                        term_data = []
                        for term_info in terms:
                            term_data.append({
                                'Term': term_info['term'],
                                'Contribution': f"{term_info['contribution']:.4f}",
                                'Weight 1': f"{term_info.get('tfidf1', 0):.4f}",
                                'Weight 2': f"{term_info.get('tfidf2', 0):.4f}"
                            })
                        
                        st.dataframe(pd.DataFrame(term_data), use_container_width=True)
    
    else:
        st.success("✅ No plagiarism detected above the threshold!")
    
    # File analysis
    st.subheader("📋 File Analysis")
    
    file_analysis_data = []
    for i, file_path in enumerate(report.file_paths):
        file_name = Path(file_path).name
        
        # Get feature importance if possible
        try:
            features = detector.get_feature_importance(file_path, top_k=5)
            top_features = ", ".join([f[0] for f in features[:3]])
        except:
            top_features = "N/A"
        
        file_analysis_data.append({
            'File': file_name,
            'Path': file_path,
            'Top Features': top_features
        })
    
    df_files = pd.DataFrame(file_analysis_data)
    st.dataframe(df_files, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download CSV Report"):
            csv_data = generate_csv_report(report)
            st.download_button(
                "Download CSV",
                csv_data,
                "plagiarism_report.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("📋 Download JSON Report"):
            json_data = generate_json_report(report)
            st.download_button(
                "Download JSON",
                json_data,
                "plagiarism_report.json",
                "application/json"
            )
    
    with col3:
        if st.button("📄 Download HTML Report"):
            html_data = generate_html_report(report)
            st.download_button(
                "Download HTML",
                html_data,
                "plagiarism_report.html",
                "text/html"
            )


def generate_csv_report(report):
    """Generate CSV report data"""
    csv_buffer = io.StringIO()
    
    # Write header
    csv_buffer.write("File1,File2,Similarity_Score,Is_Plagiarism,File1_Tokens,File2_Tokens,Common_Tokens\n")
    
    # Write data
    for result in report.plagiarism_cases:
        csv_buffer.write(f"{Path(result.file1_path).name},{Path(result.file2_path).name},"
                        f"{result.similarity_score:.4f},{result.is_plagiarism},"
                        f"{result.file1_tokens},{result.file2_tokens},{result.common_tokens}\n")
    
    return csv_buffer.getvalue()


def generate_json_report(report):
    """Generate JSON report data"""
    json_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_files': report.total_files,
            'total_comparisons': report.total_comparisons,
            'processing_time': report.processing_time,
            'threshold': report.threshold,
            'plagiarism_cases_count': len(report.plagiarism_cases)
        },
        'statistics': report.statistics,
        'plagiarism_cases': []
    }
    
    for result in report.plagiarism_cases:
        case_data = {
            'file1': Path(result.file1_path).name,
            'file2': Path(result.file2_path).name,
            'similarity_score': result.similarity_score,
            'is_plagiarism': result.is_plagiarism,
            'file1_tokens': result.file1_tokens,
            'file2_tokens': result.file2_tokens,
            'common_tokens': result.common_tokens
        }
        json_data['plagiarism_cases'].append(case_data)
    
    return json.dumps(json_data, indent=2)


def generate_html_report(report):
    """Generate HTML report data"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CodeGuard Plagiarism Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .case {{ background: #fff; border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
            .high {{ border-left: 5px solid #ff4444; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>CodeGuard Plagiarism Detection Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Files Analyzed:</strong> {report.total_files}</p>
            <p><strong>Plagiarism Cases:</strong> {len(report.plagiarism_cases)}</p>
        </div>
        
        <h2>Detected Cases</h2>
    """
    
    for i, result in enumerate(report.plagiarism_cases, 1):
        html_content += f"""
        <div class="case high">
            <h3>Case #{i}</h3>
            <p><strong>Files:</strong> {Path(result.file1_path).name} ↔ {Path(result.file2_path).name}</p>
            <p><strong>Similarity:</strong> {result.similarity_score:.4f}</p>
            <p><strong>Common Tokens:</strong> {result.common_tokens}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content


if __name__ == "__main__":
    main()