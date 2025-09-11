# CodeGuard - Source Code Plagiarism Detection System

A comprehensive plagiarism detection system for source code using AST tokenization, TF-IDF vectorization, and cosine similarity analysis.

## 🎯 Overview

CodeGuard implements a research-grade plagiarism detection system specifically designed for academic use in programming courses. The system uses Abstract Syntax Tree (AST) analysis combined with TF-IDF weighting and cosine similarity to detect structural similarities in source code, making it robust against common plagiarism techniques like variable renaming and code reordering.

## ✨ Key Features

- **Multi-language Support**: Python and JavaScript (extensible to other languages)
- **AST-based Analysis**: Structural code analysis rather than simple text comparison
- **Custom TF-IDF Implementation**: Built from scratch with academic research requirements
- **Advanced Similarity Metrics**: Cosine similarity with configurable thresholds
- **Comprehensive Reporting**: HTML, CSV, JSON, and visual reports
- **Batch Processing**: Analyze entire directories or specific file lists
- **Clustering**: Automatic detection of similar code clusters
- **Research-ready**: Designed for academic research and thesis work

## 🏗️ Architecture

```
Input: Source Code Files
         ↓
AST Parser & Tokenizer (Python/JavaScript)
         ↓
TF-IDF Vectorization (Custom Implementation)
         ↓
Cosine Similarity Calculation
         ↓
Plagiarism Detection & Clustering
         ↓
Output: Reports & Visualizations
```

## 📁 Project Structure

```
CodeGuard/
├── main.py                 # Main entry point
├── core/                   # Core detection components
│   ├── ast_tokenizer.py    # AST parsing and tokenization
│   ├── tfidf_vectorizer.py # Custom TF-IDF implementation
│   ├── similarity.py       # Cosine similarity calculator
│   └── detector.py         # Main plagiarism detection engine
├── utils/                  # Utility modules
│   ├── file_handler.py     # File I/O operations
│   ├── preprocessor.py     # Code preprocessing
│   └── reporter.py         # Report generation
├── tests/                  # Test suite
│   ├── test_ast_tokenizer.py
│   ├── test_tfidf.py
│   └── test_similarity.py
├── data/                   # Sample data and test cases
│   └── sample_codes/       # Sample source files
└── requirements.txt        # Dependencies
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CodeGuard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install tree-sitter languages (optional for JavaScript support):**
   ```bash
   pip install tree-sitter-python tree-sitter-javascript
   ```

## 📖 Usage

### Web Interface: GitHub Repository Plagiarism Check

You can analyze all Python files in a public GitHub repository directly from the web UI:

1. Start the FastAPI server:
    ```bash
    python api_enhanced.py
    ```
2. Open the web interface in your browser (usually at http://localhost:8000).
3. Go to the **GitHub Repo** tab.
4. Enter the public GitHub repository URL (e.g., `https://github.com/pallets/flask`).
5. Click **Mulai Analisis Repository**.
6. Wait for the analysis to complete. Results will be shown in the Results section, including a table of file pairs, similarity scores, and plagiarism status.

**Note:** Only Python files (`.py`) are currently analyzed from the repository. Support for other languages can be added as needed.

#### API Endpoint

You can also trigger analysis via API:

```http
POST /analyze/github
Form fields:
   github_url: <public GitHub repo URL>
   threshold: <similarity threshold, optional>
```

The response will include an `analysis_id`. Poll `/status/{analysis_id}` to get the results when processing is complete.

---

### Command Line Interface

**Analyze a directory of source files:**
```bash
python main.py --directory ./student_submissions --threshold 0.8 --format html
```

**Analyze specific files:**
```bash
python main.py --files file1.py file2.py file3.js --heatmap --detailed-comparisons
```

**Generate comprehensive reports:**
```bash
python main.py --directory ./code --format all --heatmap --detailed-comparisons --verbose
```

### Advanced Options

```bash
python main.py --help
```

**Key parameters:**
- `--threshold`: Similarity threshold for plagiarism detection (default: 0.7)
- `--min-tokens`: Minimum tokens required for analysis (default: 10)
- `--extensions`: File extensions to analyze (default: .py .js)
- `--format`: Report format (html, csv, json, all)
- `--heatmap`: Generate similarity heatmap visualization
- `--detailed-comparisons`: Generate detailed comparison reports

### Python API

```python
from core.detector import PlagiarismDetector

# Initialize detector
detector = PlagiarismDetector(similarity_threshold=0.7)

# Analyze directory
report = detector.detect_in_directory("./source_files")

# Generate reports
from utils.reporter import ReportGenerator
reporter = ReportGenerator()
reporter.generate_html_report(report, "report.html")
reporter.print_console_report(report)
```

## 🔬 Research Methodology

This system implements the following research methodology:

1. **AST Tokenization**: Converts source code to structural tokens using Abstract Syntax Trees
2. **TF-IDF Weighting**: Applies Term Frequency-Inverse Document Frequency weighting to identify significant patterns
3. **Cosine Similarity**: Measures similarity between TF-IDF vectors using cosine similarity
4. **Threshold Classification**: Classifies similarities above threshold as potential plagiarism

### Formula Implementation

**Term Frequency (TF):**
```
tf(t,d) = count(t,d) / |d|
```

**Inverse Document Frequency (IDF):**
```
idf(t) = log(N / df(t))
```

**TF-IDF Weight:**
```
tfidf(t,d,D) = tf(t,d) × idf(t,D)
```

**Cosine Similarity:**
```
cos(θ) = (A·B) / (||A|| × ||B||)
```

## 📊 Report Types

### 1. Console Report
Real-time analysis results with summary statistics

### 2. HTML Report
Comprehensive web-based report with:
- Executive summary
- Similarity statistics
- Detailed case analysis
- File listings

### 3. CSV Export
Machine-readable format for further analysis:
- File pairs and similarity scores
- Token counts and metadata
- Classification results

### 4. JSON Report
Structured data format including:
- Complete analysis metadata
- Detailed explanations
- Contributing term analysis

### 5. Similarity Heatmap
Visual representation of pairwise similarities

### 6. Detailed Comparisons
Side-by-side analysis of high-similarity cases

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ast_tokenizer.py -v

# Run with coverage
pytest tests/ --cov=core --cov=utils --cov-report=html
```

## ⚙️ Configuration

### Detection Parameters
- **similarity_threshold**: Minimum similarity to classify as plagiarism (0.0-1.0)
- **min_tokens**: Minimum token count for file inclusion
- **min_df**: Minimum document frequency for TF-IDF terms
- **max_df**: Maximum document frequency for TF-IDF terms

### File Filtering
- **extensions**: Supported file extensions
- **ignore_patterns**: Patterns to exclude (node_modules, __pycache__, etc.)
- **recursive**: Enable/disable recursive directory search

## 📈 Performance

**Benchmarks:**
- Processes 100 Python files (avg. 50 lines) in ~15 seconds
- Memory efficient with sparse matrix optimizations
- Scales linearly with number of documents
- Supports parallel processing for large datasets

## 🎓 Academic Use

This system is specifically designed for:
- **Course Assignment Analysis**: Detect plagiarism in programming assignments
- **Research Studies**: Academic research on code similarity
- **Thesis Projects**: Implementation studies on plagiarism detection
- **Educational Tools**: Teaching about code analysis and similarity metrics

## 🔍 Detection Capabilities

**Robust against common plagiarism techniques:**
- ✅ Variable and function renaming
- ✅ Code reordering and restructuring
- ✅ Comment and whitespace changes
- ✅ Style and formatting variations
- ✅ Partial code copying
- ✅ Logic preservation with syntax changes

**Limitations:**
- ❌ Algorithm logic changes
- ❌ Complete rewrites with different approaches
- ❌ Cross-language translations

## 📋 Requirements

**System Requirements:**
- Python 3.7+
- 4GB+ RAM (for large datasets)
- Modern CPU (multi-core recommended)

**Python Dependencies:**
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- tree-sitter >= 0.20.0 (optional)
- pytest >= 6.2.0 (testing)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 📚 Citation

If you use this system in academic research, please cite:

```bibtex
@software{codeguard2024,
  title={CodeGuard: Source Code Plagiarism Detection using AST-based TF-IDF Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/codeguard}
}
```

## 📞 Support

For questions, issues, or contributions:
- 📧 Email: your.email@example.com
- 🐛 Issues: Create GitHub issue
- 💬 Discussions: GitHub Discussions tab

---

**Built for academic integrity and educational excellence** 🎓
