# Prompt untuk Claude Code: Implementasi Sistem Deteksi Kemiripan Source Code

## Konteks Penelitian
Saya sedang mengerjakan penelitian skripsi dengan judul "Implementasi Metode TF-IDF dan Cosine Similarity pada Token Abstract Syntax Tree untuk Deteksi Kemiripan Source Code". Tujuan penelitian ini adalah mencegah plagiarisme pada tugas source code program mahasiswa.

## Spesifikasi Sistem yang Dibutuhkan

### 1. Arsitektur Sistem
```
Input: Direktori source code files
↓
Parser & AST Tokenizer
↓
TF-IDF Vectorizer  
↓
Cosine Similarity Calculator
↓
Output: Similarity Matrix & Plagiarism Report
```

### 2. Komponen Utama yang Harus Diimplementasikan

#### A. AST Parser dan Tokenizer
- **Fungsi**: Mengubah source code menjadi Abstract Syntax Tree, kemudian mengekstrak token dari node-node AST
- **Input**: File source code (.py, .js, .cpp, dll)
- **Output**: String token AST yang merepresentasikan struktur kode
- **Requirements**:
  - Support multiple programming languages (minimal Python dan JavaScript)
  - Robust error handling untuk syntax errors
  - Normalisasi token (ignore whitespace, comments, variable names)

#### B. TF-IDF Vectorizer
- **Fungsi**: Mengubah kumpulan token AST menjadi vektor numerik dengan pembobotan TF-IDF
- **Formula yang harus diimplementasikan**:
  - TF (Term Frequency): `tf(t,d) = f(t,d) / Σf(t',d)`
  - IDF (Inverse Document Frequency): `idf(t) = log(N/df(t))`
  - TF-IDF: `tfidf(t,d,D) = tf(t,d) × idf(t,D)`
- **Output**: Sparse matrix representation

#### C. Cosine Similarity Calculator
- **Fungsi**: Menghitung kemiripan antar vektor TF-IDF
- **Formula**: `cos(θ) = (A·B) / (||A|| × ||B||)`
- **Output**: Similarity matrix (n×n)

#### D. Plagiarism Detector & Reporter
- **Fungsi**: Identifikasi pasangan kode yang mirip berdasarkan threshold
- **Features**:
  - Configurable similarity threshold (default: 0.7)
  - Ranking berdasarkan similarity score
  - Detailed comparison report
  - Export hasil ke CSV/JSON

### 3. Struktur Project yang Diinginkan
```
source_code_plagiarism_detector/
├── main.py                 # Entry point aplikasi
├── core/
│   ├── __init__.py
│   ├── ast_tokenizer.py    # AST parsing dan tokenization
│   ├── tfidf_vectorizer.py # TF-IDF implementation
│   ├── similarity.py       # Cosine similarity calculator
│   └── detector.py         # Main plagiarism detection logic
├── utils/
│   ├── __init__.py
│   ├── file_handler.py     # File I/O operations
│   ├── preprocessor.py     # Code preprocessing
│   └── reporter.py         # Report generation
├── tests/
│   ├── test_ast_tokenizer.py
│   ├── test_tfidf.py
│   └── test_similarity.py
├── data/
│   └── sample_codes/       # Dataset untuk testing
├── requirements.txt
└── README.md
```

### 4. Dependencies yang Harus Digunakan
```python
# Core libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# AST parsing
ast (built-in)
tree-sitter>=0.20.0
tree-sitter-python>=0.20.0
tree-sitter-javascript>=0.20.0

# Visualization & Reporting
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.6.0

# Web interface (optional)
streamlit>=1.15.0
# atau flask>=2.0.0

# Testing
pytest>=6.2.0
pytest-cov>=3.0.0
```

### 5. Fitur Khusus yang Dibutuhkan

#### A. Multi-language Support
- Python (.py)
- JavaScript (.js)
- C++ (.cpp, .cc) - optional
- Extensible untuk bahasa lain

#### B. Preprocessing Features
- Remove comments dan docstrings
- Normalize variable names ke generic tokens
- Handle different coding styles
- Skip binary files dan non-code files

#### C. Advanced Detection Features
- Detect partial plagiarism (substring matching)
- Handle code reordering (functions/classes dalam urutan berbeda)
- Ignore common boilerplate code
- Configurable sensitivity levels

#### D. Reporting & Visualization
- Similarity heatmap
- Detailed side-by-side code comparison
- Statistical analysis (mean, std deviation similarity scores)
- Export ke multiple formats (HTML, PDF, JSON)

### 6. Performance Requirements
- Mampu memproses minimal 100 file source code dalam waktu < 30 detik
- Memory efficient untuk dataset besar
- Parallel processing untuk file handling
- Progress tracking untuk long-running operations

### 7. Testing Strategy
- Unit tests untuk setiap komponen
- Integration tests untuk end-to-end workflow
- Performance benchmarks
- Test dengan dataset real dari tugas mahasiswa
- Validation terhadap ground truth plagiarism cases

## Implementation Instructions

### Phase 1: Core Implementation
1. Implement AST tokenizer with Python support
2. Build TF-IDF vectorizer from scratch (don't just use sklearn)
3. Implement cosine similarity calculator
4. Create basic plagiarism detection workflow

### Phase 2: Enhancement
1. Add multi-language support
2. Implement advanced preprocessing
3. Add web interface with Streamlit
4. Create comprehensive test suite

### Phase 3: Optimization & Reporting
1. Performance optimization
2. Advanced visualization
3. Detailed reporting system
4. Documentation and user guide

## Expected Deliverables
1. Complete working system according to specifications
2. Comprehensive test suite with coverage > 80%
3. Documentation (README, API docs, user guide)
4. Sample dataset and example usage
5. Performance benchmarks and analysis
6. Research-ready output for thesis writing

## Validation Criteria
- System must detect plagiarism with minimum 85% accuracy
- False positive rate < 15%
- Handle common plagiarism techniques (variable renaming, code reordering, style changes)
- User-friendly interface for non-technical users

Please implement this system step by step, starting with core components, then enhancements, and finally optimization. Ensure high code quality with proper error handling, logging, and documentation.

## Additional Research Context
This system is designed for academic use in Indonesian universities, specifically for detecting plagiarism in programming assignments. The implementation should be robust enough to handle various student coding patterns while maintaining high accuracy in detection.

The research methodology follows these key principles:
1. **AST-based analysis** over text-based comparison for structural understanding
2. **TF-IDF weighting** to identify significant code patterns
3. **Cosine similarity** for robust similarity measurement regardless of code length
4. **Threshold-based classification** for practical plagiarism detection

Expected research outcomes include improved plagiarism detection accuracy compared to traditional text-based methods, especially when students modify variable names, code formatting, or function ordering.
