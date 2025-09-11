import re
from typing import List, Dict
# Removed sklearn dependency for portability; implementing lightweight TF-IDF & cosine manually
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)

class EnhancedPlagiarismDetector:
    """Lightweight token-based similarity analyzer."""
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def enhanced_tokenize_code(self, code: str) -> List[str]:
        lines = []
        for ln in code.splitlines():
            if '#' in ln:
                ln = ln.split('#', 1)[0]
            ln = ln.strip()
            if ln:
                lines.append(ln)
        joined = '\n'.join(lines)
        tokens: List[str] = []
        tokens += [f"FUNC_{m}" for m in re.findall(r'def\s+(\w+)', joined)]
        tokens += [f"CLASS_{m}" for m in re.findall(r'class\s+(\w+)', joined)]
        kw_list = re.findall(r'\b(if|else|elif|for|while|return|import|from|def|class|try|except|with|as|in|and|or|not)\b', joined)
        tokens += [f"KW_{k}" for k in kw_list]
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', joined)
        reserved = {"if","else","elif","for","while","return","import","from","def","class","try","except","with","as","in","and","or","not"}
        for ident in identifiers:
            if ident in reserved:
                continue
            tokens.append(f"ID_{ident}")
        tokens += [f"NUM" for _ in re.findall(r'\b\d+(?:\.\d+)?\b', joined)]
        tokens += [f"OP_{op}" for op in re.findall(r'[+\-*/=]+', joined)]
        return tokens

    def analyze_similarity(self, code1: str, code2: str, threshold: float | None = None) -> Dict:
        if threshold is None:
            threshold = self.threshold
        t1 = self.enhanced_tokenize_code(code1)
        t2 = self.enhanced_tokenize_code(code2)
        if len(t1) < 3 or len(t2) < 3:
            return {
                'similarity_score': 0.0,
                'similarity_percentage': '0.0%',
                'is_plagiarism': False,
                'confidence': 'low',
                'level': 'INSUFFICIENT_DATA',
                'interpretation': 'Kode terlalu pendek untuk dianalisis'
            }
        doc1 = ' '.join(t1)
        doc2 = ' '.join(t2)
        # Lightweight TF-IDF implementation
        docs = [doc1.split(), doc2.split()]
        df = Counter()
        for tokens in docs:
            for term in set(tokens):
                df[term] += 1
        N = len(docs)
        tfidf_vectors = []
        vocab = set(doc1.split()) | set(doc2.split())
        for tokens in docs:
            tf = Counter(tokens)
            vec = {}
            for term in vocab:
                tf_val = tf[term]
                idf = math.log((N + 1) / (df[term] + 1)) + 1
                vec[term] = tf_val * idf
            tfidf_vectors.append(vec)
        # Cosine similarity
        v1, v2 = tfidf_vectors
        dot = sum(v1[t] * v2[t] for t in vocab)
        norm1 = math.sqrt(sum(v1[t]**2 for t in vocab)) or 1.0
        norm2 = math.sqrt(sum(v2[t]**2 for t in vocab)) or 1.0
        sim = dot / (norm1 * norm2)
        # Ensure native Python bool (avoid potential numpy.bool issues from comparisons)
        is_plag = bool(sim >= threshold)
        if sim >= 0.9:
            level, risk, interp = ('VERY_HIGH', 'CRITICAL', 'Sangat tinggi - kemungkinan besar plagiarisme')
        elif sim >= 0.7:
            level, risk, interp = ('HIGH', 'HIGH', 'Tinggi - indikasi kuat')
        elif sim >= 0.5:
            level, risk, interp = ('MEDIUM', 'MEDIUM', 'Sedang - ada kemiripan struktur')
        elif sim >= 0.3:
            level, risk, interp = ('LOW', 'LOW', 'Rendah - beberapa kemiripan minor')
        else:
            level, risk, interp = ('VERY_LOW', 'MINIMAL', 'Sangat rendah - berbeda')
        confidence = 'high' if len(t1) >= 20 and len(t2) >= 20 else 'medium' if len(t1) >= 10 and len(t2) >= 10 else 'low'
        return {
            'similarity_score': float(sim),
            'similarity_percentage': f"{sim:.1%}",
            'is_plagiarism': is_plag,
            'threshold_used': threshold,
            'confidence': confidence,
            'level': level,
            'risk_level': risk,
            'interpretation': interp,
            'analysis_details': {
                'tokens1_count': len(t1),
                'tokens2_count': len(t2),
                'token_overlap': len(set(t1) & set(t2))
            },
            'recommendations': []
        }

    def detect_plagiarism(self, files: List[tuple], threshold: float = 0.7) -> List[Dict]:
        """Compare all file pairs and return detailed results."""
        results = []
        n = len(files)
        for i in range(n):
            for j in range(i+1, n):
                file1, code1 = files[i]
                file2, code2 = files[j]
                sim_result = self.analyze_similarity(code1, code2, threshold)
                results.append({
                    'file1': file1,
                    'file2': file2,
                    'similarity': float(sim_result['similarity_score']),
                    'is_plagiarized': bool(sim_result['is_plagiarism']),
                    'plagiarizedFragment': None  # Placeholder for future fragment extraction
                })
        return results
