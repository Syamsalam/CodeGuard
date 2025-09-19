"""Multi-file analysis pipeline.

Berisi fungsi reusable untuk menganalisis banyak file kode menggunakan
AST tokenizer + TF-IDF + cosine similarity, dengan dukungan weighting
dan explain mode. Dipisahkan dari `api_enhanced.py` agar dapat dipakai
oleh task GitHub maupun endpoint lain.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Callable
from collections import Counter
import random
import numpy as np
from fastapi import HTTPException
import logging

from core.ast_tokenizer import ASTTokenizer
from core.tfidf_vectorizer import TFIDFVectorizer
from core.similarity import CosineSimilarityCalculator

logger = logging.getLogger(__name__)

# Disimpan untuk endpoint statistik (jika diperlukan akses global)
last_similarity_matrix = None
last_similarity_feature_names: List[str] = []

def run_multi_file_analysis(
    file_contents: List[tuple],
    cfg: Dict,
    threshold: float,
    explain: bool,
    explain_top_k: int,
    progress_cb: Optional[Callable[[int, int], None]] = None
):
    """Execute similarity analysis across multiple files.

    Parameters
    ----------
    file_contents : List[tuple]
        List of (filename, code_str).
    cfg : Dict
        Konfigurasi token & TF-IDF + weighting dari preset atau input manual.
    threshold : float
        Ambang batas klasifikasi 'Plagiat'.
    explain : bool
        Jika True, sertakan top fitur kontribusi.
    explain_top_k : int
        Jumlah fitur teratas per pasangan ketika explain.
    """
    tokenizer = ASTTokenizer()
    tokens_list = []
    processed_names = []
    for idx_file, (name, code) in enumerate(file_contents, start=1):
        ext = name.split('.')[-1].lower()
        lang = 'python' if ext == 'py' else ('javascript' if ext in ['js', 'jsx', 'ts', 'tsx'] else 'python')
        raw_tokens = tokenizer.tokenize_code(code, lang)
        normalized_tokens = tokenizer.normalize_tokens(raw_tokens, keep_identifier_detail=cfg.get('keep_identifier_detail', False))
        # Optionally enrich with AST path n-grams (structural shingles)
        if cfg.get('enable_path_ngrams'):
            pn_min = cfg.get('path_ngram_min', 2)
            pn_max = cfg.get('path_ngram_max', 4)
            pn_hash = cfg.get('path_ngram_hash', True)
            pn_hash_len = cfg.get('path_ngram_hash_len', 8)
            path_tokens = tokenizer.generate_path_ngrams(code, lang, pn_min, pn_max, pn_hash, pn_hash_len)
            # Optional capping of path n-gram token diversity
            cap = cfg.get('path_ngram_cap', 0)
            cap_strategy = cfg.get('path_ngram_cap_strategy', 'frequency').lower()
            cap_seed = cfg.get('path_ngram_cap_seed')
            if cap and cap > 0:
                unique_tokens = list(set(path_tokens))
                original_unique = len(unique_tokens)
                selected_set = None
                if original_unique > cap:
                    if cap_strategy == 'random':
                        if cap_seed is not None:
                            random.seed(cap_seed)
                        selected_set = set(random.sample(unique_tokens, cap))
                    else:  # frequency (default)
                        counts = Counter(path_tokens)
                        # Stable deterministic ordering: by (-count, token)
                        sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                        selected_set = {tok for tok, _ in sorted_tokens[:cap]}
                    if selected_set is not None:
                        path_tokens = [t for t in path_tokens if t in selected_set]
                    cap_debug_entry = {
                        'file': name,
                        'original_unique': original_unique,
                        'cap': cap,
                        'strategy': cap_strategy,
                        'kept_unique': len(set(path_tokens)),
                        'kept_tokens': sorted(set(path_tokens))
                    }
                else:
                    cap_debug_entry = {
                        'file': name,
                        'original_unique': original_unique,
                        'cap': cap,
                        'strategy': cap_strategy,
                        'kept_unique': original_unique,
                        'kept_tokens': sorted(unique_tokens)
                    }
            else:
                cap_debug_entry = {
                    'file': name,
                    'original_unique': len(set(path_tokens)),
                    'cap': 0,
                    'strategy': None,
                    'kept_unique': len(set(path_tokens)),
                    'kept_tokens': None
                }
            # Attach path debug info to a per-run list (store on tokenizer to avoid global var clutter)
            if not hasattr(tokenizer, '_path_cap_debug'):
                tokenizer._path_cap_debug = []  # type: ignore[attr-defined]
            tokenizer._path_cap_debug.append(cap_debug_entry)  # type: ignore[attr-defined]
            # Mark them so weighting can target prefix PATH
            normalized_tokens.extend(path_tokens)
        filtered_tokens = tokenizer.filter_tokens(
            normalized_tokens,
            remove_node_tokens=cfg.get('remove_node_tokens', False),
            remove_literals=cfg.get('remove_literals', False),
            remove_operators=cfg.get('remove_operators', False),
            remove_keywords=cfg.get('remove_keywords', False),
            min_token_length=cfg.get('min_token_length', 0)
        )
        if filtered_tokens:
            tokens_list.append(filtered_tokens)
            processed_names.append(name)
        logger.info(f"[preset] Tokens for {name}: kept={len(filtered_tokens)} tokens")
        if progress_cb:
            # Laporkan jumlah file yang sudah diproses (setelah filtering) dibanding total
            progress_cb(idx_file, len(file_contents))
    if len(tokens_list) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file valid setelah filtering.")
    vectorizer = TFIDFVectorizer(
        min_df=cfg.get('tf_min_df',1),
        max_df=cfg.get('tf_max_df',1.0),
        lowercase=True,
        normalize=True,
        sublinear_tf=cfg.get('tf_sublinear_tf', False),
        use_idf=cfg.get('tf_use_idf', True),
        smooth_idf=cfg.get('tf_smooth_idf', True),
        norm=cfg.get('tf_norm','l2'),
        max_features=(cfg.get('tf_max_features') or None)
    )
    vectorizer.fit(tokens_list)
    tfidf_matrix = vectorizer.transform(tokens_list)
    feature_names = vectorizer.get_feature_names()
    # Apply weights
    structural_weight = cfg.get('structural_weight',1.0)
    var_weight = cfg.get('var_weight',1.0)
    operator_weight = cfg.get('operator_weight',1.0)
    path_weight = cfg.get('path_weight',1.0)
    keyword_weight = cfg.get('keyword_weight',1.0)
    if 0 < structural_weight < 1.0:
        structural_prefixes = (
            'NODE_', 'VAR_', 'FUNC_DEF', 'CLASS_DEF', 'OPERATOR', 'BINARY_OP', 'COMPARE', 'OP_', 'CMP_'
        )
        structural_indices = [idx for idx, term in enumerate(feature_names)
                              if any(term.startswith(p) for p in structural_prefixes)]
        if structural_indices:
            tfidf_matrix[:, structural_indices] *= structural_weight
    if 0 < var_weight < 1.0:
        var_indices = [idx for idx, term in enumerate(feature_names)
                       if term.startswith('var_use') or term.startswith('var_assign')]
        if var_indices:
            tfidf_matrix[:, var_indices] *= var_weight
    if 0 < operator_weight < 1.0:
        operator_indices = [idx for idx, term in enumerate(feature_names)
                            if term.startswith('op_') or term.startswith('cmp_') or term in ('binary_op','operator')]
        if operator_indices:
            tfidf_matrix[:, operator_indices] *= operator_weight
    if 0 < path_weight < 1.0:
        # Case-insensitive detection of generated PATH n-gram tokens (lowercased by vectorizer)
        path_indices = [idx for idx, term in enumerate(feature_names) if term.lower().startswith('path')]
        if path_indices:
            tfidf_matrix[:, path_indices] *= path_weight
    if 0 < keyword_weight < 1.0:
        # Treat *_def tokens (func_def, class_def, etc.) as keyword-style structural markers
        keyword_indices = [idx for idx, term in enumerate(feature_names) if term.lower().endswith('_def')]
        if keyword_indices:
            tfidf_matrix[:, keyword_indices] *= keyword_weight
    similarity_calc = CosineSimilarityCalculator()
    similarity_matrix = similarity_calc.calculate_similarity_matrix(tfidf_matrix)
    global last_similarity_matrix, last_similarity_feature_names
    last_similarity_matrix = similarity_matrix
    last_similarity_feature_names = feature_names
    comparisons_detail = []
    explain_details = []
    filesCount = len(tokens_list)
    similarityScores = []
    plagiarismCount = 0
    comparisons = 0
    for i in range(filesCount):
        for j in range(i+1, filesCount):
            sim = similarity_matrix[i, j]
            similarityScores.append(sim)
            status = "Plagiat" if sim >= threshold else ("Mirip" if sim >= 0.5 else "Aman")
            if sim >= threshold:
                plagiarismCount += 1
            comparisons += 1
            entry = {
                "file1": processed_names[i],
                "file2": processed_names[j],
                "similarity": sim,
                "status": status,
                "plagiarizedFragment": None
            }
            comparisons_detail.append(entry)
            if explain:
                vec_i = tfidf_matrix[i]
                vec_j = tfidf_matrix[j]
                contributions = vec_i * vec_j
                top_idx = contributions.argsort()[::-1]
                top_terms = []
                count_added = 0
                for idx in top_idx:
                    if contributions[idx] <= 0:
                        continue
                    term = feature_names[idx]
                    top_terms.append({
                        "term": term,
                        "contribution": float(contributions[idx]),
                        "weight_file1": float(vec_i[idx]),
                        "weight_file2": float(vec_j[idx])
                    })
                    count_added += 1
                    if count_added >= explain_top_k:
                        break
                explain_details.append({
                    "file1": processed_names[i],
                    "file2": processed_names[j],
                    "similarity": sim,
                    "top_features": top_terms
                })
    # Siapkan representasi matrix untuk dikirim ke frontend
    # Hati-hati terkait ukuran: untuk saat ini kirim full matrix + tfidf dibulatkan 4 desimal
    try:
        similarity_matrix_list = np.round(similarity_matrix.astype(float), 4).tolist()
        tfidf_matrix_list = np.round(tfidf_matrix.astype(float), 4).tolist()
    except Exception:
        similarity_matrix_list = similarity_matrix.tolist()
        tfidf_matrix_list = tfidf_matrix.tolist()

    # Extract accumulated path cap debug info if present
    path_cap_debug = getattr(tokenizer, '_path_cap_debug', None)
    return {
        "filesCount": filesCount,
        "comparisons": comparisons,
        "plagiarismCount": plagiarismCount,
        "comparisonsDetail": comparisons_detail,
        "similarityScores": similarityScores,
        "similarityMatrix": similarity_matrix_list,
        "fileNames": processed_names,
        "tfidf": {
            "featureNames": feature_names,
            "matrix": tfidf_matrix_list
        },
        "structuralWeightApplied": structural_weight if 0 < structural_weight < 1.0 else None,
        "variableWeightApplied": var_weight if 0 < var_weight < 1.0 else None,
        "operatorWeightApplied": operator_weight if 0 < operator_weight < 1.0 else None,
    "pathWeightApplied": path_weight if 0 < path_weight < 1.0 else None,
    "keywordWeightApplied": keyword_weight if 0 < keyword_weight < 1.0 else None,
        "explain": explain_details if explain else None,
        "tuningConfig": {
            "threshold": threshold,
            "tokenFilters": {
                "remove_node_tokens": cfg.get('remove_node_tokens', False),
                "remove_literals": cfg.get('remove_literals', False),
                "remove_operators": cfg.get('remove_operators', False),
                "remove_keywords": cfg.get('remove_keywords', False),
                "min_token_length": cfg.get('min_token_length', 0),
                "keep_identifier_detail": cfg.get('keep_identifier_detail', False)
            },
            "tfidf": {
                "min_df": cfg.get('tf_min_df',1),
                "max_df": cfg.get('tf_max_df',1.0),
                "sublinear_tf": cfg.get('tf_sublinear_tf', False),
                "use_idf": cfg.get('tf_use_idf', True),
                "smooth_idf": cfg.get('tf_smooth_idf', True),
                "norm": cfg.get('tf_norm','l2'),
                "max_features": cfg.get('tf_max_features',0)
            },
            "path_ngrams": {
                "enabled": cfg.get('enable_path_ngrams', False),
                "min": cfg.get('path_ngram_min',2),
                "max": cfg.get('path_ngram_max',4),
                "hash": cfg.get('path_ngram_hash', True),
                "hash_len": cfg.get('path_ngram_hash_len',8),
                "cap": cfg.get('path_ngram_cap',0),
                "cap_strategy": cfg.get('path_ngram_cap_strategy','frequency'),
                "cap_debug": path_cap_debug
            },
            "weights": {
                "structural": structural_weight,
                "variable": var_weight,
                "operator": operator_weight,
                "path": path_weight,
                "keyword": keyword_weight
            }
        }
    }

__all__ = [
    'run_multi_file_analysis',
    'last_similarity_matrix',
    'last_similarity_feature_names'
]
