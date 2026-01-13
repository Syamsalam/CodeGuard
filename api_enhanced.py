#!/usr/bin/env python3
"""
Enhanced FastAPI server with detailed plagiarism reporting
"""
# =============================
# Imports & Global Config
# =============================
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import re
import tempfile
import os
import asyncio
import uuid
import requests
import zipfile
import io
from typing import List, Dict, Optional
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
# (sklearn imports dihapus karena tidak dipakai lagi setelah refactor modular)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeGuard Enhanced Plagiarism Detection",
    version="2.0",
    description="API untuk deteksi plagiarisme kode sumber menggunakan AST, TF-IDF, dan Cosine Similarity. Cocok untuk integrasi website lain."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""Refactored: Single SETTINGS configuration & multi-file pipeline dipindah ke modul core.*"""

# =============================
# Global State (API-level)
# =============================
analysis_results = {}

# Import konfigurasi & pipeline eksternal
from core.preset_config import get_settings, get_threshold
from core.multi_file_analysis import run_multi_file_analysis, last_similarity_matrix, last_similarity_feature_names
from core.github_task import run_github_analysis_background
from core.github_task import run_github_account_crossrepo_background
from core.github_task import run_github_crossrepo_urls_background
from core.github_task import run_github_pair_detail_background
from core.cross_project_task import run_cross_project_analysis_background

###############################################
# (Pipeline helper dipindah ke core.multi_file_analysis)
###############################################

###############################################
# Endpoint: Preset listing
###############################################
@app.get("/analyze/settings", summary="Daftar pengaturan konfigurasi")
async def list_settings():
    settings = get_settings()
    return {
        "settings": {
            "description": "Single unified configuration for plagiarism detection",
            "default_threshold": settings['threshold'],
            "params": settings
        }
    }

###############################################
# Endpoint: Compare with preset
###############################################
@app.post("/analyze/compare", summary="Bandingkan banyak file dengan pengaturan")
async def compare_files(
    threshold: Optional[float] = Form(None, description="Override threshold (opsional)"),
    explain: bool = Form(False),
    explain_top_k: int = Form(5),
    files: List[UploadFile] = File(...),
    tolerate_invalid_python: bool = Form(False, description="Jika True, lanjut meski ada file .py tidak valid (fallback).")
):
    settings = get_settings()
    th = threshold if threshold is not None else settings['threshold']
    file_contents = []
    for f in files:
        code = (await f.read()).decode(errors='ignore')
        if code.strip():
            file_contents.append((f.filename, code))
    if len(file_contents) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
    # Validate Python files syntax
    invalid_py = []
    tokenizer = ASTTokenizer()
    for name, code in file_contents:
        if name.lower().endswith('.py') and not tokenizer.is_valid_python_syntax(code):
            invalid_py.append(name)
    if invalid_py and not tolerate_invalid_python:
        raise HTTPException(status_code=400, detail={"message": "Ditemukan file Python tidak valid secara sintaks.", "files": invalid_py, "hint": "Perbaiki indentasi/struktur blok sebelum analisis."})
    logger.info(f"Applying settings threshold={th} params={settings}")
    result = run_multi_file_analysis(file_contents, settings, th, explain, explain_top_k)
    if invalid_py:
        result['warnings'] = [f"File Python tidak valid: {', '.join(invalid_py)} (analisis dilanjutkan dengan fallback)"]
    result['settings_applied'] = settings
    return result

###############################################
# Endpoint: Main page
###############################################
@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Serve external index.html UI"""
    return FileResponse("index.html")

# -------------------------------------------------
# CLI entrypoint (jalankan langsung: python api_enhanced.py)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_enhanced:app", host="0.0.0.0", port=8000, reload=False)

###############################################
# Endpoint: GitHub analysis & background task
###############################################
from core.github_analyzer import GitHubRepositoryAnalyzer  # still used for direct instantiation if needed
github_analyzer = GitHubRepositoryAnalyzer()  # retained for backward compatibility (status listing, etc.)

@app.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis ID tidak ditemukan")
    return analysis_results[analysis_id]

@app.post("/analyze/github", summary="Analisis repository GitHub")
async def analyze_github_repository(
    github_url: str = Form(..., description="URL repository GitHub"),
    threshold: Optional[float] = Form(None, description="Ambang batas similarity (default dari settings)"),
    explain: bool = Form(False, description="Aktifkan explain mode"),
    explain_top_k: int = Form(5, description="Top fitur saat explain"),
    skip_matrices: bool = Form(False, description="Jika True, sembunyikan similarityMatrix & tfidf dari hasil untuk mengurangi payload besar")
):
    if not github_url.startswith('https://github.com/'):
        raise HTTPException(status_code=400, detail="URL harus berupa repository GitHub yang valid")
    settings = get_settings()
    th = threshold if threshold is not None else get_threshold()
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = {
        'status': 'processing',
        'started_at': datetime.now().isoformat(),
        'repository_url': github_url,
        'type': 'github_repository',
        'settings_applied': settings,
        'threshold': th
    }
    # Jalankan task background via modul eksternal
    from core.github_task import run_github_analysis_background  # local import to avoid circular (if any)
    asyncio.create_task(run_github_analysis_background(analysis_results, analysis_id, github_url, th, settings, explain, explain_top_k, skip_matrices))
    return {
        'analysis_id': analysis_id,
        'status': 'started',
        'message': 'Analisis repository GitHub dimulai',
        'repository_url': github_url,
        'settings_applied': settings,
        'skip_matrices': skip_matrices
    }

## Background GitHub task dipindah -> core.github_task.run_github_analysis_background

###############################################
# Endpoint: Manual analysis
###############################################
from core.ast_tokenizer import ASTTokenizer
from core.tfidf_vectorizer import TFIDFVectorizer
from core.similarity import CosineSimilarityCalculator

from pydantic import BaseModel, Field

class ManualAnalysisRequest(BaseModel):
    code1: str = Field(..., description="Kode sumber pertama")
    code2: str = Field(..., description="Kode sumber kedua")
    lang1: str = Field('python', description="Bahasa kode pertama (python/javascript)")
    lang2: str = Field('python', description="Bahasa kode kedua (python/javascript)")
    threshold: Optional[float] = Field(None, description="Ambang batas kemiripan (default dari settings)")
    min_tokens: int = Field(1, description="Minimum token yang digunakan dalam analisis")
    explain: bool = Field(False, description="Aktifkan explain mode untuk melihat top fitur.")
    explain_top_k: int = Field(5, description="Jumlah fitur teratas saat explain mode.")
    tolerate_invalid_python: bool = Field(False, description="Jika True, lanjutkan analisis meskipun sintaks Python tidak valid (gunakan fallback).")

@app.post("/analyze/manual", summary="Analisis manual dua kode", response_description="Hasil analisis kemiripan dua kode")
async def analyze_manual_code(payload: ManualAnalysisRequest = Body(...)):
    """Analisis dua kode sumber secara manual menggunakan settings default."""
    try:
        if not payload.code1.strip() or not payload.code2.strip():
            return {"error": "Kode tidak boleh kosong."}
        threshold = payload.threshold if payload.threshold is not None else get_threshold()
        warnings = []
        tokenizer = ASTTokenizer()
        # Validasi sintaks Python sebelum analisis
        if payload.lang1 == 'python' and not tokenizer.is_valid_python_syntax(payload.code1):
            msg = "Kode manual1.py tidak valid sebagai Python (periksa indentasi/struktur)."
            if not payload.tolerate_invalid_python:
                raise HTTPException(status_code=400, detail={"message": msg, "hint": "Perbaiki indentasi blok setelah def/if/for, dsb.", "file": "manual1.py"})
            warnings.append(msg)
        if payload.lang2 == 'python' and not tokenizer.is_valid_python_syntax(payload.code2):
            msg = "Kode manual2.py tidak valid sebagai Python (periksa indentasi/struktur)."
            if not payload.tolerate_invalid_python:
                raise HTTPException(status_code=400, detail={"message": msg, "hint": "Perbaiki indentasi blok setelah def/if/for, dsb.", "file": "manual2.py"})
            warnings.append(msg)
        # Gunakan pipeline lengkap reuse helper
        settings = get_settings()
        # Bangun pseudo filename agar deteksi bahasa sesuai
        def _ext_for_lang(lang: str) -> str:
            return 'py' if lang=='python' else ('ts' if lang=='typescript' else 'js')
        fname1 = f"manual1.{ _ext_for_lang(payload.lang1) }"
        fname2 = f"manual2.{ _ext_for_lang(payload.lang2) }"
        result = run_multi_file_analysis(
            [(fname1, payload.code1), (fname2, payload.code2)],
            settings,
            threshold,
            payload.explain,
            payload.explain_top_k
        )
        result['settings_applied'] = settings
        if warnings:
            result['warnings'] = warnings
        return result
    except Exception as e:
        logger.error(f"Error in manual code analysis: {str(e)}")
        return {"error": str(e), "detail": "Gagal menganalisis kode. Pastikan input sudah benar dan kode tidak kosong."}
###############################################
# Endpoint: Multi-file compare
###############################################
from fastapi import UploadFile
from typing import List

@app.post("/analyze/compare_detailed", summary="Bandingkan banyak file dengan parameter detail", response_description="Hasil analisis kemiripan multi-file")
async def compare_multiple_files_detailed(
    files: List[UploadFile] = File(..., description="File kode sumber yang diupload"),
    threshold: Optional[float] = Form(None, description="Ambang batas kemiripan (default dari settings)"),
    min_tokens: int = Form(1, description="Minimum token yang digunakan dalam analisis"),
    # Token filtering options
    remove_node_tokens: bool = Form(False, description="Hilangkan token tipe NODE_"),
    remove_literals: bool = Form(False, description="Hilangkan token literal (STR/NUM/BOOL/LITERAL)"),
    remove_operators: bool = Form(False, description="Hilangkan token operator (OP_, CMP_, OPERATOR)"),
    remove_keywords: bool = Form(False, description="Hilangkan keyword Python"),
    min_token_length: int = Form(0, description="Minimum panjang token yang dipertahankan"),
    keep_identifier_detail: bool = Form(False, description="Pertahankan detail identifier & operator (kurangi normalisasi agresif)"),
    # TF-IDF tuning
    tf_min_df: int = Form(1, description="Minimum document frequency"),
    tf_max_df: float = Form(1.0, description="Maximum document frequency (fraction atau absolute)"),
    tf_sublinear_tf: bool = Form(False, description="Gunakan sublinear tf (1+log(tf))"),
    tf_use_idf: bool = Form(True, description="Gunakan bobot IDF"),
    tf_smooth_idf: bool = Form(True, description="Gunakan smoothing IDF"),
    tf_norm: str = Form('l2', description="Normalisasi vektor (l2/none)"),
    tf_max_features: int = Form(0, description="Batasi jumlah fitur TF-IDF (0 = tidak dibatasi)"),
    explain: bool = Form(False, description="Aktifkan explain mode (top kontribusi fitur per pasangan)"),
    explain_top_k: int = Form(5, description="Jumlah fitur top per pasangan ketika explain mode"),
    structural_weight: float = Form(1.0, description="Faktor pengurang bobot token struktural (0-1, misal 0.3)"),
    var_weight: float = Form(1.0, description="Faktor bobot khusus untuk token variabel VAR_USE / VAR_ASSIGN (0-1)"),
    operator_weight: float = Form(1.0, description="Faktor bobot untuk token operator (OP_, CMP_, BINARY_OP, OPERATOR) (0-1)"),
    tolerate_invalid_python: bool = Form(False, description="Jika True, lanjut meski ada file .py tidak valid (fallback).")
):
    """
    Bandingkan banyak file kode sumber untuk deteksi plagiarisme dengan parameter detail.
    Cocok untuk integrasi API website lain.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file diperlukan untuk analisis.")
    # Gunakan settings default dengan override parameter yang ada
    settings = get_settings()
    th = threshold if threshold is not None else get_threshold()
    file_contents = []
    for f in files:
        code = (await f.read()).decode(errors='ignore')
        if code.strip():
            file_contents.append((f.filename, code))
    if len(file_contents) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
    # Validate Python syntax
    invalid_py = []
    tokenizer = ASTTokenizer()
    for name, code in file_contents:
        if name.lower().endswith('.py') and not tokenizer.is_valid_python_syntax(code):
            invalid_py.append(name)
    if invalid_py and not tolerate_invalid_python:
        raise HTTPException(status_code=400, detail={"message": "Ditemukan file Python tidak valid secara sintaks.", "files": invalid_py, "hint": "Perbaiki indentasi/struktur blok sebelum analisis."})
    
    # Override settings dengan parameter yang diberikan
    custom_settings = settings.copy()
    custom_settings.update({
        'min_tokens': min_tokens,
        'remove_node_tokens': remove_node_tokens,
        'remove_literals': remove_literals,
        'remove_operators': remove_operators,
        'remove_keywords': remove_keywords,
        'min_token_length': min_token_length,
        'keep_identifier_detail': keep_identifier_detail,
        'tf_min_df': tf_min_df,
        'tf_max_df': tf_max_df,
        'tf_sublinear_tf': tf_sublinear_tf,
        'tf_use_idf': tf_use_idf,
        'tf_smooth_idf': tf_smooth_idf,
        'tf_norm': tf_norm,
        'tf_max_features': tf_max_features,
        'structural_weight': structural_weight,
        'var_weight': var_weight,
        'operator_weight': operator_weight
    })
    
    result = run_multi_file_analysis(file_contents, custom_settings, th, explain, explain_top_k)
    if invalid_py:
        result['warnings'] = [f"File Python tidak valid: {', '.join(invalid_py)} (analisis dilanjutkan dengan fallback)"]
    result['settings_applied'] = custom_settings
    return result

###############################################
# Endpoint: GitHub analisis cross repositories  
###############################################
    tokens_list = []
    processed_names = []
    for name, code in file_contents:
        ext = name.split('.')[-1].lower()
        lang = 'python' if ext == 'py' else ('typescript' if ext in ['ts','tsx'] else ('javascript' if ext in ['js','jsx'] else 'python'))
        langs.append(lang)
        raw_tokens = tokenizer.tokenize_code(code, lang)
        normalized_tokens = tokenizer.normalize_tokens(raw_tokens, keep_identifier_detail=keep_identifier_detail)
        filtered_tokens = tokenizer.filter_tokens(
            normalized_tokens,
            remove_node_tokens=remove_node_tokens,
            remove_literals=remove_literals,
            remove_operators=remove_operators,
            remove_keywords=remove_keywords,
            min_token_length=min_token_length
        )
        tokens = filtered_tokens
        logger.info(f"Tokens for {name} (raw={len(raw_tokens)} normalized={len(normalized_tokens)} kept={len(tokens)}): {tokens}")
        if len(tokens) >= 1:
            tokens_list.append(tokens)
            processed_names.append(name)
        else:
            logger.warning(f"File {name} jumlah token kurang dari minimum setelah tokenisasi (min 1).")
    logger.info(f"Tokens list: {tokens_list}")
    # Handle max_features=0 -> None
    max_features = tf_max_features if tf_max_features > 0 else None
    vectorizer = TFIDFVectorizer(
        min_df=tf_min_df,
        max_df=tf_max_df,
        lowercase=True,
        normalize=True,
        sublinear_tf=tf_sublinear_tf,
        use_idf=tf_use_idf,
        smooth_idf=tf_smooth_idf,
        norm=tf_norm,
        max_features=max_features
    )
    logger.info("Mulai proses TF-IDF fit...")
    vectorizer.fit(tokens_list)
    logger.info(f"TFIDF selesai fit. Jumlah dokumen: {len(tokens_list)}")
    logger.info(f"TFIDF Vocabulary: {vectorizer.get_vocabulary()}")
    logger.info("Mulai proses TF-IDF transform...")
    tfidf_matrix = vectorizer.transform(tokens_list)
    feature_names = vectorizer.get_feature_names()
    if 0 < structural_weight < 1.0:
        structural_prefixes = (
            'NODE_', 'VAR_', 'FUNC_DEF', 'CLASS_DEF', 'OPERATOR', 'BINARY_OP', 'COMPARE', 'OP_', 'CMP_'
        )
        structural_indices = [idx for idx, term in enumerate(feature_names)
                              if any(term.startswith(p) for p in structural_prefixes)]
        if structural_indices:
            tfidf_matrix[:, structural_indices] *= structural_weight
    if 0 < var_weight < 1.0:
        var_indices = [idx for idx, term in enumerate(feature_names) if term.startswith('var_use') or term.startswith('var_assign')]
        if var_indices:
            tfidf_matrix[:, var_indices] *= var_weight
    if 0 < operator_weight < 1.0:
        operator_indices = [idx for idx, term in enumerate(feature_names)
                            if term.startswith('op_') or term.startswith('cmp_') or term in ('binary_op', 'operator')]
        if operator_indices:
            tfidf_matrix[:, operator_indices] *= operator_weight
    similarity_calc = CosineSimilarityCalculator()
    similarity_matrix = similarity_calc.calculate_similarity_matrix(tfidf_matrix)
    global last_similarity_matrix, last_similarity_feature_names
    last_similarity_matrix = similarity_matrix
    last_similarity_feature_names = feature_names
    logger.info(f"Similarity Matrix: {similarity_matrix}")
    # Buat hasil detail
    comparisons_detail = []
    explain_details = []
    filesCount = len(tokens_list)
    comparisons = 0
    plagiarismCount = 0
    similarityScores = []
    for i in range(filesCount):
        for j in range(i+1, filesCount):
            sim = similarity_matrix[i, j]
            similarityScores.append(sim)
            status = "Plagiat" if sim >= threshold else ("Mirip" if sim >= 0.5 else "Aman")
            if sim >= threshold:
                plagiarismCount += 1
            comparisons += 1
            pair_entry = {
                "file1": file_contents[i][0],
                "file2": file_contents[j][0],
                "similarity": sim,
                "status": status,
                "plagiarizedFragment": None
            }
            comparisons_detail.append(pair_entry)
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
                    term = last_similarity_feature_names[idx]
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
                    "file1": file_contents[i][0],
                    "file2": file_contents[j][0],
                    "similarity": sim,
                    "top_features": top_terms
                })
    # Prepare matrices and token payloads for UI
    try:
        similarity_matrix_list = similarity_matrix.astype(float).tolist()
        tfidf_matrix_list = tfidf_matrix.astype(float).tolist()
    except Exception:
        similarity_matrix_list = similarity_matrix.tolist()
        tfidf_matrix_list = tfidf_matrix.tolist()

    result = {
        "filesCount": filesCount,
        "comparisons": comparisons,
        "plagiarismCount": plagiarismCount,
        "processingTime": "0.1",
        "comparisonsDetail": comparisons_detail,
        "similarityScores": similarityScores,
        "similarityMatrix": similarity_matrix_list,
        "fileNames": processed_names,
        "tfidf": {
            "featureNames": feature_names,
            "matrix": tfidf_matrix_list
        },
        "astTokens": {
            "files": [
                {"file": processed_names[i], "tokens": tokens_list[i]} for i in range(len(processed_names))
            ]
        },
        "tuningConfig": {
            "threshold": threshold,
            "tokenFilters": {
                "remove_node_tokens": remove_node_tokens,
                "remove_literals": remove_literals,
                "remove_operators": remove_operators,
                "remove_keywords": remove_keywords,
                "min_token_length": min_token_length,
                "keep_identifier_detail": keep_identifier_detail
            },
            "tfidf": {
                "min_df": tf_min_df,
                "max_df": tf_max_df,
                "sublinear_tf": tf_sublinear_tf,
                "use_idf": tf_use_idf,
                "smooth_idf": tf_smooth_idf,
                "norm": tf_norm,
                "max_features": tf_max_features
            }
        },
        "structuralWeightApplied": structural_weight if 0 < structural_weight < 1.0 else None,
        "variableWeightApplied": var_weight if 0 < var_weight < 1.0 else None,
        "operatorWeightApplied": operator_weight if 0 < operator_weight < 1.0 else None,
        "explain": explain_details if explain else None,
        "warnings": [f"File Python tidak valid: {', '.join(invalid_py)} (analisis dilanjutkan dengan fallback)"] if invalid_py else None
    }
    return result

###############################################
# Endpoint: GitHub account cross-repo comparison
###############################################
@app.post("/analyze/github/account", summary="Bandingkan banyak repository dalam satu akun (cross-repo)")
async def analyze_github_account_crossrepo(
    username: str = Form(..., description="Username atau organisasi GitHub"),
    threshold: Optional[float] = Form(None, description="Ambang batas similarity (default dari settings)"),
    include_forks: bool = Form(False, description="Sertakan fork repos"),
    max_repos: int = Form(5, description="Batas jumlah repo yang dianalisis"),
    token: Optional[str] = Form(None, description="GitHub token (opsional) untuk akses rate limit lebih longgar / private repos"),
    explain: bool = Form(False, description="Aktifkan explain mode (tidak digunakan untuk agregasi repo)"),
    explain_top_k: int = Form(5, description="Top fitur saat explain (tidak digunakan untuk agregasi repo)"),
    skip_matrices: bool = Form(True, description="Sembunyikan matriks berat pada hasil")
):
    settings = get_settings()
    th = threshold if threshold is not None else get_threshold()
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = {
        'status': 'processing',
        'started_at': datetime.now().isoformat(),
        'type': 'github_account_crossrepo',
        'account': username,
        'settings_applied': settings,
        'threshold': th
    }
    asyncio.create_task(run_github_account_crossrepo_background(
        analysis_results,
        analysis_id,
        username,
        th,
        settings,
        explain,
        explain_top_k,
        skip_matrices,
        include_forks,
        max_repos,
        token
    ))
    return {
        'analysis_id': analysis_id,
        'status': 'started',
        'message': 'Analisis cross-repo akun GitHub dimulai',
        'account': username,
        'settings_applied': settings,
        'include_forks': include_forks,
        'max_repos': max_repos,
        'skip_matrices': skip_matrices
    }

###############################################
# Endpoint: Cross-Repository by URLs
###############################################
@app.post("/analyze/github/cross_urls", summary="Bandingkan beberapa repository berdasarkan daftar URL")
async def analyze_github_cross_urls(
    repo_urls: str = Form(..., description="Daftar URL GitHub, satu per baris"),
    threshold: Optional[float] = Form(None, description="Ambang batas similarity (default dari settings)"),
    explain: bool = Form(False, description="Aktifkan explain mode (tidak digunakan untuk agregasi repo)"),
    explain_top_k: int = Form(5, description="Top fitur saat explain (tidak digunakan untuk agregasi repo)"),
    skip_matrices: bool = Form(True, description="Sembunyikan matriks berat pada hasil")
):
    settings = get_settings()
    th = threshold if threshold is not None else get_threshold()
    urls = [u.strip() for u in repo_urls.splitlines() if u.strip()]
    if len(urls) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 URL repository diperlukan")
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = {
        'status': 'processing',
        'started_at': datetime.now().isoformat(),
        'type': 'github_crossrepo_urls',
        'settings_applied': settings,
        'threshold': th
    }
    asyncio.create_task(run_github_crossrepo_urls_background(
        analysis_results,
        analysis_id,
        urls,
        th,
        settings,
        explain,
        explain_top_k,
        skip_matrices
    ))
    return {
        'analysis_id': analysis_id,
        'status': 'started',
        'message': 'Analisis cross-repo (URLs) dimulai',
        'repo_urls': urls,
        'settings_applied': settings,
        'skip_matrices': skip_matrices
    }

###############################################
# Endpoint: Repo pair detail (drill-down)
###############################################
@app.post("/analyze/github/pair_detail", summary="Analisis detail dua repository (matriks & token)")
async def analyze_github_pair_detail(
    repo_a_url: str = Form(..., description="URL repo GitHub A"),
    repo_b_url: str = Form(..., description="URL repo GitHub B"),
    threshold: Optional[float] = Form(None, description="Ambang batas similarity (default dari settings)"),
    explain: bool = Form(False, description="Aktifkan explain mode"),
    explain_top_k: int = Form(5, description="Top fitur saat explain"),
    skip_matrices: bool = Form(False, description="Sembunyikan matriks pada hasil")
):
    if not (repo_a_url and repo_b_url and repo_a_url.startswith('https://github.com/') and repo_b_url.startswith('https://github.com/')):
        raise HTTPException(status_code=400, detail="URL harus berupa repository GitHub yang valid")
    settings = get_settings()
    th = threshold if threshold is not None else get_threshold()
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = {
        'status': 'processing',
        'started_at': datetime.now().isoformat(),
        'type': 'github_pair_detail',
        'repo_a_url': repo_a_url,
        'repo_b_url': repo_b_url,
        'settings_applied': settings,
        'threshold': th
    }
    from core.github_task import run_github_pair_detail_background
    asyncio.create_task(run_github_pair_detail_background(
        analysis_results,
        analysis_id,
        repo_a_url,
        repo_b_url,
        th,
        settings,
        explain,
        explain_top_k,
        skip_matrices
    ))
    return {
        'analysis_id': analysis_id,
        'status': 'started',
        'message': 'Analisis detail pasangan repository dimulai',
        'repo_a_url': repo_a_url,
        'repo_b_url': repo_b_url,
        'skip_matrices': skip_matrices
    }

###############################################
# Endpoint: Cross-Project Manual Upload
###############################################
@app.post("/analyze/cross_project", summary="Analisis cross-project dari upload manual")
async def analyze_cross_project_manual(
    files: List[UploadFile] = File(..., description="File kode sumber yang diupload (dengan struktur folder)"),
    threshold: Optional[float] = Form(None, description="Ambang batas kemiripan (default dari settings)"),
    explain: bool = Form(False, description="Aktifkan explain mode"),
    explain_top_k: int = Form(5, description="Top fitur saat explain"),
    skip_matrices: bool = Form(False, description="Sembunyikan matriks berat pada hasil"),
    tolerate_invalid_python: bool = Form(False, description="Jika True, lanjut meski ada file .py tidak valid")
):
    """
    Analisis cross-project dari upload manual.
    Files akan digroup berdasarkan struktur folder (project1/file.py, project2/file.py)
    dan dibandingkan seperti GitHub cross-repo analysis.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file diperlukan untuk analisis.")
    
    settings = get_settings()
    th = threshold if threshold is not None else get_threshold()
    
    # Prepare file data
    file_contents = []
    for f in files:
        try:
            content = (await f.read()).decode(errors='ignore')
            if content.strip():
                # Use the full filename (including folder path) for project grouping
                filename = f.filename or f"unknown_file_{len(file_contents)}"
                file_contents.append((filename, content))
        except Exception as e:
            logger.warning(f"Error reading file {f.filename}: {e}")
    
    if len(file_contents) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
    
    # Validate Python syntax
    invalid_py = []
    if not tolerate_invalid_python:
        tokenizer = ASTTokenizer()
        for name, code in file_contents:
            if name.lower().endswith('.py') and not tokenizer.is_valid_python_syntax(code):
                invalid_py.append(name)
        
        if invalid_py:
            raise HTTPException(status_code=400, detail={
                "message": "Ditemukan file Python tidak valid secara sintaks.",
                "files": invalid_py,
                "hint": "Perbaiki indentasi/struktur blok atau aktifkan tolerate_invalid_python."
            })
    
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = {
        'status': 'processing',
        'started_at': datetime.now().isoformat(),
        'type': 'cross_project_manual',
        'total_files': len(file_contents),
        'settings_applied': settings,
        'threshold': th
    }
    
    # Start background analysis
    asyncio.create_task(run_cross_project_analysis_background(
        analysis_results,
        analysis_id,
        file_contents,
        th,
        settings,
        explain,
        explain_top_k,
        skip_matrices
    ))
    
    return {
        'analysis_id': analysis_id,
        'status': 'started',
        'message': 'Analisis cross-project manual dimulai',
        'total_files': len(file_contents),
        'settings_applied': settings,
        'skip_matrices': skip_matrices
    }
 