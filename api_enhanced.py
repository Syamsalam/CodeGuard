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

"""Refactored: PRESETS & multi-file pipeline dipindah ke modul core.*"""

# =============================
# Global State (API-level)
# =============================
analysis_results = {}

# Import konfigurasi & pipeline eksternal
from core.preset_config import PRESETS
from core.multi_file_analysis import run_multi_file_analysis, last_similarity_matrix, last_similarity_feature_names
from core.github_task import run_github_analysis_background

###############################################
# (Pipeline helper dipindah ke core.multi_file_analysis)
###############################################

###############################################
# Endpoint: Preset listing
###############################################
@app.get("/analyze/presets", summary="Daftar preset konfigurasi")
async def list_presets():
    return {"presets": {name: {"description": data['description'], "default_threshold": data['default_threshold'], "params": data['params']} for name, data in PRESETS.items()}}

###############################################
# Endpoint: Compare with preset
###############################################
@app.post("/analyze/compare_preset", summary="Bandingkan banyak file dengan preset")
async def compare_with_preset(
    preset: str = Form(..., description="Nama preset: strict | balanced | permissive"),
    threshold: Optional[float] = Form(None, description="Override threshold (opsional)"),
    explain: bool = Form(False),
    explain_top_k: int = Form(5),
    files: List[UploadFile] = File(...)
):
    if preset not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset}' tidak ditemukan")
    preset_cfg = PRESETS[preset]
    th = threshold if threshold is not None else preset_cfg['default_threshold']
    file_contents = []
    for f in files:
        code = (await f.read()).decode(errors='ignore')
        if code.strip():
            file_contents.append((f.filename, code))
    if len(file_contents) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
    logger.info(f"Applying preset '{preset}' threshold={th} params={preset_cfg['params']}")
    result = run_multi_file_analysis(file_contents, preset_cfg['params'], th, explain, explain_top_k)
    result['preset'] = preset
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

@app.post("/analyze/github", summary="Analisis repository GitHub (mendukung preset)")
async def analyze_github_repository(
    github_url: str = Form(..., description="URL repository GitHub"),
    threshold: float = Form(0.7, description="Ambang batas similarity"),
    preset: Optional[str] = Form('strict', description="Preset (strict|balanced|permissive)"),
    explain: bool = Form(False, description="Aktifkan explain mode"),
    explain_top_k: int = Form(5, description="Top fitur saat explain"),
    skip_matrices: bool = Form(False, description="Jika True, sembunyikan similarityMatrix & tfidf dari hasil untuk mengurangi payload besar")
):
    if not github_url.startswith('https://github.com/'):
        raise HTTPException(status_code=400, detail="URL harus berupa repository GitHub yang valid")
    if preset and preset not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset}' tidak ditemukan")
    cfg = PRESETS[preset]['params'] if preset else None
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = {
        'status': 'processing',
        'started_at': datetime.now().isoformat(),
        'repository_url': github_url,
        'type': 'github_repository',
        'preset': preset,
        'threshold': threshold
    }
    # Jalankan task background via modul eksternal
    from core.github_task import run_github_analysis_background  # local import to avoid circular (if any)
    asyncio.create_task(run_github_analysis_background(analysis_results, analysis_id, github_url, threshold, cfg, explain, explain_top_k, skip_matrices))
    return {
        'analysis_id': analysis_id,
        'status': 'started',
        'message': 'Analisis repository GitHub dimulai',
        'repository_url': github_url,
        'preset': preset,
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
    threshold: float = Field(0.7, description="Ambang batas kemiripan (0-1)")
    min_tokens: int = Field(1, description="Minimum token yang digunakan dalam analisis")
    preset: Optional[str] = Field('strict', description="Nama preset (strict|balanced|permissive) untuk menerapkan konfigurasi token & TF-IDF.")
    explain: bool = Field(False, description="Aktifkan explain mode untuk melihat top fitur.")
    explain_top_k: int = Field(5, description="Jumlah fitur teratas saat explain mode.")

@app.post("/analyze/manual", summary="Analisis manual dua kode", response_description="Hasil analisis kemiripan dua kode")
async def analyze_manual_code(payload: ManualAnalysisRequest = Body(...)):
    """Analisis dua kode sumber secara manual. Sekarang mendukung preset untuk konsistensi dengan endpoint multi-file."""
    try:
        if not payload.code1.strip() or not payload.code2.strip():
            return {"error": "Kode tidak boleh kosong."}
        threshold = float(payload.threshold)
        # Jika preset diberikan gunakan pipeline lengkap reuse helper
        if payload.preset:
            preset_name = payload.preset
            if preset_name not in PRESETS:
                return {"error": f"Preset '{preset_name}' tidak ditemukan."}
            cfg = PRESETS[preset_name]['params']
            # Bangun pseudo filename agar deteksi bahasa sesuai
            fname1 = f"manual1.{ 'py' if payload.lang1=='python' else ('js' if payload.lang1=='javascript' else 'py')}"
            fname2 = f"manual2.{ 'py' if payload.lang2=='python' else ('js' if payload.lang2=='javascript' else 'py')}"
            result = run_multi_file_analysis(
                [(fname1, payload.code1), (fname2, payload.code2)],
                cfg,
                threshold,
                payload.explain,
                payload.explain_top_k
            )
            result['preset'] = preset_name
            return result
        # Tanpa preset: jalankan pipeline sederhana (legacy behaviour)
        tokenizer = ASTTokenizer()
        tokens1 = tokenizer.tokenize_code(payload.code1, payload.lang1)
        tokens2 = tokenizer.tokenize_code(payload.code2, payload.lang2)
        if len(tokens1) < payload.min_tokens or len(tokens2) < payload.min_tokens:
            return {"error": f"Jumlah token kurang dari minimum ({payload.min_tokens})"}
        vectorizer = TFIDFVectorizer()
        vectorizer.fit([tokens1, tokens2])
        tfidf_matrix = vectorizer.transform([tokens1, tokens2])
        similarity_calc = CosineSimilarityCalculator()
        similarity = similarity_calc.calculate_similarity(tfidf_matrix[0], tfidf_matrix[1])
        status = "Plagiat" if similarity >= threshold else ("Mirip" if similarity >= 0.5 else "Aman")
        return {
            "filesCount": 2,
            "comparisons": 1,
            "plagiarismCount": 1 if similarity >= threshold else 0,
            "comparisonsDetail": [{
                "file1": f"manual1.{payload.lang1}",
                "file2": f"manual2.{payload.lang2}",
                "similarity": similarity,
                "status": status,
                "plagiarizedFragment": None
            }],
            "similarityScores": [similarity]
        }
    except Exception as e:
        logger.error(f"Error in manual code analysis: {str(e)}")
        return {"error": str(e), "detail": "Gagal menganalisis kode. Pastikan input sudah benar dan kode tidak kosong."}
###############################################
# Endpoint: Multi-file compare
###############################################
from fastapi import UploadFile
from typing import List

@app.post("/analyze/compare", summary="Bandingkan banyak file", response_description="Hasil analisis kemiripan multi-file")
async def compare_multiple_files(
    files: List[UploadFile] = File(..., description="File kode sumber yang diupload"),
    threshold: float = Form(0.7, description="Ambang batas kemiripan (0-1)"),
    preset: Optional[str] = Form('strict', description="Nama preset (strict|balanced|permissive) untuk langsung menerapkan konfigurasi"),
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
    operator_weight: float = Form(1.0, description="Faktor bobot untuk token operator (OP_, CMP_, BINARY_OP, OPERATOR) (0-1)")
):
    """
    Bandingkan banyak file kode sumber untuk deteksi plagiarisme.
    Cocok untuk integrasi API website lain.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file diperlukan untuk analisis.")
    # Jika preset dipakai gunakan helper agar konsisten
    if preset:
        if preset not in PRESETS:
            raise HTTPException(status_code=404, detail=f"Preset '{preset}' tidak ditemukan")
        preset_cfg = PRESETS[preset]
        th = threshold if threshold is not None else preset_cfg['default_threshold']
        file_contents = []
        for f in files:
            code = (await f.read()).decode(errors='ignore')
            if code.strip():
                file_contents.append((f.filename, code))
        if len(file_contents) < 2:
            raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
        result = run_multi_file_analysis(file_contents, preset_cfg['params'], th, explain, explain_top_k)
        result['preset'] = preset
        return result
    file_contents = []
    for f in files:
        content = (f.filename, (await f.read()).decode(errors='ignore'))
        file_contents.append(content)
    # Hanya file yang tidak kosong
    file_contents = [(name, code) for name, code in file_contents if code.strip()]
    if len(file_contents) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
    # Modular pipeline (non-preset manual path)
    tokenizer = ASTTokenizer()
    # Deteksi bahasa dari ekstensi file
    langs = []
    tokens_list = []
    for name, code in file_contents:
        ext = name.split('.')[-1].lower()
        lang = 'python' if ext == 'py' else ('javascript' if ext in ['js', 'jsx', 'ts', 'tsx'] else 'python')
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
    result = {
        "filesCount": filesCount,
        "comparisons": comparisons,
        "plagiarismCount": plagiarismCount,
        "processingTime": "0.1",
        "comparisonsDetail": comparisons_detail,
        "similarityScores": similarityScores,
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
        "explain": explain_details if explain else None
    }
    return result