#!/usr/bin/env python3
"""
Enhanced FastAPI server with detailed plagiarism reporting
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeGuard Enhanced Plagiarism Detection",
    version="2.0",
    description="API untuk deteksi plagiarisme kode sumber menggunakan AST, TF-IDF, dan Cosine Similarity. Cocok untuk integrasi website lain."
)

# Global storage for analysis results
analysis_results = {}

@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Serve external index.html UI"""
    return FileResponse("index.html")

# ---------------------------------------------------------------------------
# Core detection helper classes (restored after cleanup)
# ---------------------------------------------------------------------------


from core.enhanced_detector import EnhancedPlagiarismDetector
from core.github_analyzer import GitHubRepositoryAnalyzer
from core.file_plagiarism import analyze_files_for_plagiarism

# Global detector and analyzer instances
detector = EnhancedPlagiarismDetector()
github_analyzer = GitHubRepositoryAnalyzer()


@app.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis ID tidak ditemukan")
    
    return analysis_results[analysis_id]

@app.post("/analyze/github")
async def analyze_github_repository(
    github_url: str = Form(...),
    threshold: float = Form(0.7)
):
    """Analyze GitHub repository for plagiarism"""
    try:
        # Validate GitHub URL
        if not github_url.startswith('https://github.com/'):
            raise HTTPException(status_code=400, detail="URL harus berupa repository GitHub yang valid")
        
        analysis_id = str(uuid.uuid4())
        
        # Store initial status
        analysis_results[analysis_id] = {
            'status': 'processing',
            'started_at': datetime.now().isoformat(),
            'repository_url': github_url,
            'type': 'github_repository'
        }
        
        # Run analysis in background
        asyncio.create_task(run_github_analysis_background(analysis_id, github_url, threshold))
        
        return {
            'analysis_id': analysis_id,
            'status': 'started',
            'message': 'Analisis repository GitHub dimulai',
            'repository_url': github_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GitHub analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

async def run_github_analysis_background(analysis_id: str, github_url: str, threshold: float):
    """Run GitHub repository analysis in background"""
    try:
        logger.info(f"Starting GitHub analysis {analysis_id} for {github_url}")
        
        # Download files from repo
        files = github_analyzer.download_repo(github_url)
        if len(files) < 2:
            analysis_results[analysis_id] = {
                'status': 'insufficient_files',
                'message': 'Repository needs at least 2 Python files for comparison',
                'total_files': len(files),
                'files_found': [f[0] for f in files]
            }
            return
        # Modular pipeline
        tokenizer = ASTTokenizer()
        langs = []
        tokens_list = []
        for name, code in files:
            ext = name.split('.')[-1].lower()
            lang = 'python' if ext == 'py' else ('javascript' if ext in ['js', 'jsx', 'ts', 'tsx'] else 'python')
            langs.append(lang)
            tokens = tokenizer.tokenize_code(code, lang)
            tokens_list.append(tokens)
        logger.info(f"[GITHUB] Tokens list: {tokens_list}")
        vectorizer = TFIDFVectorizer()
        vectorizer.fit(tokens_list)
        logger.info(f"[GITHUB] TFIDF Vocabulary: {vectorizer.get_vocabulary()}")
        tfidf_matrix = vectorizer.transform(tokens_list)
        logger.info(f"[GITHUB] TFIDF Matrix: {tfidf_matrix}")
        similarity_calc = CosineSimilarityCalculator()
        similarity_matrix = similarity_calc.calculate_similarity_matrix(tfidf_matrix)
        logger.info(f"[GITHUB] Similarity Matrix: {similarity_matrix}")
        # Buat hasil detail
        comparisons_detail = []
        filesCount = len(files)
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
                comparisons_detail.append({
                    "file1": files[i][0],
                    "file2": files[j][0],
                    "similarity": sim,
                    "status": status,
                    "plagiarizedFragment": None
                })
        analysis_results[analysis_id] = {
            'status': 'completed',
            'started_at': analysis_results[analysis_id]['started_at'],
            'completed_at': datetime.now().isoformat(),
            'type': 'github_repository',
            'repository_url': github_url,
            'total_files': filesCount,
            'total_comparisons': comparisons,
            'plagiarism_cases': plagiarismCount,
            'all_comparisons': comparisons_detail,
            'results': comparisons_detail,
            'comparisonsDetail': comparisons_detail,
            'similarityScores': similarityScores,
            'summary': {
                'plagiarism_rate': (plagiarismCount / comparisons) if comparisons else 0.0,
                'threshold_used': threshold
            }
        }
        
        logger.info(f"GitHub analysis {analysis_id} completed. Found {results.get('plagiarism_cases', 0)} cases")
        
    except Exception as e:
        logger.error(f"Error in background GitHub analysis {analysis_id}: {str(e)}")
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat(),
            'type': 'github_repository'
        }

# Endpoint baru: menerima banyak file dan membandingkan semuanya
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

@app.post("/analyze/manual", summary="Analisis manual dua kode", response_description="Hasil analisis kemiripan dua kode")
async def analyze_manual_code(payload: ManualAnalysisRequest = Body(...)):
    """
    Analisis dua kode sumber secara manual menggunakan AST, TF-IDF, dan Cosine Similarity.
    Cocok untuk integrasi API website lain.
    """
    try:
    code1 = payload.code1
    code2 = payload.code2
    lang1 = payload.lang1
    lang2 = payload.lang2
    threshold = float(payload.threshold)
    min_tokens = int(payload.min_tokens)
        if not code1.strip() or not code2.strip():
            logger.warning("Input kode kosong.")
            return {"error": "Kode tidak boleh kosong."}
        # Tokenisasi AST
        tokenizer = ASTTokenizer()
        tokens1 = tokenizer.tokenize_code(code1, lang1)
        tokens2 = tokenizer.tokenize_code(code2, lang2)
        # Filter token sesuai min_tokens
        if len(tokens1) < min_tokens or len(tokens2) < min_tokens:
            logger.warning("Jumlah token kurang dari minimum.")
            return {"error": f"Jumlah token kurang dari minimum ({min_tokens})"}
        logger.info(f"Tokens1 ({lang1}): {tokens1}")
        logger.info(f"Tokens2 ({lang2}): {tokens2}")
        # TF-IDF vektorisasi
        vectorizer = TFIDFVectorizer()
        vectorizer.fit([tokens1, tokens2])
        logger.info(f"TFIDF Vocabulary: {vectorizer.get_vocabulary()}")
        tfidf_matrix = vectorizer.transform([tokens1, tokens2])
        logger.info(f"TFIDF Matrix: {tfidf_matrix}")
        # Hitung cosine similarity
        similarity_calc = CosineSimilarityCalculator()
        similarity = similarity_calc.calculate_similarity(tfidf_matrix[0], tfidf_matrix[1])
        logger.info(f"Similarity Score: {similarity}")
        # Status
        status = "Plagiat" if similarity >= threshold else ("Mirip" if similarity >= 0.5 else "Aman")
        result = {
            "filesCount": 2,
            "comparisons": 1,
            "plagiarismCount": 1 if similarity >= threshold else 0,
            "processingTime": "0.1",
            "comparisonsDetail": [{
                "file1": f"manual1.{lang1}",
                "file2": f"manual2.{lang2}",
                "similarity": similarity,
                "status": status,
                "plagiarizedFragment": None
            }],
            "similarityScores": [similarity]
        }
        return result
    except Exception as e:
        logger.error(f"Error in manual code analysis: {str(e)}")
        return {"error": str(e), "detail": "Gagal menganalisis kode. Pastikan input sudah benar dan kode tidak kosong."}
from fastapi import UploadFile
from typing import List

@app.post("/analyze/compare", summary="Bandingkan banyak file", response_description="Hasil analisis kemiripan multi-file")
async def compare_multiple_files(
    files: List[UploadFile] = File(..., description="File kode sumber yang diupload"),
    threshold: float = Form(0.7, description="Ambang batas kemiripan (0-1)"),
    min_tokens: int = Form(1, description="Minimum token yang digunakan dalam analisis")
):
    """
    Bandingkan banyak file kode sumber untuk deteksi plagiarisme.
    Cocok untuk integrasi API website lain.
    """
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="Minimal 2 file diperlukan untuk analisis.")
        file_contents = []
        for f in files:
            content = (f.filename, (await f.read()).decode(errors='ignore'))
            file_contents.append(content)
        # Hanya file yang tidak kosong
        file_contents = [(name, code) for name, code in file_contents if code.strip()]
        if len(file_contents) < 2:
            raise HTTPException(status_code=400, detail="Minimal 2 file valid diperlukan.")
        # Modular pipeline
        tokenizer = ASTTokenizer()
        # Deteksi bahasa dari ekstensi file
        langs = []
        tokens_list = []
        for name, code in file_contents:
            ext = name.split('.')[-1].lower()
            lang = 'python' if ext == 'py' else ('javascript' if ext in ['js', 'jsx', 'ts', 'tsx'] else 'python')
            langs.append(lang)
            tokens = tokenizer.tokenize_code(code, lang)
            if len(tokens) >= min_tokens:
                tokens_list.append(tokens)
            else:
                logger.warning(f"File {name} jumlah token kurang dari minimum.")
        logger.info(f"Tokens list: {tokens_list}")
        vectorizer = TFIDFVectorizer()
        vectorizer.fit(tokens_list)
        logger.info(f"TFIDF Vocabulary: {vectorizer.get_vocabulary()}")
        tfidf_matrix = vectorizer.transform(tokens_list)
        logger.info(f"TFIDF Matrix: {tfidf_matrix}")
        similarity_calc = CosineSimilarityCalculator()
        similarity_matrix = similarity_calc.calculate_similarity_matrix(tfidf_matrix)
        logger.info(f"Similarity Matrix: {similarity_matrix}")
        # Buat hasil detail
        comparisons_detail = []
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
                comparisons_detail.append({
                    "file1": file_contents[i][0],
                    "file2": file_contents[j][0],
                    "similarity": sim,
                    "status": status,
                    "plagiarizedFragment": None
                })
        result = {
            "filesCount": filesCount,
            "comparisons": comparisons,
            "plagiarismCount": plagiarismCount,
            "processingTime": "0.1",
            "comparisonsDetail": comparisons_detail,
            "similarityScores": similarityScores
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-file comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
