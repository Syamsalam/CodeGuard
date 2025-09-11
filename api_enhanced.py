#!/usr/bin/env python3
"""
Enhanced FastAPI server with detailed plagiarism reporting
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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

app = FastAPI(title="CodeGuard Enhanced Plagiarism Detection", version="2.0")

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
        
        # Run analysis
        results = github_analyzer.analyze_github_repository(github_url, threshold)

        # Map keys from analyze_files_for_plagiarism output
        total_files = results.get('total_files', 0)
        total_comparisons = results.get('comparisons', 0)
        plagiarism_cases = results.get('plagiarism_cases', 0)
        comparisons_detail = results.get('comparisonsDetail', results.get('all_comparisons', []))

        analysis_results[analysis_id] = {
            'status': 'completed',
            'started_at': analysis_results[analysis_id]['started_at'],
            'completed_at': datetime.now().isoformat(),
            'type': 'github_repository',
            'repository_url': github_url,
            'total_files': total_files,
            'total_comparisons': total_comparisons,
            'plagiarism_cases': plagiarism_cases,
            # Provide both naming variants the frontend normalization handles
            'all_comparisons': comparisons_detail,
            'results': comparisons_detail,
            'comparisonsDetail': comparisons_detail,
            'similarityScores': results.get('similarityScores', []),
            'summary': {
                'plagiarism_rate': (plagiarism_cases / total_comparisons) if total_comparisons else 0.0,
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
from fastapi import UploadFile
from typing import List

@app.post("/analyze/compare")
async def compare_multiple_files(
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.7),
    min_tokens: int = Form(1)
):
    """Compare multiple uploaded files for plagiarism"""
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
        result = analyze_files_for_plagiarism(file_contents, threshold=threshold)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-file comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting CodeGuard Enhanced FastAPI Server...")
    print("📊 Using ENHANCED similarity algorithm with detailed reporting")
    print("🌐 Access: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
