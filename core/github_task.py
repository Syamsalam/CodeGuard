"""GitHub repository analysis task.

Mengekstrak logika asynchronous/background analisis repository GitHub
dari `api_enhanced.py` agar endpoint tetap ringkas.

Fungsi utama:
 - run_github_analysis_background: melakukan download repo dan analisis

Catatan: Fungsi ini tidak tergantung pada FastAPI secara langsung kecuali
untuk struktur data hasil. Dapat diuji secara terpisah.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from functools import lru_cache
import time

from core.github_analyzer import GitHubRepositoryAnalyzer
from core.multi_file_analysis import run_multi_file_analysis
from core.ast_tokenizer import ASTTokenizer
from core.tfidf_vectorizer import TFIDFVectorizer
from core.similarity import CosineSimilarityCalculator

logger = logging.getLogger(__name__)

github_analyzer = GitHubRepositoryAnalyzer()

# ==============================================
# Simple in-memory rate limiting (sliding window)
# ==============================================
_RATE_LIMIT_WINDOW_SECONDS = 60  # 1 minute window
_RATE_LIMIT_MAX_REQUESTS = 10     # max GitHub analyses per window
_rate_limit_events: List[float] = []

def _allow_request_now() -> bool:
    now = time.time()
    # drop old timestamps
    while _rate_limit_events and now - _rate_limit_events[0] > _RATE_LIMIT_WINDOW_SECONDS:
        _rate_limit_events.pop(0)
    if len(_rate_limit_events) >= _RATE_LIMIT_MAX_REQUESTS:
        return False
    _rate_limit_events.append(now)
    return True

# ==============================================
# Repository download caching (time-based)
# ==============================================
_REPO_CACHE_TTL_SECONDS = 300  # 5 minutes
_repo_cache: Dict[str, Dict[str, object]] = {}

def _get_cached_repo(url: str) -> Optional[List[Tuple[str, str]]]:
    entry = _repo_cache.get(url)
    if not entry:
        return None
    if time.time() - entry['time'] > _REPO_CACHE_TTL_SECONDS:
        # expired
        _repo_cache.pop(url, None)
        return None
    return entry['files']  # type: ignore

def _store_repo_cache(url: str, files: List[Tuple[str, str]]):
    _repo_cache[url] = { 'files': files, 'time': time.time() }

async def run_github_analysis_background(
    analysis_results: Dict,
    analysis_id: str,
    github_url: str,
    threshold: float,
    cfg: Optional[Dict],
    explain: bool,
    explain_top_k: int,
    skip_matrices: bool = False
):
    """Run GitHub repository analysis and update shared analysis_results dict.

    Parameters
    ----------
    analysis_results : dict (mutable)
        Global/state dict yang dikirim dari FastAPI module.
    analysis_id : str
        ID unik analisis.
    github_url : str
        URL repo GitHub.
    threshold : float
        Ambang batas plagiat.
    cfg : Optional[Dict]
        Konfigurasi preset jika ada.
    explain : bool
        Aktifkan explain mode.
    explain_top_k : int
        Top fitur per pasangan.
    """
    logger.info(f"[GitHub] Mulai analysis id={analysis_id} repo={github_url} preset={cfg is not None}")
    # Rate limiting check
    if not _allow_request_now():
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': 'Rate limit exceeded. Try again later.'
        }
        logger.warning("[GitHub] Rate limit exceeded for request")
        return
    try:
        cached = _get_cached_repo(github_url)
        if cached is not None:
            logger.info("[GitHub] Using cached repository files")
            files = cached
        else:
            files = github_analyzer.download_repo(github_url)
            _store_repo_cache(github_url, files)
    except Exception as e:
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': f'Download failed: {e}',
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub] Download error: {e}")
        return
    if len(files) < 2:
        analysis_results[analysis_id] = {
            'status': 'insufficient_files',
            'message': 'Repository needs at least 2 code files',
            'files_found': [f[0] for f in files]
        }
        return
    try:
        processed_tracker = {'count': 0}
        def _progress_cb(done: int, total: int):
            # update intermediate progress (only increasing)
            processed_tracker['count'] = done
            analysis_results[analysis_id]['processed_files'] = done
            analysis_results[analysis_id]['total_files'] = total
            analysis_results[analysis_id]['status'] = 'processing'
        if cfg:
            result = run_multi_file_analysis(files, cfg, threshold, explain, explain_top_k, progress_cb=_progress_cb)
        else:
            # Basic fallback (no preset)
            tokenizer = ASTTokenizer()
            tokens_list: List[List[str]] = []
            total_files = len(files)
            for i,(name, code) in enumerate(files, start=1):
                ext = name.split('.')[-1].lower()
                lang = 'python' if ext == 'py' else ('javascript' if ext in ['js', 'jsx', 'ts', 'tsx'] else 'python')
                tokens_list.append(tokenizer.tokenize_code(code, lang))
                analysis_results[analysis_id]['processed_files'] = i
                analysis_results[analysis_id]['total_files'] = total_files
            vectorizer = TFIDFVectorizer()
            vectorizer.fit(tokens_list)
            tfidf_matrix = vectorizer.transform(tokens_list)
            similarity_calc = CosineSimilarityCalculator()
            similarity_matrix = similarity_calc.calculate_similarity_matrix(tfidf_matrix)
            comps = []
            filesCount = len(files)
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
                    comps.append({
                        'file1': files[i][0],
                        'file2': files[j][0],
                        'similarity': sim,
                        'status': status
                    })
            result = {
                'filesCount': filesCount,
                'comparisons': comparisons,
                'plagiarismCount': plagiarismCount,
                'comparisonsDetail': comps,
                'similarityScores': similarityScores
            }
        # Derive aggregated fields for frontend compatibility
        # Optionally strip heavy matrices if requested
        if skip_matrices:
            result.pop('similarityMatrix', None)
            result.pop('tfidf', None)
        comparisons_detail = result.get('comparisonsDetail', [])
        plagiarism_cases = sum(1 for c in comparisons_detail if c.get('status') == 'Plagiat')
        total_files = result.get('filesCount') or result.get('files_count') or 0
        total_comparisons = result.get('comparisons') or len(comparisons_detail)
        analysis_results[analysis_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'repository_url': github_url,
            'preset': analysis_results[analysis_id].get('preset'),
            'threshold_used': threshold,
            'result': result,
            'skip_matrices': skip_matrices,
            # legacy keys
            'comparisonsDetail': comparisons_detail,
            'similarityScores': result.get('similarityScores'),
            # aggregated keys expected by UI mapping
            'total_files': total_files,
            'total_comparisons': total_comparisons,
            'plagiarism_cases': plagiarism_cases,
            'all_comparisons': comparisons_detail,
            'processed_files': total_files
        })
        logger.info(f"[GitHub] Analysis selesai id={analysis_id}")
    except Exception as e:
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub] Error processing repo: {e}")

__all__ = ["run_github_analysis_background"]
