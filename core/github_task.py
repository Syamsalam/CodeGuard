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
from core.enhanced_detector import EnhancedPlagiarismDetector
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
                lang = 'python' if ext == 'py' else ('typescript' if ext in ['ts','tsx'] else ('javascript' if ext in ['js','jsx'] else 'python'))
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

# ==============================================================
# Cross-repository (account-level) analysis helpers and task
# ==============================================================
import requests
class RateLimitExceeded(Exception):
    def __init__(self, remaining: Optional[str], reset: Optional[str]):
        self.remaining = remaining
        self.reset = reset
        msg = "GitHub API rate limit exceeded"
        if remaining is not None:
            msg += f" (remaining={remaining})"
        if reset is not None:
            msg += f"; reset at {reset}"
        super().__init__(msg)

def _gh_headers(token: Optional[str] = None) -> dict:
    h = { 'Accept': 'application/vnd.github+json', 'User-Agent': 'CodeGuard' }
    if token:
        h['Authorization'] = f'Bearer {token}'
    return h

def _fetch_repos_for_account(username: str, include_forks: bool, max_repos: int, token: Optional[str] = None) -> List[dict]:
    """
    Fetch public repositories for a user/org (GitHub). Tries user endpoint first, then org.
    Returns a list of repo objects (subset fields used: name, html_url, fork).
    """
    per_page = 100
    repos: List[dict] = []
    session = requests.Session()
    session.headers.update(_gh_headers(token))
    endpoints = [
        f"https://api.github.com/users/{username}/repos",
        f"https://api.github.com/orgs/{username}/repos",
    ]
    for base in endpoints:
        page = 1
        try:
            while len(repos) < max_repos:
                url = f"{base}?per_page={per_page}&page={page}"
                r = session.get(url, timeout=15)
                # Explicitly handle rate limiting to surface clear errors
                if r.status_code == 403:
                    remaining = r.headers.get('X-RateLimit-Remaining')
                    reset = r.headers.get('X-RateLimit-Reset')
                    raise RateLimitExceeded(remaining, reset)
                if r.status_code == 404:
                    break  # try next endpoint
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list) or not data:
                    break
                for repo in data:
                    if (not include_forks) and repo.get('fork'):
                        continue
                    repos.append({
                        'name': repo.get('name'),
                        'full_name': repo.get('full_name'),
                        'html_url': repo.get('html_url'),
                        'default_branch': repo.get('default_branch'),
                        'fork': repo.get('fork'),
                    })
                    if len(repos) >= max_repos:
                        break
                if len(data) < per_page:
                    break
                page += 1
        except Exception as e:
            logger.warning(f"[GitHub] Failed fetching repos from {base}: {e}")
        if repos:
            break  # prefer first successful endpoint
    return repos

async def run_github_account_crossrepo_background(
    analysis_results: Dict,
    analysis_id: str,
    username: str,
    threshold: float,
    cfg: Optional[Dict],
    explain: bool,
    explain_top_k: int,
    skip_matrices: bool,
    include_forks: bool,
    max_repos: int,
    token: Optional[str] = None
):
    """Analyze multiple repositories under a GitHub account and compare repos pairwise.

    Produces aggregated per-repo-pair metrics and top file-pair matches.
    """
    logger.info(f"[GitHub-XRepo] Start analysis id={analysis_id} user={username}")
    if not _allow_request_now():
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': 'Rate limit exceeded. Try again later.'
        }
        return
    try:
        repos = _fetch_repos_for_account(username, include_forks, max_repos, token)
        if not repos:
            analysis_results[analysis_id] = {
                'status': 'error',
                'error': f'No repositories found for {username}'
            }
            return
        # Download files for each repo
        repo_files: List[Tuple[str, List[Tuple[str, str]]]] = []
        repo_urls: List[str] = []
        repo_urls: List[str] = []
        total_to_download = len(repos)
        for idx, r in enumerate(repos, start=1):
            repo_url = r.get('html_url')
            try:
                cached = _get_cached_repo(repo_url)
                if cached is not None:
                    files = cached
                else:
                    files = github_analyzer.download_repo(repo_url)
                    _store_repo_cache(repo_url, files)
                repo_files.append((r.get('name') or r.get('full_name') or repo_url.split('/')[-1], files))
                repo_urls.append(repo_url)
            except Exception as e:
                logger.warning(f"[GitHub-XRepo] Skip repo {repo_url}: {e}")
            analysis_results[analysis_id]['processed_repos'] = idx
            analysis_results[analysis_id]['total_repos'] = total_to_download
            analysis_results[analysis_id]['status'] = 'processing'
        if len(repo_files) < 2:
            analysis_results[analysis_id] = {
                'status': 'error',
                'error': 'Need at least 2 repositories with code files to compare.'
            }
            return
        # Cross-repo comparisons
        detector = EnhancedPlagiarismDetector(threshold=threshold)
        repo_pairs_results = []
        names = [name for name, _ in repo_files]
        for i in range(len(repo_files)):
            name_a, files_a = repo_files[i]
            for j in range(i+1, len(repo_files)):
                name_b, files_b = repo_files[j]
                # Compare A vs B only
                comp = detector.detect_cross_repository_plagiarism(files_a, files_b, threshold)
                # Derive best-match (one-to-one greedy) and same-filename metrics
                def _greedy_best_match(pairs):
                    used_a = set()
                    used_b = set()
                    matched = []
                    # sort by similarity descending
                    for p in sorted(pairs, key=lambda x: x['similarity'], reverse=True):
                        a = p['repo_a_file']
                        b = p['repo_b_file']
                        if a in used_a or b in used_b:
                            continue
                        used_a.add(a)
                        used_b.add(b)
                        matched.append(p)
                    return matched
                def _basename(path: str) -> str:
                    try:
                        return path.replace('\\','/').split('/')[-1]
                    except Exception:
                        return path
                if not comp:
                    agg = {
                        'repoA': name_a,
                        'repoB': name_b,
                        'repoA_url': repo_urls[i] if i < len(repo_urls) else None,
                        'repoB_url': repo_urls[j] if j < len(repo_urls) else None,
                        'max_similarity': 0.0,
                        'mean_similarity': 0.0,
                        'plagiarized_pairs': 0,
                        'total_pairs': 0,
                        'any_plagiarism': False,
                        'top_file_pairs': []
                    }
                else:
                    sims = [c['similarity'] for c in comp]
                    max_sim = max(sims) if sims else 0.0
                    mean_sim = sum(sims)/len(sims) if sims else 0.0
                    plag_pairs = sum(1 for c in comp if c.get('is_plagiarized'))
                    # top 5 pairs by similarity
                    top_sorted = sorted(comp, key=lambda x: x['similarity'], reverse=True)[:5]
                    top_pairs = [{
                        'fileA': t['repo_a_file'],
                        'fileB': t['repo_b_file'],
                        'similarity': float(t['similarity'])
                    } for t in top_sorted]
                    # best-match greedy metrics
                    best = _greedy_best_match(comp)
                    best_sims = [p['similarity'] for p in best]
                    best_mean = (sum(best_sims)/len(best_sims)) if best_sims else 0.0
                    best_plag = sum(1 for p in best if p['similarity'] >= threshold)
                    # same-filename (basename) best match
                    same_name_pairs = [p for p in comp if _basename(p['repo_a_file']) == _basename(p['repo_b_file'])]
                    same_best = _greedy_best_match(same_name_pairs) if same_name_pairs else []
                    same_sims = [p['similarity'] for p in same_best]
                    same_mean = (sum(same_sims)/len(same_sims)) if same_sims else 0.0
                    same_plag = sum(1 for p in same_best if p['similarity'] >= threshold)
                    exact_dups = sum(1 for p in best if p['similarity'] >= 0.99)
                    agg = {
                        'repoA': name_a,
                        'repoB': name_b,
                        'repoA_url': repo_urls[i] if i < len(repo_urls) else None,
                        'repoB_url': repo_urls[j] if j < len(repo_urls) else None,
                        'max_similarity': float(max_sim),
                        'mean_similarity': float(mean_sim),
                        'plagiarized_pairs': int(plag_pairs),
                        'total_pairs': int(len(comp)),
                        'any_plagiarism': bool(plag_pairs > 0),
                        'top_file_pairs': top_pairs,
                        'best_match_mean_similarity': float(best_mean),
                        'best_match_plagiarized_pairs': int(best_plag),
                        'best_match_total_pairs': int(len(best)),
                        'same_name_mean_similarity': float(same_mean),
                        'same_name_plagiarized_pairs': int(same_plag),
                        'same_name_total_pairs': int(len(same_best)),
                        'exact_duplicate_pairs': int(exact_dups)
                    }
                repo_pairs_results.append(agg)
        # Sort pairs by max_similarity desc
        repo_pairs_results.sort(key=lambda x: x['max_similarity'], reverse=True)
        result = {
            'account': username,
            'repositoriesAnalyzed': len(names),
            'repoNames': names,
            'threshold_used': threshold,
            'repoPairs': repo_pairs_results
        }
        analysis_results[analysis_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'type': 'github_account_crossrepo',
            'result': result,
            'skip_matrices': skip_matrices,
            'total_repos': len(names),
            'processed_repos': len(names)
        })
        logger.info(f"[GitHub-XRepo] Analysis complete id={analysis_id}")
    except RateLimitExceeded as e:
        # Structured rate limit info for UI
        reset_ts = None
        try:
            if e.reset is not None:
                reset_ts = int(e.reset)
        except Exception:
            reset_ts = None
        err_msg = str(e) + '. Tambahkan GitHub token di UI untuk meningkatkan batas dan coba lagi.'
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': err_msg,
            'rate_limit': {
                'remaining': e.remaining,
                'reset': reset_ts,
                'reset_iso': (datetime.fromtimestamp(reset_ts).isoformat() if reset_ts else None)
            },
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub-XRepo] Rate limit: {err_msg}")
    except Exception as e:
        # Provide actionable message on rate limit
        err_msg = str(e)
        if 'rate limit' in err_msg.lower():
            err_msg += '. Tambahkan GitHub token di UI untuk meningkatkan batas dan coba lagi.'
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': err_msg,
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub-XRepo] Error: {e}")

__all__.append("run_github_account_crossrepo_background")

# ==============================================================
# Cross-repository via explicit list of repo URLs
# ==============================================================
async def run_github_crossrepo_urls_background(
    analysis_results: Dict,
    analysis_id: str,
    repo_urls: List[str],
    threshold: float,
    cfg: Optional[Dict],
    explain: bool,
    explain_top_k: int,
    skip_matrices: bool = True
):
    """Compare multiple repositories specified by their GitHub URLs pairwise.

    For each pair (A,B), compute cross-repo similarities using
    EnhancedPlagiarismDetector.detect_cross_repository_plagiarism and aggregate.
    """
    logger.info(f"[GitHub-XRepo-URLs] Start analysis id={analysis_id} urls={len(repo_urls)}")
    if not _allow_request_now():
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': 'Rate limit exceeded. Try again later.'
        }
        return
    try:
        # Normalize and validate URLs
        urls = []
        for u in repo_urls:
            if not isinstance(u, str):
                continue
            url = u.strip()
            if not url:
                continue
            if url.endswith('.git'):
                url = url[:-4]
            if not url.startswith('https://github.com/'):
                continue
            urls.append(url)
        urls = list(dict.fromkeys(urls))  # de-duplicate while preserving order
        if len(urls) < 2:
            analysis_results[analysis_id] = {
                'status': 'error',
                'error': 'Need at least 2 valid GitHub URLs.'
            }
            return
        # Download files for each URL
        repo_files: List[Tuple[str, List[Tuple[str, str]]]] = []
        total_to_download = len(urls)
        for idx, repo_url in enumerate(urls, start=1):
            try:
                cached = _get_cached_repo(repo_url)
                if cached is not None:
                    files = cached
                else:
                    files = github_analyzer.download_repo(repo_url)
                    _store_repo_cache(repo_url, files)
                # derive display name owner/repo
                try:
                    parts = repo_url.replace('https://github.com/','').split('/')
                    disp = f"{parts[0]}/{parts[1]}"
                except Exception:
                    disp = repo_url.split('/')[-1]
                repo_files.append((disp, files))
                repo_urls.append(repo_url)
            except Exception as e:
                logger.warning(f"[GitHub-XRepo-URLs] Skip {repo_url}: {e}")
            analysis_results[analysis_id]['processed_repos'] = idx
            analysis_results[analysis_id]['total_repos'] = total_to_download
            analysis_results[analysis_id]['status'] = 'processing'
        if len(repo_files) < 2:
            analysis_results[analysis_id] = {
                'status': 'error',
                'error': 'No sufficient repositories with code to compare.'
            }
            return
        detector = EnhancedPlagiarismDetector(threshold=threshold)
        repo_pairs_results = []
        names = [name for name, _ in repo_files]
        for i in range(len(repo_files)):
            name_a, files_a = repo_files[i]
            for j in range(i+1, len(repo_files)):
                name_b, files_b = repo_files[j]
                comp = detector.detect_cross_repository_plagiarism(files_a, files_b, threshold)
                def _greedy_best_match(pairs):
                    used_a = set()
                    used_b = set()
                    matched = []
                    for p in sorted(pairs, key=lambda x: x['similarity'], reverse=True):
                        a = p['repo_a_file']
                        b = p['repo_b_file']
                        if a in used_a or b in used_b:
                            continue
                        used_a.add(a)
                        used_b.add(b)
                        matched.append(p)
                    return matched
                def _basename(path: str) -> str:
                    try:
                        return path.replace('\\','/').split('/')[-1]
                    except Exception:
                        return path
                if not comp:
                    agg = {
                        'repoA': name_a,
                        'repoB': name_b,
                        'repoA_url': repo_urls[i] if i < len(repo_urls) else None,
                        'repoB_url': repo_urls[j] if j < len(repo_urls) else None,
                        'max_similarity': 0.0,
                        'mean_similarity': 0.0,
                        'plagiarized_pairs': 0,
                        'total_pairs': 0,
                        'any_plagiarism': False,
                        'top_file_pairs': []
                    }
                else:
                    sims = [c['similarity'] for c in comp]
                    max_sim = max(sims) if sims else 0.0
                    mean_sim = sum(sims)/len(sims) if sims else 0.0
                    plag_pairs = sum(1 for c in comp if c.get('is_plagiarized'))
                    top_sorted = sorted(comp, key=lambda x: x['similarity'], reverse=True)[:5]
                    top_pairs = [{
                        'fileA': t['repo_a_file'],
                        'fileB': t['repo_b_file'],
                        'similarity': float(t['similarity'])
                    } for t in top_sorted]
                    best = _greedy_best_match(comp)
                    best_sims = [p['similarity'] for p in best]
                    best_mean = (sum(best_sims)/len(best_sims)) if best_sims else 0.0
                    best_plag = sum(1 for p in best if p['similarity'] >= threshold)
                    same_name_pairs = [p for p in comp if _basename(p['repo_a_file']) == _basename(p['repo_b_file'])]
                    same_best = _greedy_best_match(same_name_pairs) if same_name_pairs else []
                    same_sims = [p['similarity'] for p in same_best]
                    same_mean = (sum(same_sims)/len(same_sims)) if same_sims else 0.0
                    same_plag = sum(1 for p in same_best if p['similarity'] >= threshold)
                    exact_dups = sum(1 for p in best if p['similarity'] >= 0.99)
                    agg = {
                        'repoA': name_a,
                        'repoB': name_b,
                        'repoA_url': repo_urls[i] if i < len(repo_urls) else None,
                        'repoB_url': repo_urls[j] if j < len(repo_urls) else None,
                        'max_similarity': float(max_sim),
                        'mean_similarity': float(mean_sim),
                        'plagiarized_pairs': int(plag_pairs),
                        'total_pairs': int(len(comp)),
                        'any_plagiarism': bool(plag_pairs > 0),
                        'top_file_pairs': top_pairs,
                        'best_match_mean_similarity': float(best_mean),
                        'best_match_plagiarized_pairs': int(best_plag),
                        'best_match_total_pairs': int(len(best)),
                        'same_name_mean_similarity': float(same_mean),
                        'same_name_plagiarized_pairs': int(same_plag),
                        'same_name_total_pairs': int(len(same_best)),
                        'exact_duplicate_pairs': int(exact_dups)
                    }
                repo_pairs_results.append(agg)
        repo_pairs_results.sort(key=lambda x: x['max_similarity'], reverse=True)
        result = {
            'repositoriesAnalyzed': len(names),
            'repoNames': names,
            'threshold_used': threshold,
            'repoPairs': repo_pairs_results
        }
        analysis_results[analysis_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'type': 'github_crossrepo_urls',
            'result': result,
            'skip_matrices': skip_matrices,
            'total_repos': len(names),
            'processed_repos': len(names)
        })
        logger.info(f"[GitHub-XRepo-URLs] Analysis complete id={analysis_id}")
    except Exception as e:
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub-XRepo-URLs] Error: {e}")

__all__.append("run_github_crossrepo_urls_background")

# ==============================================================
# Repo pair detail matrices (drill-down)
# ==============================================================
async def run_github_pair_detail_background(
    analysis_results: Dict,
    analysis_id: str,
    repo_a_url: str,
    repo_b_url: str,
    threshold: float,
    cfg: Optional[Dict],
    explain: bool,
    explain_top_k: int,
    skip_matrices: bool = False
):
    """Download two repositories, combine their files, and run the full multi-file analysis
    to produce matrices (cosine similarity, TF-IDF) and tokens for UI drill-down.
    """
    logger.info(f"[GitHub-XRepo-Detail] Start pair detail id={analysis_id} A={repo_a_url} B={repo_b_url}")
    if not _allow_request_now():
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': 'Rate limit exceeded. Try again later.'
        }
        return
    try:
        # Download with cache
        def _dl(url: str):
            cached = _get_cached_repo(url)
            if cached is not None:
                return cached
            files = github_analyzer.download_repo(url)
            _store_repo_cache(url, files)
            return files
        files_a = _dl(repo_a_url)
        files_b = _dl(repo_b_url)
        if len(files_a) == 0 or len(files_b) == 0:
            analysis_results[analysis_id] = {
                'status': 'error',
                'error': 'Repositories have no supported code files.'
            }
            return
        # Prefix filenames with repo to keep origin clear
        def _prefix(url: str) -> str:
            try:
                p = url.replace('https://github.com/','').strip('/')
                owner, repo = p.split('/')[:2]
                return f"{owner}/{repo}"
            except Exception:
                return url.split('/')[-1]
        pa = _prefix(repo_a_url)
        pb = _prefix(repo_b_url)
        combined = [(f"{pa}/{name}", code) for name, code in files_a] + [(f"{pb}/{name}", code) for name, code in files_b]
        # Run pipeline (preset if provided)
        if cfg:
            result = run_multi_file_analysis(combined, cfg, threshold, explain, explain_top_k)
        else:
            # Fallback: minimal pipeline (but here better to require preset for matrices consistency)
            result = run_multi_file_analysis(combined, {}, threshold, explain, explain_top_k)
        if skip_matrices:
            result.pop('similarityMatrix', None)
            result.pop('tfidf', None)
        analysis_results[analysis_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'type': 'github_pair_detail',
            'result': result,
            'skip_matrices': skip_matrices
        })
        logger.info(f"[GitHub-XRepo-Detail] Complete id={analysis_id}")
    except RateLimitExceeded as e:
        reset_ts = None
        try:
            if e.reset is not None:
                reset_ts = int(e.reset)
        except Exception:
            reset_ts = None
        err_msg = str(e) + '. Tambahkan GitHub token di UI untuk meningkatkan batas dan coba lagi.'
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': err_msg,
            'rate_limit': {
                'remaining': e.remaining,
                'reset': reset_ts,
                'reset_iso': (datetime.fromtimestamp(reset_ts).isoformat() if reset_ts else None)
            },
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub-XRepo-Detail] Rate limit: {err_msg}")
    except Exception as e:
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }
        logger.error(f"[GitHub-XRepo-Detail] Error: {e}")

__all__.append("run_github_pair_detail_background")
