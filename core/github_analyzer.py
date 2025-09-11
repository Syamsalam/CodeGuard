import requests
import zipfile
import io
import logging
from typing import List, Dict
from .enhanced_detector import EnhancedPlagiarismDetector

logger = logging.getLogger(__name__)

class GitHubRepositoryAnalyzer:
    branches_fallback = ["main", "master", "develop", "dev", "trunk"]
    def __init__(self):
        self.detector = EnhancedPlagiarismDetector()
        self.max_files = 400

    def _candidate_branches(self, owner: str, repo: str) -> List[str]:
        out: List[str] = []
        api = f"https://api.github.com/repos/{owner}/{repo}"
        try:
            r = requests.get(api, timeout=8)
            if r.status_code == 200:
                d = r.json().get('default_branch')
                if d:
                    out.append(d)
        except Exception as e:
            logger.warning(f"Branch probe failed: {e}")
        for b in self.branches_fallback:
            if b not in out:
                out.append(b)
        return out

    def download_repo(self, url: str) -> List[tuple]:
        if url.endswith('.git'):
            url = url[:-4]
        if not url.startswith('https://github.com/'):
            raise ValueError('Invalid GitHub URL')
        parts = url.replace('https://github.com/', '').split('/')
        if len(parts) < 2:
            raise ValueError('Invalid GitHub repository URL')
        owner, repo = parts[0], parts[1]
        attempts = []
        zf = None
        used = None
        for br in self._candidate_branches(owner, repo):
            zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{br}"
            try:
                resp = requests.get(zip_url, timeout=25)
                attempts.append(f"{br}:{resp.status_code}")
                if resp.status_code == 200:
                    zf = zipfile.ZipFile(io.BytesIO(resp.content))
                    used = br
                    break
            except Exception as e:
                attempts.append(f"{br}:ERR")
                logger.warning(f"Download fail {br}: {e}")
        if not zf:
            raise ValueError(f"Cannot download repo. Attempts: {attempts}")
        py_files: List[tuple] = []
        for fi in zf.filelist:
            if fi.is_dir() or not fi.filename.endswith('.py'):
                continue
            if '__pycache__' in fi.filename or '/.' in fi.filename:
                continue
            try:
                raw = zf.read(fi.filename)
                try:
                    content = raw.decode('utf-8')
                except UnicodeDecodeError:
                    content = raw.decode('latin-1', errors='ignore')
                if len(content.strip()) < 15:
                    continue
                name = fi.filename.split('/')[-1]
                py_files.append((name, content))
                if len(py_files) >= self.max_files:
                    break
            except Exception:
                continue
        if len(py_files) < 1:
            raise ValueError('No Python files extracted')
        return py_files

    def analyze_github_repository(self, url: str, threshold: float = 0.7) -> Dict:
        files = self.download_repo(url)
        if len(files) < 2:
            return {
                'status': 'insufficient_files',
                'message': 'Repository needs at least 2 Python files for comparison',
                'total_files': len(files),
                'files_found': [f[0] for f in files]
            }
        from .file_plagiarism import analyze_files_for_plagiarism
        return analyze_files_for_plagiarism(files, threshold, repo_url=url)
