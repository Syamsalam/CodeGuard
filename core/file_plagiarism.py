from typing import List, Tuple, Dict, Optional
from .enhanced_detector import EnhancedPlagiarismDetector

def analyze_files_for_plagiarism(files: List[Tuple[str, str]], threshold: float = 0.7, repo_url: Optional[str] = None) -> Dict:
    detector = EnhancedPlagiarismDetector()
    results = detector.detect_plagiarism(files, threshold=threshold)
    plagiarism_cases = [r for r in results if r['is_plagiarized']]
    report = {
        'repo_url': repo_url,
        'total_files': len(files),
        'comparisons': len(results),
        'plagiarism_cases': len(plagiarism_cases),
        'comparisonsDetail': results,
        'similarityScores': [r['similarity'] for r in results]
    }
    return report
