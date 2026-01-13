"""
Main Plagiarism Detection Engine that combines all components.
Orchestrates AST tokenization, TF-IDF vectorization, and similarity calculation.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .ast_tokenizer import ASTTokenizer
from .tfidf_vectorizer import TFIDFVectorizer
from .similarity import CosineSimilarityCalculator
from .preset_config import get_preset_params, validate_preset_params


@dataclass
class DetectionResult:
    """Result of plagiarism detection analysis"""
    file1_path: str
    file2_path: str
    similarity_score: float
    is_plagiarism: bool
    file1_tokens: int
    file2_tokens: int
    common_tokens: int
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class DetectionReport:
    """Complete plagiarism detection report"""
    total_files: int
    total_comparisons: int
    plagiarism_cases: List[DetectionResult]
    similarity_matrix: np.ndarray
    file_paths: List[str]
    statistics: Dict[str, float]
    processing_time: float
    threshold: float


class PlagiarismDetector:
    def __init__(self, 
                 similarity_threshold: float = 0.72,
                 min_tokens: int = 10,
                 min_df: int = 1,
                 max_df: float = 0.80,
                 normalize_vectors: bool = True):
        """
        Initialize Plagiarism Detector.
        
        Args:
            similarity_threshold: Minimum similarity to consider plagiarism
            min_tokens: Minimum number of tokens required for analysis
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            normalize_vectors: Whether to normalize TF-IDF vectors
        """
        self.similarity_threshold = similarity_threshold
        self.min_tokens = min_tokens
        
        # Initialize components
        self.tokenizer = ASTTokenizer()
        self.vectorizer = TFIDFVectorizer(
            min_df=min_df, 
            max_df=max_df, 
            normalize=normalize_vectors
        )
        self.similarity_calculator = CosineSimilarityCalculator()
        
        # Internal state
        self.is_fitted = False
        self.file_tokens = {}
        self.file_paths = []
        self.tfidf_matrix = None
    
    def detect_in_directory(self, directory_path: str, 
                           file_extensions: List[str] = ['.py', '.js'],
                           recursive: bool = True,
                           progress_callback: Optional[callable] = None) -> DetectionReport:
        """
        Detect plagiarism in all source code files in a directory.
        
        Args:
            directory_path: Path to directory containing source files
            file_extensions: List of file extensions to analyze
            recursive: Whether to search recursively in subdirectories
            progress_callback: Optional callback for progress updates
            
        Returns:
            DetectionReport with analysis results
        """
        start_time = time.time()
        
        # Find all source code files
        source_files = self._find_source_files(directory_path, file_extensions, recursive)
        
        if len(source_files) < 2:
            raise ValueError("At least 2 source files are required for comparison")
        
        # Detect plagiarism between files
        report = self.detect_between_files(source_files, progress_callback)
        report.processing_time = time.time() - start_time
        
        return report
    
    def detect_between_files(self, file_paths: List[str],
                           progress_callback: Optional[callable] = None) -> DetectionReport:
        """
        Detect plagiarism between a list of files.
        
        Args:
            file_paths: List of file paths to compare
            progress_callback: Optional callback for progress updates
            
        Returns:
            DetectionReport with analysis results
        """
        if len(file_paths) < 2:
            raise ValueError("At least 2 files are required for comparison")
        
        # Tokenize all files
        if progress_callback:
            progress_callback("Tokenizing files...")
        
        all_tokens = []
        valid_files = []
        file_token_counts = {}
        
        for file_path in tqdm(file_paths, desc="Tokenizing files"):
            try:
                tokens = self.tokenizer.tokenize_file(file_path)
                
                if len(tokens) >= self.min_tokens:
                    all_tokens.append(tokens)
                    valid_files.append(file_path)
                    file_token_counts[file_path] = len(tokens)
                else:
                    print(f"Warning: Skipping {file_path} - insufficient tokens ({len(tokens)} < {self.min_tokens})")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if len(valid_files) < 2:
            raise ValueError("At least 2 valid files with sufficient tokens are required")
        
        self.file_paths = valid_files
        self.file_tokens = {path: all_tokens[i] for i, path in enumerate(valid_files)}
        
        # Fit TF-IDF vectorizer and transform documents
        if progress_callback:
            progress_callback("Computing TF-IDF vectors...")
        
        self.tfidf_matrix = self.vectorizer.fit_transform(all_tokens)
        self.is_fitted = True
        
        # Calculate similarity matrix
        if progress_callback:
            progress_callback("Computing similarity matrix...")
        
        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(self.tfidf_matrix)
        
        # Find plagiarism cases
        if progress_callback:
            progress_callback("Identifying plagiarism cases...")
        
        similar_pairs = self.similarity_calculator.find_similar_pairs(
            similarity_matrix, 
            threshold=self.similarity_threshold
        )
        
        # Create detection results
        plagiarism_cases = []
        for file1_idx, file2_idx, similarity in similar_pairs:
            file1_path = valid_files[file1_idx]
            file2_path = valid_files[file2_idx]
            
            # Get token counts and common tokens
            tokens1 = self.file_tokens[file1_path]
            tokens2 = self.file_tokens[file2_path]
            common_tokens = len(set(tokens1) & set(tokens2))
            
            # Create explanation
            explanation = self.vectorizer.explain_similarity(tokens1, tokens2)
            
            result = DetectionResult(
                file1_path=file1_path,
                file2_path=file2_path,
                similarity_score=similarity,
                is_plagiarism=similarity >= self.similarity_threshold,
                file1_tokens=len(tokens1),
                file2_tokens=len(tokens2),
                common_tokens=common_tokens,
                explanation=explanation
            )
            
            plagiarism_cases.append(result)
        
        # Calculate statistics
        stats = self.similarity_calculator.calculate_similarity_statistics(similarity_matrix)
        
        # Create report
        report = DetectionReport(
            total_files=len(valid_files),
            total_comparisons=len(valid_files) * (len(valid_files) - 1) // 2,
            plagiarism_cases=plagiarism_cases,
            similarity_matrix=similarity_matrix,
            file_paths=valid_files,
            statistics=stats,
            processing_time=0.0,  # Will be set by caller
            threshold=self.similarity_threshold
        )
        
        return report
    
    def analyze_file_against_corpus(self, target_file: str, 
                                   corpus_files: List[str]) -> List[DetectionResult]:
        """
        Analyze a single file against a corpus of files.
        
        Args:
            target_file: Path to the file to analyze
            corpus_files: List of corpus file paths to compare against
            
        Returns:
            List of detection results sorted by similarity
        """
        if not self.is_fitted:
            # Fit on corpus first
            self.detect_between_files(corpus_files + [target_file])
        
        results = []
        target_idx = self.file_paths.index(target_file)
        
        for i, corpus_file in enumerate(self.file_paths):
            if corpus_file == target_file:
                continue
            
            similarity = self.tfidf_matrix[target_idx].dot(self.tfidf_matrix[i].T)
            if hasattr(similarity, 'item'):
                similarity = similarity.item()
            
            if similarity >= self.similarity_threshold:
                tokens1 = self.file_tokens[target_file]
                tokens2 = self.file_tokens[corpus_file]
                common_tokens = len(set(tokens1) & set(tokens2))
                
                explanation = self.vectorizer.explain_similarity(tokens1, tokens2)
                
                result = DetectionResult(
                    file1_path=target_file,
                    file2_path=corpus_file,
                    similarity_score=similarity,
                    is_plagiarism=similarity >= self.similarity_threshold,
                    file1_tokens=len(tokens1),
                    file2_tokens=len(tokens2),
                    common_tokens=common_tokens,
                    explanation=explanation
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def get_similarity_clusters(self, min_cluster_size: int = 2) -> List[List[str]]:
        """
        Get clusters of similar files.
        
        Args:
            min_cluster_size: Minimum number of files in a cluster
            
        Returns:
            List of clusters (each cluster is a list of file paths)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(self.tfidf_matrix)
        clusters = self.similarity_calculator.detect_clusters(
            similarity_matrix, 
            threshold=self.similarity_threshold
        )
        
        # Convert indices to file paths and filter by size
        file_clusters = []
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                file_cluster = [self.file_paths[idx] for idx in cluster]
                file_clusters.append(file_cluster)
        
        return file_clusters
    
    def _find_source_files(self, directory: str, extensions: List[str], 
                          recursive: bool = True) -> List[str]:
        """Find source code files in directory"""
        source_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            files = list(directory_path.glob(f"{pattern}{ext}"))
            source_files.extend([str(f) for f in files if f.is_file()])
        
        return sorted(source_files)
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model components"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted detector")
        
        import pickle
        
        model_data = {
            'vectorizer': self.vectorizer,
            'file_paths': self.file_paths,
            'file_tokens': self.file_tokens,
            'tfidf_matrix': self.tfidf_matrix,
            'similarity_threshold': self.similarity_threshold,
            'min_tokens': self.min_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.file_paths = model_data['file_paths']
        self.file_tokens = model_data['file_tokens']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.similarity_threshold = model_data['similarity_threshold']
        self.min_tokens = model_data['min_tokens']
        self.is_fitted = True
    
    def get_feature_importance(self, file_path: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get most important features (tokens) for a specific file.
        
        Args:
            file_path: Path to the file to analyze
            top_k: Number of top features to return
            
        Returns:
            List of (token, importance) tuples
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if file_path not in self.file_paths:
            raise ValueError(f"File not found in fitted corpus: {file_path}")
        
        file_idx = self.file_paths.index(file_path)
        tfidf_vector = self.tfidf_matrix[file_idx]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names()
        
        # Get top features
        if hasattr(tfidf_vector, 'toarray'):
            tfidf_vector = tfidf_vector.toarray().flatten()
        
        top_indices = np.argsort(tfidf_vector)[::-1][:top_k]
        
        important_features = []
        for idx in top_indices:
            if tfidf_vector[idx] > 0:
                important_features.append((feature_names[idx], tfidf_vector[idx]))
        
        return important_features