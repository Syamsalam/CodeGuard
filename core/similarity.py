"""
Cosine Similarity Calculator for measuring code similarity.
Implements cosine similarity with optimizations for sparse matrices.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.sparse import csr_matrix
import math


class CosineSimilarityCalculator:
    def __init__(self, use_sparse: bool = True):
        """
        Initialize Cosine Similarity Calculator.
        
        Args:
            use_sparse: Whether to use sparse matrix optimizations
        """
        self.use_sparse = use_sparse
    
    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Formula: cos(θ) = (A·B) / (||A|| × ||B||)
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have same length")
        
        # Handle zero vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate dot product and cosine similarity
        dot_product = np.dot(vector1, vector2)
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range (handle floating point errors)
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def calculate_similarity_matrix(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix for all documents.
        
        Args:
            tfidf_matrix: TF-IDF matrix of shape (n_documents, n_features)
            
        Returns:
            Similarity matrix of shape (n_documents, n_documents)
        """
        n_docs = tfidf_matrix.shape[0]
        
        if n_docs == 0:
            return np.array([])
        
        # Use optimized matrix multiplication for better performance
        if self.use_sparse and hasattr(tfidf_matrix, 'toarray'):
            # Handle sparse matrices
            similarity_matrix = self._calculate_sparse_similarity(tfidf_matrix)
        else:
            # Standard dense matrix calculation
            similarity_matrix = self._calculate_dense_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def _calculate_dense_similarity(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using dense matrix operations"""
        # Normalize vectors to unit length
        norms = np.linalg.norm(tfidf_matrix, axis=1)
        
        # Handle zero vectors
        norms[norms == 0] = 1
        normalized_matrix = tfidf_matrix / norms[:, np.newaxis]
        
        # Calculate similarity matrix using matrix multiplication
        similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        # Ensure diagonal is 1.0 and handle floating point errors
        np.fill_diagonal(similarity_matrix, 1.0)
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        return similarity_matrix
    
    def _calculate_sparse_similarity(self, sparse_matrix) -> np.ndarray:
        """Calculate similarity matrix for sparse TF-IDF matrix"""
        # Convert to CSR format for efficient operations
        if not isinstance(sparse_matrix, csr_matrix):
            sparse_matrix = csr_matrix(sparse_matrix)
        
        # Normalize rows to unit length
        normalized_matrix = self._normalize_sparse_matrix(sparse_matrix)
        
        # Calculate similarity matrix
        similarity_matrix = normalized_matrix.dot(normalized_matrix.T)
        
        # Convert back to dense array
        if hasattr(similarity_matrix, 'toarray'):
            similarity_matrix = similarity_matrix.toarray()
        
        # Ensure diagonal is 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        return similarity_matrix
    
    def _normalize_sparse_matrix(self, sparse_matrix):
        """Normalize rows of sparse matrix to unit length"""
        # Calculate row norms
        row_norms = np.sqrt(np.array(sparse_matrix.multiply(sparse_matrix).sum(axis=1)).flatten())
        
        # Handle zero norms
        row_norms[row_norms == 0] = 1
        
        # Create diagonal matrix of inverse norms
        inv_norms = 1.0 / row_norms
        norm_diag = csr_matrix((inv_norms, (range(len(inv_norms)), range(len(inv_norms)))),
                               shape=(len(inv_norms), len(inv_norms)))
        
        # Normalize the matrix
        normalized = norm_diag.dot(sparse_matrix)
        
        return normalized
    
    def find_similar_pairs(self, similarity_matrix: np.ndarray, 
                          threshold: float = 0.7, 
                          exclude_diagonal: bool = True) -> List[Tuple[int, int, float]]:
        """
        Find pairs of documents with similarity above threshold.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Minimum similarity threshold
            exclude_diagonal: Whether to exclude self-similarities
            
        Returns:
            List of (doc1_idx, doc2_idx, similarity) tuples sorted by similarity
        """
        similar_pairs = []
        n_docs = similarity_matrix.shape[0]
        
        for i in range(n_docs):
            start_j = i + 1 if exclude_diagonal else i
            for j in range(start_j, n_docs):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))
        
        # Sort by similarity score descending
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def get_top_k_similar(self, document_idx: int, similarity_matrix: np.ndarray, 
                         k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k most similar documents to a given document.
        
        Args:
            document_idx: Index of the reference document
            similarity_matrix: Pairwise similarity matrix
            k: Number of similar documents to return
            
        Returns:
            List of (doc_idx, similarity) tuples sorted by similarity
        """
        if document_idx >= similarity_matrix.shape[0]:
            raise ValueError("Document index out of bounds")
        
        similarities = similarity_matrix[document_idx].copy()
        
        # Exclude self-similarity
        similarities[document_idx] = -1
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_similar = [(idx, similarities[idx]) for idx in top_indices 
                      if similarities[idx] > 0]
        
        return top_similar
    
    def calculate_similarity_statistics(self, similarity_matrix: np.ndarray, 
                                      exclude_diagonal: bool = True) -> Dict[str, float]:
        """
        Calculate statistical measures for similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            exclude_diagonal: Whether to exclude diagonal elements
            
        Returns:
            Dictionary with statistical measures
        """
        if similarity_matrix.size == 0:
            return {}
        
        if exclude_diagonal:
            # Extract upper triangle excluding diagonal
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            similarities = similarity_matrix[mask]
        else:
            similarities = similarity_matrix.flatten()
        
        if len(similarities) == 0:
            return {}
        
        stats = {
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75))
        }
        
        return stats
    
    def detect_clusters(self, similarity_matrix: np.ndarray, 
                       threshold: float = 0.8) -> List[List[int]]:
        """
        Detect clusters of highly similar documents.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Minimum similarity to consider documents in same cluster
            
        Returns:
            List of clusters (each cluster is a list of document indices)
        """
        n_docs = similarity_matrix.shape[0]
        
        if n_docs == 0:
            return []
        
        # Create adjacency matrix for similarities above threshold
        adjacency = (similarity_matrix >= threshold).astype(int)
        
        # Find connected components (clusters)
        visited = [False] * n_docs
        clusters = []
        
        for i in range(n_docs):
            if not visited[i]:
                cluster = []
                self._dfs_cluster(i, adjacency, visited, cluster)
                if len(cluster) > 1:  # Only include clusters with multiple documents
                    clusters.append(cluster)
        
        return clusters
    
    def _dfs_cluster(self, node: int, adjacency: np.ndarray, 
                    visited: List[bool], cluster: List[int]) -> None:
        """Depth-first search for finding connected components"""
        visited[node] = True
        cluster.append(node)
        
        for neighbor in range(len(adjacency)):
            if adjacency[node][neighbor] and not visited[neighbor]:
                self._dfs_cluster(neighbor, adjacency, visited, cluster)
    
    def explain_similarity(self, vector1: np.ndarray, vector2: np.ndarray, 
                          feature_names: List[str], top_k: int = 10) -> Dict[str, any]:
        """
        Explain similarity by showing contributing features.
        
        Args:
            vector1: First TF-IDF vector
            vector2: Second TF-IDF vector
            feature_names: Names of features corresponding to vector indices
            top_k: Number of top contributing features to show
            
        Returns:
            Dictionary with similarity explanation
        """
        similarity = self.calculate_similarity(vector1, vector2)
        
        # Calculate feature contributions (element-wise product)
        contributions = vector1 * vector2
        
        # Get top contributing features
        top_indices = np.argsort(contributions)[::-1][:top_k]
        
        contributing_features = []
        for idx in top_indices:
            if contributions[idx] > 0 and idx < len(feature_names):
                contributing_features.append({
                    'feature': feature_names[idx],
                    'contribution': float(contributions[idx]),
                    'weight1': float(vector1[idx]),
                    'weight2': float(vector2[idx])
                })
        
        return {
            'similarity': similarity,
            'contributing_features': contributing_features,
            'total_features': len(feature_names),
            'non_zero_features1': int(np.count_nonzero(vector1)),
            'non_zero_features2': int(np.count_nonzero(vector2))
        }