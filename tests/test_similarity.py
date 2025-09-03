"""
Test suite for cosine similarity calculator functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.similarity import CosineSimilarityCalculator


class TestCosineSimilarityCalculator:
    
    def setup_method(self):
        """Setup test environment"""
        self.calculator = CosineSimilarityCalculator()
    
    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors"""
        vector = np.array([1, 2, 3, 4])
        similarity = self.calculator.calculate_similarity(vector, vector)
        
        assert abs(similarity - 1.0) < 1e-10
    
    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors"""
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        similarity = self.calculator.calculate_similarity(vector1, vector2)
        
        assert abs(similarity - 0.0) < 1e-10
    
    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors"""
        vector1 = np.array([1, 2, 3])
        vector2 = np.array([-1, -2, -3])
        similarity = self.calculator.calculate_similarity(vector1, vector2)
        
        # Cosine similarity should be 0 due to clipping (originally would be -1)
        assert similarity == 0.0
    
    def test_zero_vectors(self):
        """Test handling of zero vectors"""
        zero_vector = np.array([0, 0, 0])
        normal_vector = np.array([1, 2, 3])
        
        similarity1 = self.calculator.calculate_similarity(zero_vector, normal_vector)
        similarity2 = self.calculator.calculate_similarity(zero_vector, zero_vector)
        
        assert similarity1 == 0.0
        assert similarity2 == 0.0
    
    def test_unequal_length_vectors(self):
        """Test error handling for vectors of different lengths"""
        vector1 = np.array([1, 2, 3])
        vector2 = np.array([1, 2])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_similarity(vector1, vector2)
    
    def test_similarity_matrix_calculation(self):
        """Test pairwise similarity matrix calculation"""
        # Create test TF-IDF matrix (3 documents, 4 features)
        tfidf_matrix = np.array([
            [0.5, 0.3, 0.2, 0.0],
            [0.3, 0.5, 0.0, 0.2],
            [0.0, 0.2, 0.5, 0.3]
        ])
        
        similarity_matrix = self.calculator.calculate_similarity_matrix(tfidf_matrix)
        
        # Check shape
        assert similarity_matrix.shape == (3, 3)
        
        # Check diagonal elements (self-similarity should be 1.0)
        np.testing.assert_allclose(np.diag(similarity_matrix), 1.0, atol=1e-10)
        
        # Check symmetry
        np.testing.assert_allclose(similarity_matrix, similarity_matrix.T, atol=1e-10)
        
        # Check value range [0, 1]
        assert np.all(similarity_matrix >= 0)
        assert np.all(similarity_matrix <= 1)
    
    def test_empty_matrix(self):
        """Test handling of empty TF-IDF matrix"""
        empty_matrix = np.array([]).reshape(0, 0)
        similarity_matrix = self.calculator.calculate_similarity_matrix(empty_matrix)
        
        assert similarity_matrix.size == 0
    
    def test_single_document_matrix(self):
        """Test similarity matrix for single document"""
        single_doc_matrix = np.array([[0.5, 0.3, 0.2]])
        similarity_matrix = self.calculator.calculate_similarity_matrix(single_doc_matrix)
        
        assert similarity_matrix.shape == (1, 1)
        assert abs(similarity_matrix[0, 0] - 1.0) < 1e-10
    
    def test_find_similar_pairs(self):
        """Test finding similar document pairs"""
        # Create similarity matrix with known values
        similarity_matrix = np.array([
            [1.0, 0.8, 0.3, 0.9],
            [0.8, 1.0, 0.2, 0.7],
            [0.3, 0.2, 1.0, 0.1],
            [0.9, 0.7, 0.1, 1.0]
        ])
        
        similar_pairs = self.calculator.find_similar_pairs(
            similarity_matrix, 
            threshold=0.75
        )
        
        # Should find pairs with similarity >= 0.75
        expected_pairs = {(0, 1, 0.8), (0, 3, 0.9)}
        found_pairs = {(i, j, sim) for i, j, sim in similar_pairs}
        
        assert len(similar_pairs) == 2
        assert found_pairs == expected_pairs
        
        # Check sorting (should be in descending order of similarity)
        similarities = [sim for _, _, sim in similar_pairs]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_get_top_k_similar(self):
        """Test getting top-k similar documents"""
        similarity_matrix = np.array([
            [1.0, 0.8, 0.3, 0.9, 0.1],
            [0.8, 1.0, 0.2, 0.7, 0.4],
            [0.3, 0.2, 1.0, 0.1, 0.9],
            [0.9, 0.7, 0.1, 1.0, 0.2],
            [0.1, 0.4, 0.9, 0.2, 1.0]
        ])
        
        # Get top 3 similar documents to document 0
        top_similar = self.calculator.get_top_k_similar(0, similarity_matrix, k=3)
        
        assert len(top_similar) == 3
        
        # Should be sorted by similarity (descending)
        similarities = [sim for _, sim in top_similar]
        assert similarities == sorted(similarities, reverse=True)
        
        # Top match should be document 3 with similarity 0.9
        assert top_similar[0] == (3, 0.9)
    
    def test_calculate_similarity_statistics(self):
        """Test similarity statistics calculation"""
        similarity_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.4],
            [0.6, 0.4, 1.0]
        ])
        
        stats = self.calculator.calculate_similarity_statistics(
            similarity_matrix, 
            exclude_diagonal=True
        )
        
        # Should calculate statistics for upper triangle (0.8, 0.6, 0.4)
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        expected_values = [0.8, 0.6, 0.4]
        assert abs(stats['mean'] - np.mean(expected_values)) < 1e-10
        assert abs(stats['median'] - np.median(expected_values)) < 1e-10
        assert abs(stats['min'] - 0.4) < 1e-10
        assert abs(stats['max'] - 0.8) < 1e-10
    
    def test_detect_clusters(self):
        """Test clustering of similar documents"""
        # Create similarity matrix with two clear clusters
        similarity_matrix = np.array([
            [1.0, 0.9, 0.1, 0.1, 0.1],  # Cluster 1: docs 0, 1
            [0.9, 1.0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8, 0.1],  # Cluster 2: docs 2, 3
            [0.1, 0.1, 0.8, 1.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0]   # Singleton: doc 4
        ])
        
        clusters = self.calculator.detect_clusters(similarity_matrix, threshold=0.8)
        
        assert len(clusters) == 2  # Two clusters (singleton is excluded)
        
        # Check that clusters contain expected documents
        cluster_sets = [set(cluster) for cluster in clusters]
        expected_clusters = [{0, 1}, {2, 3}]
        
        assert len(cluster_sets) == len(expected_clusters)
        for expected_cluster in expected_clusters:
            assert expected_cluster in cluster_sets
    
    def test_explain_similarity(self):
        """Test similarity explanation with feature contributions"""
        vector1 = np.array([0.5, 0.3, 0.0, 0.2])
        vector2 = np.array([0.4, 0.6, 0.0, 0.0])
        feature_names = ['token1', 'token2', 'token3', 'token4']
        
        explanation = self.calculator.explain_similarity(
            vector1, vector2, feature_names, top_k=3
        )
        
        assert isinstance(explanation, dict)
        assert 'similarity' in explanation
        assert 'contributing_features' in explanation
        assert 'total_features' in explanation
        assert 'non_zero_features1' in explanation
        assert 'non_zero_features2' in explanation
        
        # Check similarity calculation
        expected_similarity = self.calculator.calculate_similarity(vector1, vector2)
        assert abs(explanation['similarity'] - expected_similarity) < 1e-10
        
        # Check contributing features
        contributing_features = explanation['contributing_features']
        assert isinstance(contributing_features, list)
        assert len(contributing_features) <= 3
        
        # Features should be sorted by contribution
        if len(contributing_features) > 1:
            contributions = [f['contribution'] for f in contributing_features]
            assert contributions == sorted(contributions, reverse=True)
    
    def test_sparse_matrix_handling(self):
        """Test handling of sparse matrices"""
        from scipy.sparse import csr_matrix
        
        # Create sparse TF-IDF matrix
        dense_matrix = np.array([
            [0.5, 0.0, 0.3, 0.0],
            [0.0, 0.4, 0.0, 0.6],
            [0.2, 0.0, 0.0, 0.8]
        ])
        sparse_matrix = csr_matrix(dense_matrix)
        
        # Calculate similarity with sparse matrix support enabled
        calculator = CosineSimilarityCalculator(use_sparse=True)
        similarity_matrix = calculator.calculate_similarity_matrix(sparse_matrix)
        
        # Compare with dense calculation
        dense_calculator = CosineSimilarityCalculator(use_sparse=False)
        dense_similarity = dense_calculator.calculate_similarity_matrix(dense_matrix)
        
        np.testing.assert_allclose(similarity_matrix, dense_similarity, atol=1e-10)
    
    def test_performance_with_large_matrix(self):
        """Test performance with larger matrices"""
        # Create a larger random TF-IDF matrix
        np.random.seed(42)  # For reproducible results
        large_matrix = np.random.rand(50, 100)
        
        # Normalize rows to simulate TF-IDF vectors
        norms = np.linalg.norm(large_matrix, axis=1)
        large_matrix = large_matrix / norms[:, np.newaxis]
        
        # Should complete without errors
        similarity_matrix = self.calculator.calculate_similarity_matrix(large_matrix)
        
        assert similarity_matrix.shape == (50, 50)
        assert np.allclose(np.diag(similarity_matrix), 1.0, atol=1e-10)
        assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-10)