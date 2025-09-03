"""
Test suite for TF-IDF vectorizer functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tfidf_vectorizer import TFIDFVectorizer


class TestTFIDFVectorizer:
    
    def setup_method(self):
        """Setup test environment"""
        self.vectorizer = TFIDFVectorizer()
        
        # Sample token documents for testing
        self.sample_documents = [
            ['function', 'def', 'return', 'if', 'variable'],
            ['function', 'class', 'method', 'return', 'variable'],
            ['def', 'return', 'for', 'loop', 'variable'],
            ['function', 'method', 'class', 'object', 'return']
        ]
    
    def test_fit_transform(self):
        """Test fit and transform functionality"""
        tfidf_matrix = self.vectorizer.fit_transform(self.sample_documents)
        
        assert tfidf_matrix.shape[0] == len(self.sample_documents)
        assert tfidf_matrix.shape[1] > 0
        assert isinstance(tfidf_matrix, np.ndarray)
        
        # Check that the vectorizer is fitted
        assert self.vectorizer._fitted
        assert len(self.vectorizer.vocabulary_) > 0
        assert len(self.vectorizer.idf_weights_) > 0
    
    def test_vocabulary_creation(self):
        """Test vocabulary building"""
        self.vectorizer.fit(self.sample_documents)
        
        vocabulary = self.vectorizer.get_vocabulary()
        feature_names = self.vectorizer.get_feature_names()
        
        assert isinstance(vocabulary, dict)
        assert isinstance(feature_names, list)
        assert len(vocabulary) == len(feature_names)
        
        # Check that common tokens are in vocabulary
        expected_tokens = {'function', 'def', 'return', 'variable'}
        assert expected_tokens.issubset(set(vocabulary.keys()))
    
    def test_idf_calculation(self):
        """Test IDF weight calculation"""
        self.vectorizer.fit(self.sample_documents)
        
        idf_weights = self.vectorizer.get_idf_weights()
        
        assert isinstance(idf_weights, dict)
        assert len(idf_weights) > 0
        assert all(weight > 0 for weight in idf_weights.values())
        
        # Terms appearing in fewer documents should have higher IDF
        # 'function' appears in 3/4 documents, 'loop' appears in 1/4
        if 'function' in idf_weights and 'loop' in idf_weights:
            assert idf_weights['loop'] > idf_weights['function']
    
    def test_tf_calculation(self):
        """Test term frequency calculation"""
        document = ['function', 'function', 'def', 'return']
        tf_scores = self.vectorizer._calculate_tf_scores(document)
        
        assert isinstance(tf_scores, dict)
        assert tf_scores['function'] == 0.5  # 2 out of 4 tokens
        assert tf_scores['def'] == 0.25  # 1 out of 4 tokens
        assert tf_scores['return'] == 0.25  # 1 out of 4 tokens
    
    def test_normalization(self):
        """Test vector normalization"""
        self.vectorizer.normalize = True
        tfidf_matrix = self.vectorizer.fit_transform(self.sample_documents)
        
        # Check that vectors are approximately unit length
        norms = np.linalg.norm(tfidf_matrix, axis=1)
        
        # Allow for small numerical errors
        for norm in norms:
            if norm > 0:  # Skip zero vectors
                assert abs(norm - 1.0) < 1e-10
    
    def test_min_df_filtering(self):
        """Test minimum document frequency filtering"""
        # Use min_df=2 to filter out rare terms
        vectorizer = TFIDFVectorizer(min_df=2)
        vectorizer.fit(self.sample_documents)
        
        vocabulary = vectorizer.get_vocabulary()
        
        # Terms appearing in only 1 document should be filtered out
        # 'loop' and 'object' appear only once
        assert 'loop' not in vocabulary
        assert 'object' not in vocabulary
        
        # Terms appearing in 2+ documents should remain
        assert 'function' in vocabulary
        assert 'return' in vocabulary
    
    def test_max_df_filtering(self):
        """Test maximum document frequency filtering"""
        # Use max_df=0.5 to filter out very common terms
        vectorizer = TFIDFVectorizer(max_df=0.5)  # 50% of documents
        vectorizer.fit(self.sample_documents)
        
        vocabulary = vectorizer.get_vocabulary()
        
        # 'return' appears in all 4 documents (100% > 50%)
        assert 'return' not in vocabulary
        
        # Less common terms should remain
        assert 'loop' in vocabulary
    
    def test_empty_documents(self):
        """Test handling of empty documents"""
        empty_docs = [[], ['token'], []]
        
        tfidf_matrix = self.vectorizer.fit_transform(empty_docs)
        
        assert tfidf_matrix.shape[0] == len(empty_docs)
        # Empty documents should result in zero vectors
        assert np.allclose(tfidf_matrix[0], 0)
        assert np.allclose(tfidf_matrix[2], 0)
    
    def test_single_document(self):
        """Test handling of single document"""
        single_doc = [['function', 'def', 'return']]
        
        tfidf_matrix = self.vectorizer.fit_transform(single_doc)
        
        assert tfidf_matrix.shape[0] == 1
        assert tfidf_matrix.shape[1] == 3  # 3 unique tokens
        
        # All IDF weights should be log(1/1) = 0, so TF-IDF should equal TF
        expected_tf = 1.0 / 3.0  # Each token appears once out of 3 total
        assert np.allclose(tfidf_matrix[0], expected_tf)
    
    def test_transform_new_documents(self):
        """Test transforming new documents with fitted vectorizer"""
        self.vectorizer.fit(self.sample_documents)
        
        new_documents = [
            ['function', 'new_token', 'def'],
            ['return', 'variable']
        ]
        
        tfidf_matrix = self.vectorizer.transform(new_documents)
        
        assert tfidf_matrix.shape[0] == len(new_documents)
        assert tfidf_matrix.shape[1] == len(self.vectorizer.vocabulary_)
        
        # 'new_token' is not in vocabulary, so should be ignored
        # Only known tokens should contribute to the vectors
    
    def test_unfitted_transform(self):
        """Test that transform raises error when not fitted"""
        with pytest.raises(ValueError):
            self.vectorizer.transform(self.sample_documents)
    
    def test_lowercase_option(self):
        """Test lowercase conversion option"""
        vectorizer = TFIDFVectorizer(lowercase=True)
        
        mixed_case_docs = [
            ['Function', 'DEF', 'Return'],
            ['FUNCTION', 'def', 'return']
        ]
        
        tfidf_matrix = vectorizer.fit_transform(mixed_case_docs)
        vocabulary = vectorizer.get_vocabulary()
        
        # All tokens should be lowercase in vocabulary
        assert 'function' in vocabulary
        assert 'def' in vocabulary
        assert 'return' in vocabulary
        
        # Original case versions should not exist
        assert 'Function' not in vocabulary
        assert 'DEF' not in vocabulary
        assert 'Return' not in vocabulary
    
    def test_get_top_features(self):
        """Test getting top TF-IDF features for a document"""
        self.vectorizer.fit(self.sample_documents)
        
        document_tokens = ['function', 'function', 'rare_token', 'def']
        top_features = self.vectorizer.get_top_features(document_tokens, top_k=3)
        
        assert isinstance(top_features, list)
        assert len(top_features) <= 3
        
        # Each feature should be a (token, score) tuple
        for token, score in top_features:
            assert isinstance(token, str)
            assert isinstance(score, float)
            assert score > 0
        
        # Features should be sorted by score (descending)
        if len(top_features) > 1:
            scores = [score for _, score in top_features]
            assert scores == sorted(scores, reverse=True)
    
    def test_explain_similarity(self):
        """Test similarity explanation between documents"""
        self.vectorizer.fit(self.sample_documents)
        
        doc1_tokens = ['function', 'def', 'return']
        doc2_tokens = ['function', 'class', 'return']
        
        explanation = self.vectorizer.explain_similarity(doc1_tokens, doc2_tokens)
        
        assert isinstance(explanation, dict)
        assert 'similarity' in explanation
        assert 'contributing_terms' in explanation
        assert 'doc1_length' in explanation
        assert 'doc2_length' in explanation
        assert 'common_terms' in explanation
        
        assert 0 <= explanation['similarity'] <= 1
        assert explanation['doc1_length'] == len(doc1_tokens)
        assert explanation['doc2_length'] == len(doc2_tokens)
        assert explanation['common_terms'] >= 0
    
    def test_model_serialization(self):
        """Test saving and loading model"""
        import tempfile
        import os
        
        # Fit the vectorizer
        self.vectorizer.fit(self.sample_documents)
        original_vocabulary = self.vectorizer.get_vocabulary()
        original_idf = self.vectorizer.get_idf_weights()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self.vectorizer.save_model(tmp_file.name)
            
            # Create new vectorizer and load
            new_vectorizer = TFIDFVectorizer()
            new_vectorizer.load_model(tmp_file.name)
            
            # Compare loaded model
            assert new_vectorizer._fitted
            assert new_vectorizer.get_vocabulary() == original_vocabulary
            assert new_vectorizer.get_idf_weights() == original_idf
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Empty document list
        with pytest.raises(ValueError):
            self.vectorizer.fit([])
        
        # Document with repeated tokens
        repeated_docs = [['token', 'token', 'token']]
        tfidf_matrix = self.vectorizer.fit_transform(repeated_docs)
        
        # Should handle gracefully
        assert tfidf_matrix.shape[0] == 1
        assert tfidf_matrix.shape[1] == 1  # Only one unique token