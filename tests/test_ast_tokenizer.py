"""
Test suite for AST tokenizer functionality.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ast_tokenizer import ASTTokenizer


class TestASTTokenizer:
    
    def setup_method(self):
        """Setup test environment"""
        self.tokenizer = ASTTokenizer()
    
    def test_python_tokenization(self):
        """Test Python code tokenization"""
        python_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
        """
        
        tokens = self.tokenizer.tokenize_code(python_code, 'python')
        
        assert len(tokens) > 0
        assert 'NODE_FunctionDef' in tokens
        assert 'NODE_ClassDef' in tokens
        assert 'FUNC_DEF' in tokens
        assert 'CLASS_DEF' in tokens
    
    def test_javascript_tokenization(self):
        """Test JavaScript code tokenization (if tree-sitter is available)"""
        js_code = """
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Rectangle {
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }
    
    getArea() {
        return this.width * this.height;
    }
}
        """
        
        tokens = self.tokenizer.tokenize_code(js_code, 'javascript')
        
        # If tree-sitter is available, we should get tokens
        if 'javascript' in self.tokenizer._parsers:
            assert len(tokens) > 0
        else:
            # If not available, should return empty list
            assert tokens == []
    
    def test_python_normalization(self):
        """Test token normalization for Python"""
        python_code = """
def my_function(param1, param2):
    variable1 = 42
    variable2 = "hello"
    return variable1 + len(variable2)
        """
        
        tokens = self.tokenizer.tokenize_code(python_code, 'python')
        normalized_tokens = self.tokenizer.normalize_tokens(tokens)
        
        assert len(normalized_tokens) > 0
        assert 'LITERAL' in normalized_tokens or 'STR_LITERAL' in tokens
    
    def test_token_statistics(self):
        """Test token frequency statistics"""
        python_code = """
def test():
    x = 1
    y = 2
    return x + y
        """
        
        tokens = self.tokenizer.tokenize_code(python_code, 'python')
        stats = self.tokenizer.get_token_statistics(tokens)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        assert all(isinstance(count, int) for count in stats.values())
    
    def test_empty_code(self):
        """Test tokenization of empty code"""
        tokens = self.tokenizer.tokenize_code("", 'python')
        assert tokens == []
    
    def test_syntax_error_handling(self):
        """Test handling of syntax errors in Python code"""
        invalid_python = "def invalid_function("
        
        # Should not raise exception, might return empty list or fallback tokens
        tokens = self.tokenizer.tokenize_code(invalid_python, 'python')
        assert isinstance(tokens, list)
    
    def test_unsupported_language(self):
        """Test handling of unsupported language"""
        with pytest.raises(ValueError):
            self.tokenizer.tokenize_code("some code", 'unsupported')
    
    def test_preprocess_code(self):
        """Test code preprocessing"""
        code_with_comments = """
# This is a comment
def function():  # Another comment
    '''This is a docstring'''
    x = 1  # Inline comment
    return x
        """
        
        processed = self.tokenizer._preprocess_code(code_with_comments)
        
        # Comments should be removed
        assert '#' not in processed
        # Docstrings should be removed
        assert 'docstring' not in processed
        # Code structure should remain
        assert 'def' in processed
        assert 'return' in processed
    
    def test_detect_language(self):
        """Test programming language detection"""
        assert self.tokenizer._detect_language(Path('test.py')) == 'python'
        assert self.tokenizer._detect_language(Path('test.js')) == 'javascript'
        assert self.tokenizer._detect_language(Path('test.jsx')) == 'javascript'
        assert self.tokenizer._detect_language(Path('test.ts')) == 'javascript'
        assert self.tokenizer._detect_language(Path('test.tsx')) == 'javascript'
        assert self.tokenizer._detect_language(Path('test.txt')) is None
    
    def test_complex_python_constructs(self):
        """Test tokenization of complex Python constructs"""
        complex_code = """
import math
from collections import defaultdict

@decorator
class MyClass(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = defaultdict(list)
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        if val > 0:
            self._value = val
        else:
            raise ValueError("Value must be positive")
    
    def process(self, items):
        result = [x for x in items if x % 2 == 0]
        return sum(result)

# Global variable
CONSTANT = 42

def lambda_example():
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x**2, numbers))
    return squared
        """
        
        tokens = self.tokenizer.tokenize_code(complex_code, 'python')
        
        assert len(tokens) > 0
        assert 'NODE_ClassDef' in tokens
        assert 'NODE_FunctionDef' in tokens
        assert 'CLASS_DEF' in tokens
        assert 'FUNC_DEF' in tokens
        
        # Check for various node types
        token_types = set(token for token in tokens if token.startswith('NODE_'))
        expected_types = {'NODE_FunctionDef', 'NODE_ClassDef', 'NODE_Assign'}
        assert len(token_types.intersection(expected_types)) > 0
    
    def test_binary_operations(self):
        """Test tokenization of binary operations"""
        code = """
result = a + b * c - d / e
comparison = x > y and z <= w
        """
        
        tokens = self.tokenizer.tokenize_code(code, 'python')
        
        # Should contain binary operations
        binary_tokens = [token for token in tokens if 'OP_' in token or 'BINARY' in token]
        assert len(binary_tokens) > 0