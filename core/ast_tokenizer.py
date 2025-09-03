"""
AST Tokenizer for extracting structural tokens from source code.
Supports Python and JavaScript through tree-sitter parsing.
"""

import ast
import re
from typing import List, Dict, Optional, Any
from pathlib import Path
import tree_sitter


class ASTTokenizer:
    def __init__(self):
        self.supported_languages = ['python', 'javascript']
        self._parsers = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            
            # Python parser
            python_language = tree_sitter.Language(tree_sitter_python.language())
            self._parsers['python'] = tree_sitter.Parser(python_language)
            
            # JavaScript parser
            js_language = tree_sitter.Language(tree_sitter_javascript.language())
            self._parsers['javascript'] = tree_sitter.Parser(js_language)
            
        except ImportError as e:
            print(f"Warning: Tree-sitter languages not available: {e}")
            self._parsers = {}
    
    def tokenize_file(self, file_path: str) -> List[str]:
        """
        Tokenize a source code file and return AST tokens.
        
        Args:
            file_path: Path to the source code file
            
        Returns:
            List of AST tokens representing the code structure
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine language from file extension
        language = self._detect_language(file_path)
        if not language:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        return self.tokenize_code(content, language)
    
    def tokenize_code(self, code: str, language: str) -> List[str]:
        """
        Tokenize source code string and return AST tokens.
        
        Args:
            code: Source code string
            language: Programming language ('python' or 'javascript')
            
        Returns:
            List of AST tokens
        """
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Preprocess code
        cleaned_code = self._preprocess_code(code)
        
        # Use appropriate tokenization method
        if language == 'python':
            return self._tokenize_python(cleaned_code)
        elif language == 'javascript':
            return self._tokenize_javascript(cleaned_code)
        
        return []
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript'
        }
        return extension_map.get(file_path.suffix.lower())
    
    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code by removing comments, docstrings, and normalizing whitespace.
        """
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # JS/C++ style
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)   # Python style
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # JS/C++ style
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)  # Python docstrings
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)  # Python docstrings
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        code = code.strip()
        
        return code
    
    def _tokenize_python(self, code: str) -> List[str]:
        """Tokenize Python code using built-in AST module"""
        tokens = []
        
        try:
            tree = ast.parse(code)
            tokens = self._extract_python_tokens(tree)
        except SyntaxError:
            # If parsing fails, try tree-sitter as fallback
            if 'python' in self._parsers:
                tokens = self._tokenize_with_tree_sitter(code, 'python')
        
        return tokens
    
    def _tokenize_javascript(self, code: str) -> List[str]:
        """Tokenize JavaScript code using tree-sitter"""
        if 'javascript' in self._parsers:
            return self._tokenize_with_tree_sitter(code, 'javascript')
        return []
    
    def _extract_python_tokens(self, node: ast.AST) -> List[str]:
        """Recursively extract tokens from Python AST"""
        tokens = []
        
        # Add node type as token
        node_type = type(node).__name__
        tokens.append(f"NODE_{node_type}")
        
        # Handle specific node types
        if isinstance(node, ast.FunctionDef):
            tokens.append("FUNC_DEF")
            # Normalize function name to generic token
            tokens.append("FUNC_NAME")
            
        elif isinstance(node, ast.ClassDef):
            tokens.append("CLASS_DEF")
            tokens.append("CLASS_NAME")
            
        elif isinstance(node, ast.Name):
            # Normalize variable names
            if isinstance(node.ctx, ast.Store):
                tokens.append("VAR_ASSIGN")
            else:
                tokens.append("VAR_USE")
                
        elif isinstance(node, ast.Constant):
            # Categorize constants by type
            if isinstance(node.value, str):
                tokens.append("STR_LITERAL")
            elif isinstance(node.value, (int, float)):
                tokens.append("NUM_LITERAL")
            elif isinstance(node.value, bool):
                tokens.append("BOOL_LITERAL")
            else:
                tokens.append("LITERAL")
                
        elif isinstance(node, ast.BinOp):
            tokens.append("BINARY_OP")
            op_name = type(node.op).__name__
            tokens.append(f"OP_{op_name}")
            
        elif isinstance(node, ast.Compare):
            tokens.append("COMPARE")
            for op in node.ops:
                op_name = type(op).__name__
                tokens.append(f"CMP_{op_name}")
        
        # Recursively process child nodes
        for child in ast.iter_child_nodes(node):
            tokens.extend(self._extract_python_tokens(child))
        
        return tokens
    
    def _tokenize_with_tree_sitter(self, code: str, language: str) -> List[str]:
        """Tokenize code using tree-sitter parser"""
        if language not in self._parsers:
            return []
        
        parser = self._parsers[language]
        tree = parser.parse(bytes(code, 'utf8'))
        
        tokens = []
        self._extract_tree_sitter_tokens(tree.root_node, tokens)
        
        return tokens
    
    def _extract_tree_sitter_tokens(self, node: Any, tokens: List[str]) -> None:
        """Recursively extract tokens from tree-sitter AST"""
        # Add node type as token
        tokens.append(f"NODE_{node.type}")
        
        # Handle specific node types for normalization
        if node.type in ['identifier', 'property_identifier']:
            tokens.append("IDENTIFIER")
        elif node.type in ['string', 'string_literal']:
            tokens.append("STR_LITERAL")
        elif node.type in ['number', 'numeric_literal']:
            tokens.append("NUM_LITERAL")
        elif node.type in ['true', 'false', 'boolean']:
            tokens.append("BOOL_LITERAL")
        elif node.type == 'function_declaration':
            tokens.append("FUNC_DEF")
        elif node.type == 'class_declaration':
            tokens.append("CLASS_DEF")
        elif 'binary' in node.type:
            tokens.append("BINARY_OP")
        
        # Recursively process children
        for child in node.children:
            self._extract_tree_sitter_tokens(child, tokens)
    
    def get_token_statistics(self, tokens: List[str]) -> Dict[str, int]:
        """Get frequency statistics for tokens"""
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        return token_counts
    
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply additional normalization to tokens"""
        normalized = []
        
        for token in tokens:
            # Group similar tokens
            if token.startswith('NODE_'):
                normalized.append(token)
            elif 'LITERAL' in token:
                normalized.append('LITERAL')
            elif 'OP_' in token or 'CMP_' in token:
                normalized.append('OPERATOR')
            else:
                normalized.append(token)
        
        return normalized