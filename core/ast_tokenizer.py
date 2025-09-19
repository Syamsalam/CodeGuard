"""
AST Tokenizer for extracting structural tokens from source code.
Supports Python and JavaScript through tree-sitter parsing.
"""

import ast
import re
from typing import List, Dict, Optional, Any, Set
import hashlib
from pathlib import Path
import tree_sitter


class ASTTokenizer:
    def __init__(self):
        self.supported_languages = ['python', 'javascript']
        self._parsers = {}
        self._initialize_parsers()
        # Pre-computed sets for filtering
        import keyword
        self.python_keywords: Set[str] = set(keyword.kwlist)
        # Generic operator & punctuation tokens (will match normalization outputs like OP_, CMP_, etc.)
        self.operator_markers: Set[str] = {"OPERATOR", "BINARY_OP", "COMPARE"}
        self.literal_markers: Set[str] = {"LITERAL", "STR_LITERAL", "NUM_LITERAL", "BOOL_LITERAL"}
    
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
        
        # Early return for empty code
        if not code.strip():
            return []

        # Preprocess code (preserve newlines for Python to keep AST valid)
        cleaned_code = self._preprocess_code(code)
        
        # Use appropriate tokenization method
        if language == 'python':
            return self._tokenize_python(cleaned_code)
        elif language == 'javascript':
            return self._tokenize_javascript(cleaned_code)
        
        return []

    # ------------------------------------------------------------------
    # AST Path N-gram (Shingles) Generation
    # ------------------------------------------------------------------
    def generate_path_ngrams(self,
                              code: str,
                              language: str,
                              n_min: int = 2,
                              n_max: int = 4,
                              use_hash: bool = True,
                              hash_len: int = 8) -> List[str]:
        """Generate AST path n-gram tokens to enrich structural context.

        Each root-to-node path (sequence of node type names) contributes
        sliding window n-grams (n in [n_min, n_max]).

        Tokens are emitted as either:
            PATH{n}_{hash}
        or (if use_hash False)
            PATH{n}_Type1>Type2>Type3  (sanitized, truncated if very long)

        Args:
            code: Source code string
            language: 'python' or 'javascript'
            n_min: Minimum n-gram length (>=2)
            n_max: Maximum n-gram length (>= n_min)
            use_hash: Hash path string to compact vocabulary
            hash_len: Length of hex digest to retain when hashing

        Returns:
            List of path n-gram token strings.
        """
        if n_min < 2:
            n_min = 2
        if n_max < n_min:
            n_max = n_min
        if not code.strip():
            return []
        if language not in ('python','javascript'):
            return []
        paths: List[List[str]] = []
        try:
            if language == 'python':
                import ast as _pyast
                tree = _pyast.parse(code)
                # DFS collecting paths
                def _walk(node, current):
                    node_type = type(node).__name__
                    new_path = current + [node_type]
                    paths.append(new_path)
                    for child in _pyast.iter_child_nodes(node):
                        _walk(child, new_path)
                _walk(tree, [])
            else:
                # Use tree-sitter for JS if available
                if 'javascript' not in self._parsers:
                    return []
                parser = self._parsers['javascript']
                t = parser.parse(bytes(code,'utf8'))
                def _walk_ts(node, current):
                    node_type = node.type
                    new_path = current + [node_type]
                    paths.append(new_path)
                    for ch in node.children:
                        _walk_ts(ch, new_path)
                _walk_ts(t.root_node, [])
        except Exception:
            return []

        tokens: List[str] = []
        for p in paths:
            L = len(p)
            if L < n_min:
                continue
            for n in range(n_min, min(n_max, L)+1):
                # sliding windows
                for i in range(0, L-n+1):
                    window = p[i:i+n]
                    path_str = '>'.join(window)
                    if use_hash:
                        h = hashlib.sha1(path_str.encode('utf-8')).hexdigest()[:hash_len]
                        tokens.append(f"PATH{n}_{h}")
                    else:
                        # Sanitize and truncate overlong tokens
                        sanitized = path_str.replace(' ','_')
                        if len(sanitized) > 60:
                            sanitized = sanitized[:60]
                        tokens.append(f"PATH{n}_{sanitized}")
        return tokens
    
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
        # Remove single-line comments (keep newlines)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

        # Remove multi-line comments / docstrings
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Do NOT collapse all whitespace to single spaces to preserve indentation for Python
        return code.strip('\n')
    
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
    
    def normalize_tokens(self, tokens: List[str], keep_identifier_detail: bool = False) -> List[str]:
        """Apply normalization.

        Args:
            tokens: Raw extracted tokens
            keep_identifier_detail: When True, do NOT collapse VAR_ASSIGN vs VAR_USE etc. and keep OP_/CMP_ granular.
                This helps differentiate structurally similar but semantically different code.
        """
        normalized: List[str] = []
        for token in tokens:
            if keep_identifier_detail:
                # Only collapse obvious noise (node types) but keep operator specificity
                if token.startswith('NODE_'):
                    normalized.append(token)
                else:
                    normalized.append(token)
                continue
            # Default aggressive normalization
            if token.startswith('NODE_'):
                normalized.append(token)
            elif 'LITERAL' in token:
                normalized.append('LITERAL')
            elif 'OP_' in token or 'CMP_' in token:
                normalized.append('OPERATOR')
            else:
                normalized.append(token)
        return normalized

    # ------------------------------------------------------------------
    # Filtering utilities
    # ------------------------------------------------------------------
    def filter_tokens(self,
                      tokens: List[str],
                      remove_node_tokens: bool = False,
                      remove_literals: bool = False,
                      remove_operators: bool = False,
                      remove_keywords: bool = False,
                      min_token_length: int = 0) -> List[str]:
        """Filter token list based on several configurable criteria.

        Args:
            tokens: List of tokens (already normalized or raw)
            remove_node_tokens: Drop tokens starting with 'NODE_'
            remove_literals: Drop literal related markers
            remove_operators: Drop operator markers
            remove_keywords: Drop python keywords (exact match, lowercase)
            min_token_length: Keep tokens whose length >= this value
        """
        filtered: List[str] = []
        for tok in tokens:
            if remove_node_tokens and tok.startswith('NODE_'):
                continue
            if remove_literals and (tok in self.literal_markers):
                continue
            if remove_operators and (tok in self.operator_markers or tok.startswith('OP_') or tok.startswith('CMP_')):
                continue
            if remove_keywords and tok.lower() in self.python_keywords:
                continue
            if min_token_length and len(tok) < min_token_length:
                continue
            filtered.append(tok)
        return filtered