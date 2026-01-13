"""
AST Tokenizer for extracting structural tokens from source code.
Supports Python and JavaScript through tree-sitter parsing.
"""

import ast
import textwrap
import re
from typing import List, Dict, Optional, Any, Set
import hashlib
from pathlib import Path
import tree_sitter


class ASTTokenizer:
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'typescript']
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
            # Optional TypeScript
            try:
                import tree_sitter_typescript
            except ImportError:
                tree_sitter_typescript = None

            # Python parser
            python_language = tree_sitter.Language(tree_sitter_python.language())
            self._parsers['python'] = tree_sitter.Parser(python_language)

            # JavaScript parser
            js_language = tree_sitter.Language(tree_sitter_javascript.language())
            self._parsers['javascript'] = tree_sitter.Parser(js_language)

            # TypeScript parser (if available)
            if tree_sitter_typescript is not None:
                try:
                    ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
                    self._parsers['typescript'] = tree_sitter.Parser(ts_lang)
                except Exception:
                    try:
                        ts_lang = tree_sitter.Language(tree_sitter_typescript.language())
                        self._parsers['typescript'] = tree_sitter.Parser(ts_lang)
                    except Exception:
                        pass
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
        elif language == 'typescript':
            return self._tokenize_typescript(cleaned_code)
        
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
        if language not in ('python','javascript','typescript'):
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
                # Use tree-sitter for JS/TS if available
                lang_key = language if language in self._parsers else ('javascript' if 'javascript' in self._parsers else None)
                if not lang_key:
                    return []
                parser = self._parsers[lang_key]
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
            '.ts': 'typescript',
            '.tsx': 'typescript'
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

        # Dedent to avoid leading indentation breaking Python AST parsing
        # (common when code is embedded in triple-quoted strings)
        try:
            code = textwrap.dedent(code)
        except Exception:
            pass

        # Do NOT collapse all whitespace to single spaces to preserve indentation for Python
        return code.strip('\n')

    def is_valid_python_syntax(self, code: str) -> bool:
        """Quickly validate Python syntax after dedent.

        Returns True if ast.parse succeeds, otherwise False.
        """
        try:
            code_d = textwrap.dedent(code)
            ast.parse(code_d)
            return True
        except Exception:
            return False
    
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
        """Tokenize JavaScript code using tree-sitter or lightweight fallback"""
        if 'javascript' in self._parsers:
            return self._tokenize_with_tree_sitter(code, 'javascript')
        return self._tokenize_js_like_fallback(code)

    def _tokenize_typescript(self, code: str) -> List[str]:
        """Tokenize TypeScript using tree-sitter if available, otherwise fallback"""
        if 'typescript' in self._parsers:
            return self._tokenize_with_tree_sitter(code, 'typescript')
        return self._tokenize_js_like_fallback(code)

    def _tokenize_js_like_fallback(self, code: str) -> List[str]:
        """Lightweight regex-based tokenization for JS/TS when parser unavailable."""
        tokens: List[str] = []
        # Function and class definitions
        tokens += ["FUNC_DEF"] * len(re.findall(r"\bfunction\b|=>", code))
        tokens += ["CLASS_DEF"] * len(re.findall(r"\bclass\b", code))
        # Identifiers (rough)
        tokens += ["IDENTIFIER"] * len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code))
        # Literals
        tokens += ["NUM_LITERAL"] * len(re.findall(r"\b\d+(?:\.\d+)?\b", code))
        tokens += ["STR_LITERAL"] * len(re.findall(r"(['\"]).*?\1", code))
        tokens += ["BOOL_LITERAL"] * len(re.findall(r"\btrue\b|\bfalse\b", code))
        # Operators
        tokens += ["BINARY_OP"] * len(re.findall(r"[+\-*/=]+", code))
        # Control flow keywords (add as plain markers)
        for kw in ["if","else","for","while","return","switch","case","break","continue","try","catch","finally","import","from","export","async","await","let","const","var"]:
            tokens.append(kw)
        # Rough node markers for visual similarity
        tokens += ["NODE_FunctionDef"] * len(re.findall(r"\bfunction\b", code))
        tokens += ["NODE_ClassDef"] * len(re.findall(r"\bclass\b", code))
        return tokens
    
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
        self._extract_tree_sitter_tokens(tree.root_node, tokens, language)
        
        return tokens
    
    def _extract_tree_sitter_tokens(self, node: Any, tokens: List[str], language: str) -> None:
        """Recursively extract tokens from tree-sitter AST with light harmonization.

        For Python grammar, harmonize common node names to Python AST-style labels
        to reduce vocabulary drift (e.g., module->Module, function_definition->FunctionDef).
        """
        raw_type = node.type
        mapped_type = raw_type
        if language == 'python':
            py_map = {
                'module': 'Module',
                'function_definition': 'FunctionDef',
                'class_definition': 'ClassDef',
                'parameters': 'arguments',
                'argument_list': 'arguments',
                'return_statement': 'Return',
                'binary_operator': 'BinOp',
                'comparison_operator': 'Compare',
                'call': 'Call',
                'identifier': 'Name',
                'attribute': 'Attribute',
            }
            mapped_type = py_map.get(raw_type, raw_type)
        # Add node token
        tokens.append(f"NODE_{mapped_type}")

        # Handle specific node types for normalization
        if raw_type in ['identifier', 'property_identifier']:
            # Tree-sitter cannot infer VAR_USE/VAR_ASSIGN without context; use generic
            tokens.append("IDENTIFIER")
        elif raw_type in ['string', 'string_literal']:
            tokens.append("STR_LITERAL")
        elif raw_type in ['number', 'numeric_literal']:
            tokens.append("NUM_LITERAL")
        elif raw_type in ['true', 'false', 'boolean']:
            tokens.append("BOOL_LITERAL")
        # Function/class markers for JS + Python TS
        if raw_type in ['function_declaration', 'function_definition']:
            tokens.append("FUNC_DEF")
        if raw_type in ['class_declaration', 'class_definition']:
            tokens.append("CLASS_DEF")
        # Operator markers
        if 'binary' in raw_type:
            tokens.append("BINARY_OP")

        # Recurse
        for child in node.children:
            self._extract_tree_sitter_tokens(child, tokens, language)
    
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