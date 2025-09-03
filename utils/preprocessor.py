"""
Code preprocessing utilities for normalizing source code before analysis.
Handles comment removal, variable normalization, and style standardization.
"""

import re
import ast
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import hashlib


class CodePreprocessor:
    def __init__(self):
        """Initialize code preprocessor with default settings"""
        self.generic_names = {
            'var': 'VAR',
            'func': 'FUNC',
            'class': 'CLASS',
            'method': 'METHOD',
            'param': 'PARAM'
        }
        
        # Common boilerplate patterns to ignore
        self.boilerplate_patterns = [
            r'import\s+\w+',
            r'from\s+\w+\s+import\s+\w+',
            r'require\(\s*[\'\"]\w+[\'\"]\s*\)',
            r'console\.log\s*\(',
            r'print\s*\(',
            r'if\s+__name__\s*==\s*[\'\"__main__\'\"]'
        ]
    
    def preprocess_python(self, code: str, normalize_names: bool = True) -> str:
        """
        Preprocess Python code by removing comments and normalizing.
        
        Args:
            code: Python source code
            normalize_names: Whether to normalize variable/function names
            
        Returns:
            Preprocessed code
        """
        # Remove comments and docstrings
        code = self._remove_python_comments(code)
        
        # Remove blank lines and normalize whitespace
        code = self._normalize_whitespace(code)
        
        # Normalize names if requested
        if normalize_names:
            code = self._normalize_python_names(code)
        
        return code
    
    def preprocess_javascript(self, code: str, normalize_names: bool = True) -> str:
        """
        Preprocess JavaScript code by removing comments and normalizing.
        
        Args:
            code: JavaScript source code
            normalize_names: Whether to normalize variable/function names
            
        Returns:
            Preprocessed code
        """
        # Remove comments
        code = self._remove_js_comments(code)
        
        # Remove blank lines and normalize whitespace
        code = self._normalize_whitespace(code)
        
        # Normalize names if requested
        if normalize_names:
            code = self._normalize_js_names(code)
        
        return code
    
    def preprocess_file(self, file_path: str, normalize_names: bool = True) -> str:
        """
        Preprocess a source code file based on its extension.
        
        Args:
            file_path: Path to the source file
            normalize_names: Whether to normalize names
            
        Returns:
            Preprocessed code
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                code = f.read()
        
        # Determine preprocessing method based on extension
        extension = file_path.suffix.lower()
        
        if extension == '.py':
            return self.preprocess_python(code, normalize_names)
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            return self.preprocess_javascript(code, normalize_names)
        else:
            # Generic preprocessing for other languages
            return self._generic_preprocess(code, normalize_names)
    
    def remove_boilerplate(self, code: str) -> str:
        """
        Remove common boilerplate code patterns.
        
        Args:
            code: Source code
            
        Returns:
            Code with boilerplate removed
        """
        lines = code.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                continue
            
            # Check against boilerplate patterns
            is_boilerplate = False
            for pattern in self.boilerplate_patterns:
                if re.match(pattern, stripped_line, re.IGNORECASE):
                    is_boilerplate = True
                    break
            
            if not is_boilerplate:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def normalize_string_literals(self, code: str) -> str:
        """
        Normalize string literals to generic placeholders.
        
        Args:
            code: Source code
            
        Returns:
            Code with normalized string literals
        """
        # Replace string literals with generic placeholder
        # Handle both single and double quotes
        code = re.sub(r'"[^"]*"', '"STRING_LITERAL"', code)
        code = re.sub(r"'[^']*'", "'STRING_LITERAL'", code)
        
        # Handle template literals (JavaScript)
        code = re.sub(r'`[^`]*`', '`STRING_LITERAL`', code)
        
        return code
    
    def normalize_numeric_literals(self, code: str) -> str:
        """
        Normalize numeric literals to generic placeholders.
        
        Args:
            code: Source code
            
        Returns:
            Code with normalized numeric literals
        """
        # Replace integers
        code = re.sub(r'\b\d+\b', 'NUM_LITERAL', code)
        
        # Replace floats
        code = re.sub(r'\b\d+\.\d+\b', 'NUM_LITERAL', code)
        
        # Replace scientific notation
        code = re.sub(r'\b\d+\.?\d*e[+-]?\d+\b', 'NUM_LITERAL', code, flags=re.IGNORECASE)
        
        return code
    
    def extract_function_signatures(self, code: str, language: str) -> List[Dict[str, str]]:
        """
        Extract function signatures from code.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of function signature dictionaries
        """
        signatures = []
        
        if language == 'python':
            signatures = self._extract_python_functions(code)
        elif language in ['javascript', 'typescript']:
            signatures = self._extract_js_functions(code)
        
        return signatures
    
    def calculate_code_metrics(self, code: str) -> Dict[str, int]:
        """
        Calculate basic code metrics.
        
        Args:
            code: Source code
            
        Returns:
            Dictionary with code metrics
        """
        lines = code.split('\n')
        
        metrics = {
            'total_lines': len(lines),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'comment_lines': 0,
            'code_lines': 0,
            'total_chars': len(code),
            'indentation_levels': 0
        }
        
        # Count comment lines and code lines
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            elif stripped.startswith('#') or stripped.startswith('//'):
                metrics['comment_lines'] += 1
            else:
                metrics['code_lines'] += 1
        
        # Calculate average indentation
        indentations = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if indentations:
            metrics['avg_indentation'] = sum(indentations) / len(indentations)
            metrics['max_indentation'] = max(indentations)
        else:
            metrics['avg_indentation'] = 0
            metrics['max_indentation'] = 0
        
        return metrics
    
    def _remove_python_comments(self, code: str) -> str:
        """Remove Python comments and docstrings"""
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        return code
    
    def _remove_js_comments(self, code: str) -> str:
        """Remove JavaScript comments"""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code"""
        # Replace multiple whitespace with single space
        code = re.sub(r'[ \t]+', ' ', code)
        
        # Remove trailing whitespace
        code = re.sub(r' +$', '', code, flags=re.MULTILINE)
        
        # Remove multiple blank lines
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        return code.strip()
    
    def _normalize_python_names(self, code: str) -> str:
        """Normalize Python variable and function names"""
        try:
            tree = ast.parse(code)
            name_mapping = {}
            counter = {'var': 0, 'func': 0, 'class': 0}
            
            # First pass: collect all names
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name not in name_mapping:
                        counter['func'] += 1
                        name_mapping[node.name] = f"func_{counter['func']}"
                
                elif isinstance(node, ast.ClassDef):
                    if node.name not in name_mapping:
                        counter['class'] += 1
                        name_mapping[node.name] = f"class_{counter['class']}"
                
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if node.id not in name_mapping and not node.id.startswith('_'):
                        counter['var'] += 1
                        name_mapping[node.id] = f"var_{counter['var']}"
            
            # Second pass: replace names in code
            for old_name, new_name in name_mapping.items():
                # Use word boundaries to avoid partial replacements
                code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, code)
            
        except SyntaxError:
            # If parsing fails, use regex-based normalization
            code = self._regex_normalize_names(code, 'python')
        
        return code
    
    def _normalize_js_names(self, code: str) -> str:
        """Normalize JavaScript variable and function names"""
        # Regex-based normalization for JavaScript
        return self._regex_normalize_names(code, 'javascript')
    
    def _regex_normalize_names(self, code: str, language: str) -> str:
        """Fallback regex-based name normalization"""
        counter = {'var': 0, 'func': 0}
        name_mapping = {}
        
        if language == 'python':
            # Python function definitions
            func_pattern = r'def\s+(\w+)\s*\('
            var_pattern = r'(\w+)\s*='
        else:
            # JavaScript function definitions
            func_pattern = r'function\s+(\w+)\s*\('
            var_pattern = r'(?:var|let|const)\s+(\w+)\s*='
        
        # Find and replace function names
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            if func_name not in name_mapping:
                counter['func'] += 1
                name_mapping[func_name] = f"func_{counter['func']}"
        
        # Find and replace variable names
        for match in re.finditer(var_pattern, code):
            var_name = match.group(1)
            if var_name not in name_mapping and not var_name.startswith('_'):
                counter['var'] += 1
                name_mapping[var_name] = f"var_{counter['var']}"
        
        # Apply replacements
        for old_name, new_name in name_mapping.items():
            code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, code)
        
        return code
    
    def _generic_preprocess(self, code: str, normalize_names: bool = True) -> str:
        """Generic preprocessing for unsupported languages"""
        # Remove common comment styles
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # C++ style
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # C style
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python/shell style
        
        # Normalize whitespace
        code = self._normalize_whitespace(code)
        
        return code
    
    def _extract_python_functions(self, code: str) -> List[Dict[str, str]]:
        """Extract Python function signatures"""
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function info
                    args = [arg.arg for arg in node.args.args]
                    
                    functions.append({
                        'name': node.name,
                        'args': args,
                        'arg_count': len(args),
                        'line_no': node.lineno
                    })
        
        except SyntaxError:
            pass
        
        return functions
    
    def _extract_js_functions(self, code: str) -> List[Dict[str, str]]:
        """Extract JavaScript function signatures using regex"""
        functions = []
        
        # Regular function declarations
        func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)'
        
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            params_str = match.group(2).strip()
            
            # Parse parameters
            if params_str:
                params = [p.strip().split()[0] for p in params_str.split(',')]
                params = [p for p in params if p]  # Remove empty
            else:
                params = []
            
            functions.append({
                'name': func_name,
                'args': params,
                'arg_count': len(params),
                'line_no': code[:match.start()].count('\n') + 1
            })
        
        return functions