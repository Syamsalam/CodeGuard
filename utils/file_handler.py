"""
File handling utilities for source code plagiarism detection.
Handles file I/O operations, filtering, and directory management.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import fnmatch
import hashlib


class FileHandler:
    def __init__(self):
        """Initialize file handler with default settings"""
        self.supported_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.cc', '.c', '.h'}
        self.ignore_patterns = {
            '__pycache__',
            'node_modules',
            '.git',
            '.vscode',
            '.idea',
            '*.pyc',
            '*.pyo',
            '*.min.js',
            'dist',
            'build'
        }
    
    def find_source_files(self, directory: str, 
                         extensions: Optional[List[str]] = None,
                         recursive: bool = True,
                         ignore_patterns: Optional[Set[str]] = None) -> List[str]:
        """
        Find source code files in directory.
        
        Args:
            directory: Directory path to search
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            ignore_patterns: Patterns to ignore (in addition to defaults)
            
        Returns:
            List of file paths
        """
        if extensions is None:
            extensions = list(self.supported_extensions)
        
        ignore_set = self.ignore_patterns.copy()
        if ignore_patterns:
            ignore_set.update(ignore_patterns)
        
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        source_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore(d, ignore_set)]
                
                for file in files:
                    if self._should_include_file(file, extensions, ignore_set):
                        file_path = os.path.join(root, file)
                        source_files.append(file_path)
        else:
            for file in directory_path.iterdir():
                if file.is_file() and self._should_include_file(file.name, extensions, ignore_set):
                    source_files.append(str(file))
        
        return sorted(source_files)
    
    def read_file_content(self, file_path: str) -> str:
        """
        Read file content with proper encoding handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"Could not decode file: {file_path}")
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        stat_info = file_path.stat()
        
        try:
            content = self.read_file_content(str(file_path))
            lines = content.count('\n') + 1
            chars = len(content)
            
            # Calculate file hash for duplicate detection
            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        except UnicodeDecodeError:
            lines = chars = 0
            file_hash = None
        
        return {
            'path': str(file_path.absolute()),
            'name': file_path.name,
            'extension': file_path.suffix,
            'size_bytes': stat_info.st_size,
            'lines': lines,
            'characters': chars,
            'created': stat_info.st_ctime,
            'modified': stat_info.st_mtime,
            'hash': file_hash
        }
    
    def find_duplicate_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Find duplicate files based on content hash.
        
        Args:
            file_paths: List of file paths to check
            
        Returns:
            Dictionary mapping hash to list of duplicate file paths
        """
        hash_map = {}
        
        for file_path in file_paths:
            try:
                file_info = self.get_file_info(file_path)
                file_hash = file_info['hash']
                
                if file_hash:
                    if file_hash not in hash_map:
                        hash_map[file_hash] = []
                    hash_map[file_hash].append(file_path)
            
            except (FileNotFoundError, UnicodeDecodeError):
                continue
        
        # Return only groups with duplicates
        duplicates = {h: files for h, files in hash_map.items() if len(files) > 1}
        
        return duplicates
    
    def filter_files_by_size(self, file_paths: List[str], 
                           min_size: int = 0, max_size: Optional[int] = None) -> List[str]:
        """
        Filter files by size constraints.
        
        Args:
            file_paths: List of file paths
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes (None for no limit)
            
        Returns:
            Filtered list of file paths
        """
        filtered_files = []
        
        for file_path in file_paths:
            try:
                file_size = Path(file_path).stat().st_size
                
                if file_size >= min_size:
                    if max_size is None or file_size <= max_size:
                        filtered_files.append(file_path)
            
            except (FileNotFoundError, OSError):
                continue
        
        return filtered_files
    
    def organize_files_by_extension(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Organize files by their extensions.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping extension to list of file paths
        """
        organized = {}
        
        for file_path in file_paths:
            extension = Path(file_path).suffix.lower()
            
            if extension not in organized:
                organized[extension] = []
            
            organized[extension].append(file_path)
        
        return organized
    
    def create_backup(self, file_path: str, backup_dir: Optional[str] = None) -> str:
        """
        Create a backup copy of a file.
        
        Args:
            file_path: Path to the file to backup
            backup_dir: Directory for backup (default: same directory)
            
        Returns:
            Path to the backup file
        """
        source_path = Path(file_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if backup_dir:
            backup_path = Path(backup_dir) / f"{source_path.stem}_backup{source_path.suffix}"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            backup_path = source_path.parent / f"{source_path.stem}_backup{source_path.suffix}"
        
        shutil.copy2(source_path, backup_path)
        
        return str(backup_path)
    
    def cleanup_temp_files(self, directory: str, patterns: List[str]) -> int:
        """
        Clean up temporary files matching patterns.
        
        Args:
            directory: Directory to clean
            patterns: List of glob patterns to match
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return deleted_count
        
        for pattern in patterns:
            for file_path in directory_path.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_count += 1
                except OSError:
                    continue
        
        return deleted_count
    
    def validate_file_paths(self, file_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate file paths and separate valid from invalid.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Tuple of (valid_paths, invalid_paths)
        """
        valid_paths = []
        invalid_paths = []
        
        for file_path in file_paths:
            path_obj = Path(file_path)
            
            if path_obj.exists() and path_obj.is_file():
                # Check if it's a supported source file
                if path_obj.suffix.lower() in self.supported_extensions:
                    valid_paths.append(file_path)
                else:
                    invalid_paths.append(file_path)
            else:
                invalid_paths.append(file_path)
        
        return valid_paths, invalid_paths
    
    def _should_ignore(self, name: str, ignore_patterns: Set[str]) -> bool:
        """Check if file/directory should be ignored"""
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    def _should_include_file(self, filename: str, extensions: List[str], 
                           ignore_patterns: Set[str]) -> bool:
        """Check if file should be included"""
        # Check ignore patterns
        if self._should_ignore(filename, ignore_patterns):
            return False
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        return file_ext in extensions
    
    def get_directory_stats(self, directory: str) -> Dict[str, any]:
        """
        Get statistics about source files in directory.
        
        Args:
            directory: Directory path
            
        Returns:
            Dictionary with directory statistics
        """
        source_files = self.find_source_files(directory)
        
        if not source_files:
            return {
                'total_files': 0,
                'total_size': 0,
                'total_lines': 0,
                'by_extension': {},
                'largest_file': None,
                'smallest_file': None
            }
        
        total_size = 0
        total_lines = 0
        by_extension = {}
        file_sizes = []
        
        for file_path in source_files:
            try:
                file_info = self.get_file_info(file_path)
                
                total_size += file_info['size_bytes']
                total_lines += file_info['lines']
                
                ext = file_info['extension']
                if ext not in by_extension:
                    by_extension[ext] = {'count': 0, 'size': 0, 'lines': 0}
                
                by_extension[ext]['count'] += 1
                by_extension[ext]['size'] += file_info['size_bytes']
                by_extension[ext]['lines'] += file_info['lines']
                
                file_sizes.append((file_path, file_info['size_bytes']))
            
            except (FileNotFoundError, UnicodeDecodeError):
                continue
        
        # Find largest and smallest files
        if file_sizes:
            file_sizes.sort(key=lambda x: x[1])
            smallest_file = {'path': file_sizes[0][0], 'size': file_sizes[0][1]}
            largest_file = {'path': file_sizes[-1][0], 'size': file_sizes[-1][1]}
        else:
            smallest_file = largest_file = None
        
        return {
            'total_files': len(source_files),
            'total_size': total_size,
            'total_lines': total_lines,
            'by_extension': by_extension,
            'largest_file': largest_file,
            'smallest_file': smallest_file
        }