#!/usr/bin/env python3
"""
Cross-project manual upload analysis
Groups uploaded files by project/folder and performs cross-project comparison
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

from .multi_file_analysis import run_multi_file_analysis
from .enhanced_detector import EnhancedPlagiarismDetector

logger = logging.getLogger(__name__)

def group_files_by_project(files_data: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group files by project based on folder structure.
    Example: 'project1/src/main.py' -> project 'project1'
    """
    projects = {}
    
    # First pass: detect if we need to look deeper in folder structure
    sample_paths = [filepath for filepath, _ in files_data[:20]]  # Look at first 20 files
    
    # Check if all files are under one root folder
    all_under_one_root = True
    root_folder = None
    
    for filepath in sample_paths:
        normalized_path = filepath.replace('\\', '/')
        path_parts = normalized_path.split('/')
        
        if len(path_parts) < 2:
            all_under_one_root = False
            break
            
        current_root = path_parts[0]
        if root_folder is None:
            root_folder = current_root
        elif root_folder != current_root:
            all_under_one_root = False
            break
    
    # Determine project extraction level
    project_level = 0  # 0 = first folder, 1 = second folder, etc.
    
    if all_under_one_root and root_folder:
        # All files are under one root, check if there are subfolders that can be projects
        subfolders = set()
        for filepath in sample_paths:
            normalized_path = filepath.replace('\\', '/')
            path_parts = normalized_path.split('/')
            if len(path_parts) >= 3:  # root/subfolder/file
                subfolders.add(path_parts[1])
        
        if len(subfolders) > 1:
            project_level = 1  # Use second level folders as projects
    
    # Extract projects based on determined level
    for filepath, content in files_data:
        normalized_path = filepath.replace('\\', '/')
        path_parts = normalized_path.split('/')
        
        if len(path_parts) > project_level:
            if project_level == 0:
                project_name = path_parts[0]
                relative_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else path_parts[0]
            elif project_level == 1 and len(path_parts) >= 2:
                project_name = path_parts[1]  # Use second level as project name
                relative_path = '/'.join(path_parts[2:]) if len(path_parts) > 2 else path_parts[1]
            else:
                project_name = 'mixed'
                relative_path = normalized_path
        else:
            project_name = 'mixed'
            relative_path = normalized_path
        
        if project_name not in projects:
            projects[project_name] = []
        
        projects[project_name].append((relative_path, content))
    
    # If still only one project found, try artificial grouping
    if len(projects) == 1:
        single_project_name = list(projects.keys())[0] 
        files = projects[single_project_name]
        
        # Strategy 1: Group by file extension/type
        extension_projects = {}
        for filepath, content in files:
            # Extract extension
            if '.' in filepath:
                ext = filepath.split('.')[-1].lower()
            else:
                ext = 'noext'
            
            # Group similar extensions
            if ext in ['py', 'pyx', 'pyi']:
                project_name = "python_files"
            elif ext in ['js', 'jsx', 'ts', 'tsx']:
                project_name = "javascript_files"  
            elif ext in ['java', 'class']:
                project_name = "java_files"
            elif ext in ['cpp', 'cc', 'cxx', 'c', 'h', 'hpp']:
                project_name = "cpp_files"
            elif ext in ['html', 'htm', 'css', 'scss', 'sass']:
                project_name = "web_files"
            else:
                project_name = f"files_{ext}"
            
            if project_name not in extension_projects:
                extension_projects[project_name] = []
            extension_projects[project_name].append((filepath, content))
        
        if len(extension_projects) > 1:
            projects = extension_projects
        else:
            # Strategy 2: Split alphabetically into groups based on file count
            sorted_files = sorted(files, key=lambda x: x[0])
            total_files = len(sorted_files)
            
            if total_files >= 4:
                # Split into multiple groups
                num_groups = min(4, max(2, total_files // 50))  # 2-4 groups based on file count
                group_size = total_files // num_groups
                
                projects = {}
                for i in range(num_groups):
                    start_idx = i * group_size
                    end_idx = start_idx + group_size if i < num_groups - 1 else total_files
                    
                    group_name = f"{single_project_name}_group_{chr(65 + i)}"  # A, B, C, D
                    projects[group_name] = sorted_files[start_idx:end_idx]
    
    return projects

def calculate_cross_project_metrics(project_results: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """
    Calculate cross-project metrics similar to cross-repo analysis
    """
    project_pairs = []
    project_names = list(project_results.keys())
    
    detector = EnhancedPlagiarismDetector(threshold=threshold)
    
    for i in range(len(project_names)):
        for j in range(i + 1, len(project_names)):
            project_a = project_names[i]
            project_b = project_names[j]
            
            result_a = project_results[project_a]
            result_b = project_results[project_b]
            
            # Get file contents for cross-project comparison
            files_a = result_a.get('fileData', [])
            files_b = result_b.get('fileData', [])
            
            if not files_a or not files_b:
                # If no fileData, create from comparisonsDetail if available
                comp_details_a = result_a.get('comparisonsDetail', [])
                comp_details_b = result_b.get('comparisonsDetail', [])
                
                # Extract unique files from comparison details
                if comp_details_a:
                    files_set = set()
                    for detail in comp_details_a:
                        if detail.get('file1'):
                            files_set.add(detail['file1'])
                        if detail.get('file2'):
                            files_set.add(detail['file2'])
                    files_a = [(f, f"# File: {f}") for f in files_set]  # Dummy content
                
                if comp_details_b:
                    files_set = set()
                    for detail in comp_details_b:
                        if detail.get('file1'):
                            files_set.add(detail['file1'])
                        if detail.get('file2'):
                            files_set.add(detail['file2'])
                    files_b = [(f, f"# File: {f}") for f in files_set]  # Dummy content
            
            if files_a and files_b:
                try:
                    # Use EnhancedPlagiarismDetector for cross-project comparison
                    cross_result = detector.detect_cross_repository_plagiarism(
                        files_a, files_b, threshold
                    )
                    
                    # Process the results - cross_result is a List[Dict]
                    if cross_result:
                        similarities = [item['similarity'] for item in cross_result]
                        max_similarity = max(similarities) if similarities else 0
                        mean_similarity = sum(similarities) / len(similarities) if similarities else 0
                        plagiarized_pairs = sum(1 for item in cross_result if item['is_plagiarized'])
                        total_pairs = len(cross_result)
                        
                        # Get top file pairs (sorted by similarity)
                        top_pairs = sorted(cross_result, key=lambda x: x['similarity'], reverse=True)[:5]
                        
                        # Convert to expected format for UI
                        enhanced_pairs = []
                        for item in cross_result:
                            enhanced_pair = {
                                'file1_name': item['repo_a_file'],
                                'file2_name': item['repo_b_file'], 
                                'similarity_score': item['similarity'],
                                'analysis_details': item.get('details', {}),
                                'confidence': item['details'].get('confidence', 'unknown') if item.get('details') else 'unknown',
                                'level': item['details'].get('level', 'UNKNOWN') if item.get('details') else 'UNKNOWN',
                                'interpretation': item['details'].get('interpretation', 'No interpretation') if item.get('details') else 'No interpretation'
                            }
                            enhanced_pairs.append(enhanced_pair)
                    else:
                        max_similarity = 0
                        mean_similarity = 0
                        plagiarized_pairs = 0
                        total_pairs = 0
                        top_pairs = []
                        enhanced_pairs = []
                    
                    project_pairs.append({
                        'projectA': project_a,
                        'projectB': project_b,
                        'max_similarity': max_similarity,
                        'mean_similarity': mean_similarity,
                        'best_match_mean_similarity': mean_similarity,  # For UI compatibility
                        'plagiarized_pairs': plagiarized_pairs,
                        'total_pairs': total_pairs,
                        'best_match_plagiarized_pairs': plagiarized_pairs,
                        'detailed_comparisons': enhanced_pairs,
                        'best_match_total_pairs': total_pairs,
                        'top_file_pairs': top_pairs,
                        'files_a_count': len(files_a),
                        'files_b_count': len(files_b)
                    })
                except Exception as e:
                    logger.error(f"Error comparing projects {project_a} vs {project_b}: {e}")
                    # Add empty result to maintain structure
                    project_pairs.append({
                        'projectA': project_a,
                        'projectB': project_b,
                        'max_similarity': 0,
                        'mean_similarity': 0,
                        'best_match_mean_similarity': 0,
                        'plagiarized_pairs': 0,
                        'total_pairs': 0,
                        'best_match_plagiarized_pairs': 0,
                        'best_match_total_pairs': 0,
                        'top_file_pairs': [],
                        'files_a_count': 0,
                        'files_b_count': 0,
                        'error': str(e)
                    })
    
    return {
        'project_pairs': project_pairs,
        'total_projects': len(project_names),
        'total_pairs': len(project_pairs),
        'project_names': project_names
    }

async def run_cross_project_analysis_background(
    analysis_results: Dict[str, Any],
    analysis_id: str,
    files_data: List[Tuple[str, str]],
    threshold: float,
    preset_config: Optional[Dict] = None,
    explain: bool = False,
    explain_top_k: int = 5,
    skip_matrices: bool = True
):
    """
    Background task for cross-project analysis
    """
    try:
        logger.info(f"Starting cross-project analysis {analysis_id}")
        
        # Add timeout and size limits for large datasets
        MAX_FILES_PER_PROJECT = 50   # Reduced limit to prevent hanging
        MAX_TOTAL_FILES = 200        # Reduced total files limit
        MAX_PROJECT_COMPARISONS = 20 # Limit number of project comparisons
        
        if len(files_data) > MAX_TOTAL_FILES:
            logger.warning(f"Dataset too large ({len(files_data)} files), limiting to {MAX_TOTAL_FILES} files")
            files_data = files_data[:MAX_TOTAL_FILES]
        
        # Group files by project
        projects = group_files_by_project(files_data)
        
        logger.info(f"Grouping files: received {len(files_data)} files")
        for i, (filename, _) in enumerate(files_data[:5]):  # Log first 5 files
            logger.info(f"File {i}: {filename}")
        
        # Limit files per project to prevent hanging
        for project_name, project_files in projects.items():
            if len(project_files) > MAX_FILES_PER_PROJECT:
                logger.warning(f"Project '{project_name}' too large ({len(project_files)} files), limiting to {MAX_FILES_PER_PROJECT} files")
                projects[project_name] = project_files[:MAX_FILES_PER_PROJECT]
        
        logger.info(f"Detected projects: {list(projects.keys())}")
        for project_name, project_files in projects.items():
            logger.info(f"Project '{project_name}': {len(project_files)} files")
        
        # Limit number of projects to prevent combinatorial explosion
        if len(projects) > 20:
            logger.warning(f"Too many projects ({len(projects)}), limiting to first 20 projects")
            project_names = list(projects.keys())[:20]
            projects = {name: projects[name] for name in project_names}
        
        if len(projects) < 2:
            error_msg = f'Minimal 2 project/folder diperlukan untuk analisis cross-project. Ditemukan: {len(projects)} project: {list(projects.keys())}'
            logger.error(error_msg)
            analysis_results[analysis_id].update({
                'status': 'error',
                'error': error_msg,
                'completed_at': datetime.now().isoformat(),
                'debug_info': {
                    'total_files': len(files_data),
                    'detected_projects': list(projects.keys()),
                    'sample_filenames': [f[0] for f in files_data[:10]]
                }
            })
            return
        
        logger.info(f"Found {len(projects)} projects: {list(projects.keys())}")
        
        # Update progress
        analysis_results[analysis_id].update({
            'status': 'processing',
            'progress': f"Menganalisis {len(projects)} project...",
            'projects_found': list(projects.keys())
        })
        
        # Analyze each project internally
        project_results = {}
        
        for project_name, project_files in projects.items():
            if len(project_files) < 2:
                logger.warning(f"Project {project_name} has only {len(project_files)} files, skipping internal analysis")
                project_results[project_name] = {
                    'filesCount': len(project_files),
                    'comparisons': 0,
                    'plagiarismCount': 0,
                    'comparisonsDetail': [],
                    'fileData': project_files,
                    'warning': 'Too few files for internal analysis'
                }
                continue
            
            try:
                logger.info(f"Analyzing project {project_name} with {len(project_files)} files")
                
                # Skip projects that are too large for efficient processing
                if len(project_files) > 30:
                    logger.warning(f"Skipping large project {project_name} ({len(project_files)} files) to prevent hanging")
                    project_results[project_name] = {
                        'filesCount': len(project_files),
                        'comparisons': 0,
                        'plagiarismCount': 0,
                        'comparisonsDetail': [],
                        'fileData': project_files,
                        'warning': 'Project too large - skipped analysis'
                    }
                    continue
                
                # Run analysis for this project
                if preset_config:
                    result = run_multi_file_analysis(
                        project_files, 
                        preset_config, 
                        threshold, 
                        explain, 
                        explain_top_k
                    )
                else:
                    # Fallback basic analysis
                    result = {
                        'filesCount': len(project_files),
                        'comparisons': 0,
                        'plagiarismCount': 0,
                        'comparisonsDetail': [],
                        'fileData': project_files
                    }
                
                # Store file data for cross-project comparison
                result['fileData'] = project_files
                project_results[project_name] = result
                
            except Exception as e:
                logger.error(f"Error analyzing project {project_name}: {e}")
                project_results[project_name] = {
                    'filesCount': len(project_files),
                    'comparisons': 0,
                    'plagiarismCount': 0,
                    'comparisonsDetail': [],
                    'fileData': project_files,
                    'error': str(e)
                }
        
        # Update progress
        analysis_results[analysis_id].update({
            'progress': "Membandingkan antar project..."
        })
        
        # Calculate cross-project metrics
        cross_metrics = calculate_cross_project_metrics(project_results, threshold)
        
        # Collect all detailed comparisons from cross-project pairs
        all_comparisons = []
        for pair in cross_metrics.get('project_pairs', []):
            detailed_comps = pair.get('detailed_comparisons', [])
            all_comparisons.extend(detailed_comps)
        
        # Generate TF-IDF and similarity matrices for all files
        all_files_data = []
        all_file_names = []
        
        for project_name, project_files in projects.items():
            for filename, content in project_files:
                all_files_data.append((filename, content))
                all_file_names.append(filename)
        
        tfidf_data = None
        similarity_matrix = None
        
        if not skip_matrices and len(all_files_data) >= 2:
            try:
                from core.enhanced_detector import EnhancedPlagiarismDetector
                detector = EnhancedPlagiarismDetector(threshold=threshold)
                
                # Generate TF-IDF matrix
                tfidf_result = detector.generate_tfidf_matrix(all_files_data)
                if tfidf_result:
                    tfidf_data = {
                        'featureNames': tfidf_result.get('feature_names', []),
                        'matrix': tfidf_result.get('tfidf_matrix', []),
                        'vocabulary_size': len(tfidf_result.get('feature_names', []))
                    }
                
                # Generate similarity matrix
                similarity_result = detector.generate_similarity_matrix(all_files_data)
                if similarity_result:
                    similarity_matrix = similarity_result.get('similarity_matrix', [])
                
            except Exception as e:
                logger.warning(f"Failed to generate matrices: {e}")
        
        # Prepare final result data
        result_data = {
            'threshold': threshold,
            'total_projects': len(projects),
            'total_files': len(files_data),
            'processing_time': 0,  # Will be calculated if needed
            'comparisons': all_comparisons,  # For frontend compatibility
            'tfidf': tfidf_data,  # TF-IDF matrix data
            'similarity_matrix': similarity_matrix,  # Similarity matrix
            'fileNames': all_file_names,  # File names for matrix display
            'projects': {name: {
                'filesCount': proj['filesCount'],
                'comparisons': proj.get('comparisons', 0),
                'plagiarismCount': proj.get('plagiarismCount', 0),
                'files': [f[0] for f in projects[name]]  # Just filenames
            } for name, proj in project_results.items()},
            'cross_project_analysis': cross_metrics,
            'project_results': project_results if not skip_matrices else None,
            'message': f"Analisis cross-project selesai: {len(projects)} project, {len(cross_metrics['project_pairs'])} pasangan"
        }
        
        # Update analysis_results with proper structure
        analysis_results[analysis_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'type': 'cross_project_manual',
            'result': result_data  # The actual result data frontend expects
        })
        logger.info(f"Cross-project analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in cross-project analysis {analysis_id}: {e}")
        analysis_results[analysis_id].update({
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })