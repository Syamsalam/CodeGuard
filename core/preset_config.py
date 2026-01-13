"""Settings configuration module.

Konfigurasi tunggal untuk semua parameter CodeGuard:
- Token processing, TF-IDF, cosine similarity, dan threshold
- Mudah diatur dalam satu tempat tanpa kebingungan
"""

from __future__ import annotations
from typing import Dict, Any

# KONFIGURASI TUNGGAL UNTUK SEMUA SETTINGS
SETTINGS: Dict[str, Any] = {
    # === DETECTION THRESHOLD ===
    "threshold": 0.8,                    # Ambang batas similarity (70.3% = Mirip, >=72% = Plagiat)
    
    # === TOKEN PROCESSING ===
    "min_tokens": 8,                      # Minimum jumlah token untuk analisis
    "min_token_length": 0,                # Minimum panjang token yang dipertahankan
    "keep_identifier_detail": True,       # Pertahankan detail identifier
    "remove_node_tokens": True,           # Hapus token struktural NODE_
    "remove_literals": False,             # Hapus literal (STR/NUM/BOOL) - False agar lebih sensitif
    "remove_operators": False,            # Hapus operator tokens
    "remove_keywords": False,             # Hapus keyword Python
    
    # === WEIGHT CONFIGURATION ===
    "structural_weight": 0.15,            # Bobot token struktural (0-1)
    "var_weight": 0.3,                    # Bobot token variabel (0-1)
    "operator_weight": 1.0,               # Bobot token operator (0-1)
    
    # === TF-IDF CONFIGURATION ===
    "tf_min_df": 1,                       # Minimum document frequency
    "tf_max_df": 0.85,                    # Maximum document frequency (0.85 = balanced)
    "tf_sublinear_tf": True,              # Gunakan sublinear tf scaling
    "tf_use_idf": True,                   # Aktifkan IDF weighting
    "tf_smooth_idf": True,                # Gunakan smooth IDF
    "tf_norm": "l2",                      # Normalisasi vector (l1/l2/none)
    "tf_max_features": 0,                 # Batasi jumlah features (0 = unlimited)
    "tf_normalize_vectors": True,         # Normalisasi vector setelah TF-IDF
    # === COSINE SIMILARITY ===
    "cosine_use_sparse": True,            # Gunakan sparse matrix operations (lebih efisien)
    "cosine_precision": 1e-10,            # Presisi perhitungan similarity
}

def get_settings() -> Dict[str, Any]:
    """Get semua konfigurasi settings."""
    return SETTINGS.copy()

def get_setting(key: str, default=None):
    """Get nilai setting tertentu.""" 
    return SETTINGS.get(key, default)

def update_setting(key: str, value):
    """Update nilai setting tertentu."""
    SETTINGS[key] = value

def update_settings(new_settings: Dict[str, Any]):
    """Update multiple settings sekaligus."""
    SETTINGS.update(new_settings)

def get_threshold() -> float:
    """Get threshold similarity untuk deteksi plagiat."""
    return SETTINGS["threshold"]

def get_min_tokens() -> int:
    """Get minimum tokens untuk analisis."""
    return SETTINGS["min_tokens"]

def get_tfidf_params() -> Dict[str, Any]:
    """Get semua parameter TF-IDF."""
    return {
        "min_df": SETTINGS["tf_min_df"],
        "max_df": SETTINGS["tf_max_df"], 
        "sublinear_tf": SETTINGS["tf_sublinear_tf"],
        "use_idf": SETTINGS["tf_use_idf"],
        "smooth_idf": SETTINGS["tf_smooth_idf"],
        "norm": SETTINGS["tf_norm"],
        "max_features": SETTINGS["tf_max_features"],
        "normalize_vectors": SETTINGS["tf_normalize_vectors"]
    }

def get_cosine_params() -> Dict[str, Any]:
    """Get semua parameter Cosine Similarity."""
    return {
        "use_sparse": SETTINGS["cosine_use_sparse"],
        "precision": SETTINGS["cosine_precision"]
    }

def reset_to_default():
    """Reset semua settings ke nilai default."""
    global SETTINGS
    SETTINGS.update({
        "threshold": 0.8,
        "min_tokens": 8,
        "min_token_length": 0,
        "keep_identifier_detail": True,
        "remove_node_tokens": True,
        "remove_literals": False,
        "remove_operators": False,
        "remove_keywords": False,
        "structural_weight": 0.15,
        "var_weight": 0.3,
        "operator_weight": 1.0,
        "tf_min_df": 1,
        "tf_max_df": 0.85,
        "tf_sublinear_tf": True,
        "tf_use_idf": True,
        "tf_smooth_idf": True,
        "tf_norm": "l2",
        "tf_max_features": 0,
        "tf_normalize_vectors": True,
        "cosine_use_sparse": True,
        "cosine_precision": 1e-10,
    })

__all__ = ["SETTINGS", "get_settings", "get_setting", "update_setting", "update_settings", 
           "get_threshold", "get_min_tokens", "get_tfidf_params", "get_cosine_params", "reset_to_default"]
