"""Preset configuration module.

Memindahkan dictionary PRESETS dari `api_enhanced.py` agar bisa digunakan
lintas modul (multi-file analysis, GitHub task, manual compare, dsb).

Setiap preset memiliki:
 - description: penjelasan singkat
 - params: konfigurasi token & TF-IDF & weighting
 - default_threshold: ambang batas default untuk status 'Plagiat'
"""

from __future__ import annotations

from typing import Dict, Any

PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "description": "Meminimalkan similarity inflasi untuk kode sangat pendek / struktural mirip.",
        "params": {
            "keep_identifier_detail": True,
            "remove_node_tokens": True,
            "remove_literals": True,
            "remove_operators": False,
            "remove_keywords": False,
            "min_token_length": 0,
            "structural_weight": 0.1,
            "var_weight": 0.2,
            "operator_weight": 1.0,
            "tf_min_df": 1,
            "tf_max_df": 1.0,
            "tf_sublinear_tf": True,
            "tf_use_idf": True,
            "tf_smooth_idf": True,
            "tf_norm": "l2",
            "tf_max_features": 0
        },
        "default_threshold": 0.7
    },
    "balanced": {
        "description": "Kompromi antara sensitivitas dan penurunan kemiripan palsu.",
        "params": {
            "keep_identifier_detail": True,
            "remove_node_tokens": True,
            "remove_literals": False,
            "remove_operators": False,
            "remove_keywords": False,
            "min_token_length": 0,
            "structural_weight": 0.15,
            "var_weight": 0.3,
            "operator_weight": 1.0,
            "tf_min_df": 1,
            "tf_max_df": 1.0,
            "tf_sublinear_tf": True,
            "tf_use_idf": True,
            "tf_smooth_idf": True,
            "tf_norm": "l2",
            "tf_max_features": 0
        },
        "default_threshold": 0.7
    },
    "permissive": {
        "description": "Lebih mudah menandai kemiripan (lebih longgar, similarity cenderung lebih tinggi).",
        "params": {
            "keep_identifier_detail": False,
            "remove_node_tokens": False,
            "remove_literals": False,
            "remove_operators": False,
            "remove_keywords": False,
            "min_token_length": 0,
            "structural_weight": 0.3,
            "var_weight": 0.6,
            "operator_weight": 1.0,
            "tf_min_df": 1,
            "tf_max_df": 1.0,
            "tf_sublinear_tf": True,
            "tf_use_idf": True,
            "tf_smooth_idf": True,
            "tf_norm": "l2",
            "tf_max_features": 0
        },
        "default_threshold": 0.75
    }
}

__all__ = ["PRESETS"]
