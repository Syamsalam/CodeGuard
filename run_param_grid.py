#!/usr/bin/env python3
"""Parameter grid runner for /analyze/compare endpoint.

Iterates combinations of key tuning knobs to empirically find settings that
minimize similarity scores between provided sample files while keeping
pipeline functional. Adjust the FILES list or PARAM_GRID as needed.
"""
from __future__ import annotations
import itertools
import json
import os
import statistics as stats
from typing import List, Dict, Any

import requests

API_URL = os.environ.get("CODEGUARD_API", "http://localhost:8000/analyze/compare")

# Sample files relative to repo root (adjust as required)
FILES = [
    "data/sample_codes/original.py",
    "data/sample_codes/renamed_version.py",
    "data/sample_codes/different_code.py",
]

def existing_files(files: List[str]) -> List[str]:
    return [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0]

FILES = existing_files(FILES)
if len(FILES) < 2:
    raise SystemExit("Need at least two non-empty files in FILES list")

# Define a small grid; keep combinations manageable.
PARAM_GRID = {
    "structural_weight": [0.5, 0.4, 0.3, 0.25],
    "tf_min_df": [1, 2],
    "remove_node_tokens": [False, True],
    "remove_literals": [False, True],
    "remove_operators": [False, True],
    "tf_sublinear_tf": [True],
    "tf_use_idf": [True],
    "tf_norm": ["none", "l2"],
}

ORDER = list(PARAM_GRID.keys())

def iter_param_dicts():
    values_lists = [PARAM_GRID[k] for k in ORDER]
    for combo in itertools.product(*values_lists):
        yield {k: v for k, v in zip(ORDER, combo)}

def send_request(params: Dict[str, Any]) -> Dict[str, Any]:
    files_payload = [("files", (os.path.basename(p), open(p, "rb"), "text/plain")) for p in FILES]
    data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}
    # Additional fixed params for stability
    data.update({
        "threshold": 0.7,
        "min_tokens": 1,
        "explain": False,
    })
    try:
        resp = requests.post(API_URL, files=files_payload, data=data, timeout=30)
        # Close opened file handles
        for entry in files_payload:
            # entry structure: ("files", (filename, fileobj, mimetype))
            try:
                file_tuple = entry[1]
                if len(file_tuple) >= 2 and hasattr(file_tuple[1], "close"):
                    file_tuple[1].close()
            except Exception:
                pass
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text, "params": params}
        return {"data": resp.json(), "params": params}
    except Exception as e:
        return {"error": str(e), "params": params}

def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    if "data" not in result:
        return {"params": result.get("params"), "mean": None, "max": None, "error": result.get("error"), "detail": result.get("detail")}
    scores = result["data"].get("similarityScores", [])
    if not scores:
        return {"params": result.get("params"), "mean": 0, "max": 0}
    return {
        "params": result.get("params"),
        "mean": round(stats.mean(scores), 4),
        "max": round(max(scores), 4),
        "min": round(min(scores), 4),
        "n_pairs": len(scores)
    }

def main():
    summaries = []
    for i, param_set in enumerate(iter_param_dicts(), start=1):
        print(f"[{i}] Testing params: {param_set}")
        res = send_request(param_set)
        summ = summarize(res)
        summaries.append(summ)
        if summ.get("error"):
            print(f"  -> ERROR: {summ['error']} {summ.get('detail','')}")
        else:
            print(f"  -> mean={summ['mean']} max={summ['max']} min={summ['min']} pairs={summ['n_pairs']}")
    # Sort by ascending max then mean similarity
    ranked = [s for s in summaries if s.get("error") is None]
    ranked.sort(key=lambda d: (d['max'], d['mean']))
    print("\n=== Top 10 Lowest Similarity Configurations (by max then mean) ===")
    for r in ranked[:10]:
        print(json.dumps(r, indent=2))
    # Save full results
    with open("param_grid_results.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("Saved all summaries to param_grid_results.json")

if __name__ == "__main__":
    main()
