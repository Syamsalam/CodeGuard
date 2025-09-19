from core.multi_file_analysis import run_multi_file_analysis


def test_path_ngram_cap_frequency():
    code = """
class A:\n    def f(self,x):\n        if x>0:\n            return x+1\n        else:\n            return x-1\nclass B:\n    def g(self,y):\n        for i in range(y):\n            if i%2==0:\n                y+=i\n        return y
"""
    files = [("a.py", code), ("b.py", code.replace('y','z'))]
    cfg = {
        'enable_path_ngrams': True,
        'path_ngram_min': 2,
        'path_ngram_max': 3,
        'path_ngram_hash': True,
        'path_ngram_hash_len': 6,
        'path_ngram_cap': 5,  # keep only top 5 unique path tokens by frequency
        'path_ngram_cap_strategy': 'frequency'
    }
    res = run_multi_file_analysis(files, cfg, threshold=0.7, explain=False, explain_top_k=3)
    debug = res['tuningConfig']['path_ngrams']['cap_debug']
    assert debug is not None and len(debug) == 2
    for entry in debug:
        if entry['original_unique'] > 5:
            assert entry['kept_unique'] == 5
            assert len(entry['kept_tokens']) == 5


def test_path_ngram_cap_random_reproducible():
    code = "def foo(x):\n    if x>1:\n        return x+1\n    return x-1"  # simple
    files = [("a.py", code), ("b.py", code)]
    cfg1 = {
        'enable_path_ngrams': True,
        'path_ngram_min': 2,
        'path_ngram_max': 4,
        'path_ngram_hash': True,
        'path_ngram_hash_len': 6,
        'path_ngram_cap': 3,
        'path_ngram_cap_strategy': 'random',
        'path_ngram_cap_seed': 42
    }
    cfg2 = dict(cfg1)
    res1 = run_multi_file_analysis(files, cfg1, threshold=0.7, explain=False, explain_top_k=3)
    res2 = run_multi_file_analysis(files, cfg2, threshold=0.7, explain=False, explain_top_k=3)
    d1 = res1['tuningConfig']['path_ngrams']['cap_debug'][0]['kept_tokens']
    d2 = res2['tuningConfig']['path_ngrams']['cap_debug'][0]['kept_tokens']
    # With fixed seed, selection should be identical
    assert d1 == d2
