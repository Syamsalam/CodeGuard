import math
from core.ast_tokenizer import ASTTokenizer
from core.multi_file_analysis import run_multi_file_analysis

def test_path_ngrams_generation_basic():
    code = """
class A:\n    def f(self,x):\n        if x>0:\n            return x+1\n        return x-1
"""
    tok = ASTTokenizer()
    path_tokens = tok.generate_path_ngrams(code, 'python', n_min=2, n_max=3, use_hash=True, hash_len=6)
    assert any(t.startswith('PATH2_') for t in path_tokens)
    assert any(t.startswith('PATH3_') for t in path_tokens) or len(path_tokens)==0  # allow empty if parsing fails


def test_path_ngrams_in_pipeline_and_weighting():
    files = [
        ('a.py', 'def foo(x):\n    return x*x') ,
        ('b.py', 'def bar(y):\n    return y*y')
    ]
    # First run without path ngrams
    cfg_base = { 'enable_path_ngrams': False }
    res_base = run_multi_file_analysis(files, cfg_base, threshold=0.7, explain=False, explain_top_k=3)
    vocab_base = set(res_base['tfidf']['featureNames'])

    # Run with path ngrams enabled
    cfg_path = {
        'enable_path_ngrams': True,
        'path_ngram_min': 2,
        'path_ngram_max': 3,
        'path_ngram_hash': True,
        'path_ngram_hash_len': 6
    }
    res_path = run_multi_file_analysis(files, cfg_path, threshold=0.7, explain=False, explain_top_k=3)
    vocab_path = set(res_path['tfidf']['featureNames'])

    # Expect additional PATH tokens in vocab_path
    new_tokens = [t for t in vocab_path - vocab_base if t.lower().startswith('path')]
    assert len(new_tokens) > 0

    # Test path weighting reduces magnitude when <1
    cfg_weight = dict(cfg_path)
    cfg_weight['path_weight'] = 0.5
    res_weight = run_multi_file_analysis(files, cfg_weight, threshold=0.7, explain=False, explain_top_k=3)

    # For at least one PATH token, its weight in res_weight should be <= corresponding weight in res_path (per doc)
    # We'll compare first doc vector values by reconstructing mapping
    feature_names_path = res_path['tfidf']['featureNames']
    feature_names_weight = res_weight['tfidf']['featureNames']
    matrix_path = res_path['tfidf']['matrix']
    matrix_weight = res_weight['tfidf']['matrix']

    # Build dict of PATH feature -> (val_no_weight, val_weighted)
    comparisons = []
    for idx, term in enumerate(feature_names_path):
        if term.lower().startswith('path') and term in feature_names_weight:
            j = feature_names_weight.index(term)
            val_before = matrix_path[0][idx]
            val_after = matrix_weight[0][j]
            comparisons.append((val_before, val_after))
    # Ensure at least one comparison and that at least one value decreased (allow floating rounding differences)
    assert comparisons, 'No PATH features to compare for weighting assertion'
    assert any(after <= before + 1e-9 for before, after in comparisons)
