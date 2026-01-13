import sys
import textwrap  # <--- Tambahan library untuk fix indentasi
from pathlib import Path

# Import core modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.ast_tokenizer import ASTTokenizer
    from core.tfidf_vectorizer import TFIDFVectorizer
except ImportError:
    print("Error: Jalankan script ini dari folder root project.")
    sys.exit(1)

def test_tokenizer_behavior():
    print("\n--- TEST 1: AST Tokenizer Behavior ---")
    tokenizer = ASTTokenizer()
    
    # KODE BERSIH
    code_clean = "def fungsi(x): return x + 1"
    
    # KODE KOTOR (Kita gunakan dedent agar indentasi valid)
    # Ini mensimulasikan file yang diupload user (biasanya dimulai dari kolom 0)
    raw_dirty = """
    # Ini komentar panjang
    # Penjelasan fungsi
    def    fungsi  (  x  ) : 
        return x + 1
    """
    code_dirty = textwrap.dedent(raw_dirty).strip()
    
    # Debug: Pastikan code_dirty valid
    # print(f"DEBUG Code Dirty:\n{code_dirty}\n")
    
    tokens1 = tokenizer.tokenize_code(code_clean, 'python')
    tokens2 = tokenizer.tokenize_code(code_dirty, 'python')
    
    print(f"Token Code Bersih : {tokens1[:5]} ...") # Tampilkan sebagian saja biar rapi
    print(f"Token Code Kotor  : {tokens2[:5]} ...")
    
    if tokens1 == tokens2:
        print("✅ SUKSES: Tokenizer berhasil mengabaikan komentar & spasi!")
    else:
        print("❌ GAGAL: Token berbeda.")
        print("Saran: Pastikan code_dirty adalah sintaks Python yang valid (tidak ada indentasi liar).")

def test_renaming_behavior():
    print("\n--- TEST 2: Variable Renaming ---")
    tokenizer = ASTTokenizer()
    
    # Fungsi sama, nama variabel beda
    code_a = "def hitung(angka): return angka * 2"
    code_b = "def calculate(number): return number * 2"
    
    tokens_a = tokenizer.tokenize_code(code_a, 'python')
    tokens_b = tokenizer.tokenize_code(code_b, 'python')
    
    # Kita cek apakah strukturnya mirip
    # AST normalisasi biasanya menghasilkan urutan token tipe node yang sama
    print(f"Tokens A (First 5): {tokens_a[:5]}")
    print(f"Tokens B (First 5): {tokens_b[:5]}")
    
    if len(tokens_a) == 0 or len(tokens_b) == 0:
         print("❌ GAGAL: Token kosong.")
         return

    # Hitung kesamaan sederhana (Jaccard)
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    
    # Jaccard biasa kurang akurat untuk urutan, tapi cukup untuk unit test sederhana
    # Lebih baik cek exact match karena AST menormalisasi nama variabel jadi VAR_USE
    if tokens_a == tokens_b:
        print(f"Similarity Token: 1.0 (Identik)")
        print("✅ SUKSES: Renaming terdeteksi memiliki struktur AST yang 100% SAMA.")
    else:
        overlap = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        print(f"Similarity Token: {overlap:.2f}")
        if overlap > 0.8:
            print("✅ SUKSES: Renaming terdeteksi memiliki struktur mirip.")
        else:
            print("⚠️ WARNING: Struktur dianggap terlalu berbeda.")

if __name__ == "__main__":
    test_tokenizer_behavior()
    test_renaming_behavior()