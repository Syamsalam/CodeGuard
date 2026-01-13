import os
import sys
import pandas as pd
from pathlib import Path

# Pastikan modul core bisa diimport
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.detector import PlagiarismDetector
except ImportError:
    print("Error: Tidak dapat mengimport modul 'core'. Pastikan script ini ada di root folder project.")
    sys.exit(1)

def create_scenarios(base_dir):
    """Membuat file kode dummy untuk 4 skenario pengujian."""
    data_dir = Path(base_dir) / "data" / "thesis_scenarios"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. BASE ORIGINAL
    code_base = """
def hitung_luas_segitiga(alas, tinggi):
    luas = 0.5 * alas * tinggi
    return luas

def main():
    a = 10
    t = 5
    print(f"Luas: {hitung_luas_segitiga(a, t)}")

if __name__ == "__main__":
    main()
"""

    # 2. SKENARIO COPY-PASTE (Identik)
    code_cp = code_base

    # 3. SKENARIO RENAMING (Ganti Nama Variabel)
    # Logika sama, teks berbeda total
    code_rename = """
def calculate_triangle_area(base, height):
    area_result = 0.5 * base * height
    return area_result

def execute_program():
    b = 10
    h = 5
    print(f"Luas: {calculate_triangle_area(b, h)}")

if __name__ == "__main__":
    execute_program()
"""

    # 4. SKENARIO KOMENTAR (Tambah Komentar & Spasi)
    code_comment = """
# Fungsi menghitung luas segitiga
# Rumus: 1/2 x alas x tinggi
def hitung_luas_segitiga(alas, tinggi):
    
    # Hitung luas
    luas = 0.5 * alas * tinggi
    return luas

def main():
    a = 10 # Nilai alas
    t = 5  # Nilai tinggi
    
    # Cetak hasil
    print(f"Luas: {hitung_luas_segitiga(a, t)}")

if __name__ == "__main__":
    main()
"""

    # 5. SKENARIO REORDER (Tukar Urutan Fungsi)
    code_reorder = """
def main():
    a = 10
    t = 5
    print(f"Luas: {hitung_luas_segitiga(a, t)}")

def hitung_luas_segitiga(alas, tinggi):
    luas = 0.5 * alas * tinggi
    return luas

if __name__ == "__main__":
    main()
"""

    files = {
        "Base.py": code_base,
        "Scenario1_CopyPaste.py": code_cp,
        "Scenario2_Rename.py": code_rename,
        "Scenario3_Comment.py": code_comment,
        "Scenario4_Reorder.py": code_reorder
    }

    paths = []
    for name, content in files.items():
        p = data_dir / name
        with open(p, "w", encoding="utf-8") as f:
            f.write(content.strip())
        paths.append(str(p))
    
    return paths

def run_test():
    print("="*80)
    print("   PENGUJIAN SKENARIO BAB 4 - CODEGUARD")
    print("="*80)

    # 1. Buat Data
    paths = create_scenarios(os.getcwd())
    
    # 2. Jalankan Deteksi
    # PERBAIKAN DI SINI: Tambahkan max_df=1.0 agar token tidak dibuang
    detector = PlagiarismDetector(
        similarity_threshold=0.0, 
        min_tokens=5, 
        max_df=1.0  # <--- WAJIB UNTUK DATASET KECIL
    )
    report = detector.detect_between_files(paths)
    
    matrix = report.similarity_matrix
    filenames = [Path(p).name for p in report.file_paths]
    
    try:
        base_idx = filenames.index("Base.py")
    except ValueError:
        print("Error: File Base.py tidak terdeteksi (mungkin token terlalu sedikit?)")
        return

    # 3. Format Data Tabel
    scenarios = [
        ("Scenario1_CopyPaste.py", "Copy-Paste Utuh", "100%", "Sempurna"),
        ("Scenario2_Rename.py", "Ganti Nama Variabel", "< 50%", "Normalisasi AST"),
        ("Scenario3_Comment.py", "Tambah Komentar", "< 80%", "Ignore Comments"),
        ("Scenario4_Reorder.py", "Ubah Urutan Fungsi", "< 60%", "Bag of Tokens")
    ]

    table_data = []
    for fname, label, est_str, note in scenarios:
        try:
            target_idx = filenames.index(fname)
            # Ambil skor kemiripan
            score = matrix[base_idx][target_idx] * 100
        except ValueError:
            score = 0.0
        
        table_data.append({
            "Skenario": label,
            "String Matching (Est)": est_str,
            "CodeGuard (Real)": f"{score:.2f}%",
            "Catatan": note
        })

    # 4. Tampilkan
    df = pd.DataFrame(table_data)
    print("\n[HASIL PENGUJIAN AKURASI]")
    print(df.to_string(index=False))
    print("\n" + "="*80)

if __name__ == "__main__":
    run_test()