import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Kurangi log TensorFlow
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Hindari deadlock tokenizer

from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import PyPDF2
from docx import Document
import io
import traceback
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Struktur folder untuk Railway
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
templates_dir = os.path.join(BASE_DIR, "templates")
static_dir = os.path.join(BASE_DIR, "static")

app = Flask(__name__, 
            static_folder=static_dir, 
            template_folder=templates_dir)

# ---------------------------
# Configuration - OPTIMIZED FOR MEMORY
# ---------------------------
WEIGHT_LEXICAL = 0.6
WEIGHT_STRUCTURE = 0.25
WEIGHT_SEMANTIC = 0.15
TOP_K_TFIDF = 15  # Kurangi dari 20 untuk hemat memori

# ---------------------------
# Helper functions
# ---------------------------
word_re = re.compile(r"\w+", flags=re.UNICODE)

def tokenize_text(text):
    return [t.lower() for t in word_re.findall(str(text))]

def jaccard_similarity(a, b):
    set_a = set(tokenize_text(a))
    set_b = set(tokenize_text(b))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Gagal membaca PDF: {str(e)}")

def extract_text_from_docx(file_stream):
    try:
        doc = Document(file_stream)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Gagal membaca DOCX: {str(e)}")

def extract_text_from_file(file):
    filename = file.filename.lower()
    file_content = file.read()
    
    if not file_content:
        raise ValueError("File kosong")
    
    file_stream = io.BytesIO(file_content)
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_stream)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file_stream)
    elif filename.endswith('.txt'):
        file_stream.seek(0)
        return file_stream.read().decode('utf-8', errors='ignore').strip()
    else:
        raise ValueError("Format file tidak didukung. Gunakan PDF, DOCX, atau TXT.")

# ---------------------------
# Global variables dengan lazy loading
# ---------------------------
tfidf_vectorizer = None
tfidf_matrix = None
df = None
corpus_texts = None
corpus_embeddings = None
model = None

# Cache untuk mengurangi beban memori
_search_cache = {}
_CACHE_SIZE = 10

def load_tfidf_models():
    """Load TF-IDF models saja"""
    global tfidf_vectorizer, tfidf_matrix
    try:
        if tfidf_vectorizer is None:
            print("Memuat TF-IDF vectorizer...")
            tfidf_vectorizer = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
            
        if tfidf_matrix is None:
            print("Memuat TF-IDF matrix...")
            # Gunakan memory mapping untuk matrix besar
            tfidf_matrix = joblib.load(os.path.join(models_dir, "tfidf_matrix.pkl"))
        return True
    except Exception as e:
        print("Error load TF-IDF:", str(e))
        return False

def load_corpus_data():
    """Load corpus data saja"""
    global df, corpus_texts
    try:
        if df is None:
            print("Memuat CSV corpus...")
            # Baca dengan chunk jika file besar
            df = pd.read_csv(os.path.join(models_dir, "corpus_clean.csv"))
            df = df.fillna("")
            corpus_texts = df["abstract"].astype(str).tolist()
        return True
    except Exception as e:
        print("Error load corpus:", str(e))
        return False

def load_embeddings():
    """Load embeddings saja"""
    global corpus_embeddings
    try:
        if corpus_embeddings is None:
            print("Memuat corpus embeddings...")
            # Gunakan mmap_mode untuk file numpy besar
            corpus_embeddings = np.load(
                os.path.join(models_dir, "corpus_embeddings.npy"),
                mmap_mode='r'  # Read-only memory mapping
            )
        return True
    except Exception as e:
        print("Error load embeddings:", str(e))
        return False

def load_sentence_transformer():
    """Load sentence transformer saja"""
    global model
    try:
        if model is None:
            print("Memuat SentenceTransformer...")
            model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cpu'  # Force CPU usage
            )
        return True
    except Exception as e:
        print("Error load SentenceTransformer:", str(e))
        return False

def cleanup_memory():
    """Bersihkan cache untuk hemat memori"""
    global _search_cache
    if len(_search_cache) > _CACHE_SIZE:
        # Hapus oldest entries
        keys_to_remove = list(_search_cache.keys())[:_CACHE_SIZE // 2]
        for key in keys_to_remove:
            del _search_cache[key]

# ---------------------------
# Search functions dengan optimasi memori
# ---------------------------
def tfidf_search(query, top_n=TOP_K_TFIDF):
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx_sorted = scores.argsort()[::-1][:top_n]
    return idx_sorted, scores[idx_sorted]

def compute_semantic_score(query, candidate_indices):
    q_emb = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
    subset = corpus_embeddings[candidate_indices]
    sims = cosine_similarity([q_emb], subset).flatten()
    return sims

def check_plagiarism(query, top_k=5):
    # Cache untuk query yang sama
    cache_key = hash(query[:100])  # Hash 100 karakter pertama
    if cache_key in _search_cache:
        return _search_cache[cache_key]
    
    try:
        # Lazy loading hanya model yang dibutuhkan
        if not load_tfidf_models():
            raise Exception("Gagal memuat model TF-IDF")
        if not load_corpus_data():
            raise Exception("Gagal memuat data corpus")
        
        idx_top, lexical_scores = tfidf_search(query, top_n=TOP_K_TFIDF)
        
        # Load embeddings hanya jika perlu
        structural_scores = np.array([jaccard_similarity(query, corpus_texts[i]) for i in idx_top])
        
        # Load sentence transformer dan embeddings hanya jika semantic weight > 0
        semantic_scores = np.zeros(len(idx_top))
        if WEIGHT_SEMANTIC > 0:
            if not load_embeddings() or not load_sentence_transformer():
                print("Peringatan: Model semantic tidak dimuat, menggunakan nilai default")
            else:
                semantic_scores = compute_semantic_score(query, idx_top)
        
        final_scores = (
            lexical_scores * WEIGHT_LEXICAL +
            structural_scores * WEIGHT_STRUCTURE +
            semantic_scores * WEIGHT_SEMANTIC
        )
        
        top_indices = final_scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_idx = idx_top[idx]
            results.append({
                "index": int(doc_idx),
                "title": df.loc[doc_idx, "title"] if "title" in df.columns else f"Dokumen {doc_idx}",
                "url": df.loc[doc_idx, "url"] if "url" in df.columns else "",
                "pdf": df.loc[doc_idx, "pdf"] if "pdf" in df.columns else "",
                "abstract": corpus_texts[doc_idx][:300] + "..." if len(corpus_texts[doc_idx]) > 300 else corpus_texts[doc_idx],
                "semantic": float(semantic_scores[idx]),
                "lexical": float(lexical_scores[idx]),
                "structure": float(structural_scores[idx]),
                "final_score": float(final_scores[idx]),
                "similarity": min(float(final_scores[idx]) * 100, 100)  # Cap di 100%
            })
        
        # Simpan ke cache
        _search_cache[cache_key] = results
        cleanup_memory()
        
        return results
        
    except Exception as e:
        print(f"Error dalam check_plagiarism: {str(e)}")
        traceback.print_exc()
        raise

# ---------------------------
# Endpoints
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    try:
        # Cek apakah model minimal bisa dimuat
        basic_loaded = load_tfidf_models() and load_corpus_data()
        return jsonify({
            "status": "running", 
            "models_loaded": basic_loaded,
            "memory_info": {
                "cache_size": len(_search_cache),
                "tfidf_loaded": tfidf_vectorizer is not None,
                "corpus_loaded": df is not None,
                "embeddings_loaded": corpus_embeddings is not None,
                "transformer_loaded": model is not None
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        query = ""

        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                query = extract_text_from_file(file)

        if not query:
            query = request.form.get("query", "").strip()

        if not query:
            return jsonify({"error": "Tidak ada teks yang diberikan"}), 400
        
        # Batasi panjang query untuk hemat memori
        if len(query) > 10000:
            query = query[:10000]
            print(f"Query dipotong menjadi {len(query)} karakter")

        results = check_plagiarism(query, top_k=5)  # Kurangi dari 10 ke 5

        return jsonify({
            "query_length": len(query),
            "results_found": len(results),
            "results": results
        })

    except Exception as e:
        print(f"Error di endpoint /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Endpoint untuk membersihkan cache secara manual"""
    global _search_cache
    _search_cache = {}
    return jsonify({"status": "cache cleared", "cache_size": 0})

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(static_dir, path)

# ---------------------------
# GUNICORN CONFIG UNTUK RAILWAY
# ---------------------------
# Railway akan menggunakan gunicorn secara otomatis
# Tapi kita bisa atur worker yang lebih sedikit

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Jangan gunakan debug di production
    app.run(host="0.0.0.0", port=port, debug=False)