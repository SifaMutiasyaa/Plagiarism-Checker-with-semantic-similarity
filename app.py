import os
import gc
import psutil
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import PyPDF2
from docx import Document
import io
import traceback
import warnings
warnings.filterwarnings('ignore')
import time

# ---------------------------
# Struktur folder
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
templates_dir = os.path.join(BASE_DIR, "templates")
static_dir = os.path.join(BASE_DIR, "static")

app = Flask(__name__, 
            static_folder=static_dir, 
            template_folder=templates_dir,
            static_url_path='')

# ---------------------------
# Configuration - EXTREME OPTIMIZATION
# ---------------------------
WEIGHT_LEXICAL = 0.8  # Tingkatkan weight TF-IDF
WEIGHT_STRUCTURE = 0.2  # Kurangi Jaccard
WEIGHT_SEMANTIC = 0.0  # NONAKTIFKAN SEMENTARA - ini yang paling berat!
TOP_K_TFIDF = 10  # Kurangi lagi

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
        for page in pdf_reader.pages[:10]:  # Batasi 10 halaman saja
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Gagal membaca PDF: {str(e)}")

def extract_text_from_docx(file_stream):
    try:
        doc = Document(file_stream)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs[:100] if paragraph.text.strip()])  # Batasi 100 paragraf
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
        return file_stream.read().decode('utf-8', errors='ignore').strip()[:10000]  # Batasi 10k karakter
    else:
        raise ValueError("Format file tidak didukung. Gunakan PDF, DOCX, atau TXT.")

# ---------------------------
# Model Loading dengan SWAP Strategy
# ---------------------------
class ModelManager:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.df = None
        self.corpus_texts = None
        self.loaded = False
        
    def load_minimal(self):
        """Hanya load yang benar-benar diperlukan"""
        try:
            if not self.loaded:
                print("🔄 Memulai loading model...")
                
                # 1. Load TF-IDF vectorizer
                print("📦 Memuat TF-IDF vectorizer...")
                self.tfidf_vectorizer = joblib.load(
                    os.path.join(models_dir, "tfidf.pkl")
                )
                
                # 2. Load TF-IDF matrix (dengan optimize)
                print("📦 Memuat TF-IDF matrix...")
                # Coba load dengan sparse format
                self.tfidf_matrix = joblib.load(
                    os.path.join(models_dir, "tfidf_matrix.pkl")
                )
                
                # 3. Load corpus (hanya kolom yang diperlukan)
                print("📦 Memuat CSV corpus...")
                self.df = pd.read_csv(
                    os.path.join(models_dir, "corpus_clean.csv"),
                    usecols=['abstract', 'title', 'url', 'pdf'],  # Hanya kolom yang diperlukan
                    nrows=5000  # Batasi jumlah data jika terlalu besar
                )
                self.df = self.df.fillna("")
                self.corpus_texts = self.df["abstract"].astype(str).tolist()
                
                # 4. Force garbage collection
                gc.collect()
                
                # 5. Check memory usage
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                print(f"✅ Model loaded! Memory usage: {mem_mb:.2f} MB")
                
                self.loaded = True
                return True
                
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            traceback.print_exc()
            return False
    
    def unload(self):
        """Unload models untuk hemat memory"""
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.df = None
        self.corpus_texts = None
        gc.collect()
        self.loaded = False
        print("🧹 Models unloaded from memory")

model_manager = ModelManager()

# ---------------------------
# Search functions
# ---------------------------
def tfidf_search(query, top_n=TOP_K_TFIDF):
    q_vec = model_manager.tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, model_manager.tfidf_matrix).flatten()
    idx_sorted = scores.argsort()[::-1][:top_n]
    return idx_sorted, scores[idx_sorted]

def check_plagiarism(query, top_k=3):  # Kecilkan hasil
    start_time = time.time()
    
    try:
        # Load models
        if not model_manager.load_minimal():
            raise Exception("Gagal memuat model")
        
        # Batasi query length
        query = query[:2000]  # Maksimal 2000 karakter
        
        # Cari dengan TF-IDF
        idx_top, lexical_scores = tfidf_search(query, top_n=TOP_K_TFIDF)
        
        # Hitung Jaccard similarity
        structural_scores = []
        for i in idx_top:
            score = jaccard_similarity(query, model_manager.corpus_texts[i])
            structural_scores.append(score)
        
        structural_scores = np.array(structural_scores)
        
        # Final scores (tanpa semantic)
        final_scores = (
            lexical_scores * WEIGHT_LEXICAL +
            structural_scores * WEIGHT_STRUCTURE
        )
        
        # Ambil top results
        top_indices = final_scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_idx = idx_top[idx]
            results.append({
                "index": int(doc_idx),
                "title": model_manager.df.loc[doc_idx, "title"] if "title" in model_manager.df.columns else f"Dokumen {doc_idx}",
                "url": model_manager.df.loc[doc_idx, "url"] if "url" in model_manager.df.columns else "",
                "pdf": model_manager.df.loc[doc_idx, "pdf"] if "pdf" in model_manager.df.columns else "",
                "abstract": model_manager.corpus_texts[doc_idx][:200] + "..." if len(model_manager.corpus_texts[doc_idx]) > 200 else model_manager.corpus_texts[doc_idx],
                "lexical": float(lexical_scores[idx]),
                "structure": float(structural_scores[idx]),
                "similarity": min(float(final_scores[idx]) * 100, 100),
                "processing_time": round(time.time() - start_time, 2)
            })
        
        # Unload models setelah selesai (untuk request berikutnya)
        # model_manager.unload()  # Hati-hati dengan ini
        
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
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        
        return jsonify({
            "status": "running",
            "memory_mb": round(mem_mb, 2),
            "models_loaded": model_manager.loaded,
            "config": {
                "weight_tfidf": WEIGHT_LEXICAL,
                "weight_jaccard": WEIGHT_STRUCTURE,
                "weight_semantic": WEIGHT_SEMANTIC,
                "top_k": TOP_K_TFIDF
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        query = ""
        
        # Ekstrak dari file
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                query = extract_text_from_file(file)
        
        # Atau dari text input
        if not query:
            query = request.form.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Tidak ada teks yang diberikan"}), 400
        
        # Batasi panjang query
        if len(query) > 5000:
            query = query[:5000]
        
        results = check_plagiarism(query, top_k=3)
        
        return jsonify({
            "query_length": len(query),
            "results_found": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(static_dir, path)

# ---------------------------
# GUNICORN CONFIG untuk Railway
# ---------------------------
# Railway akan execute ini
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Hanya untuk development
    app.run(host="0.0.0.0", port=port, debug=False)