import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

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

app = Flask(__name__, 
            static_folder="static", 
            template_folder="templates",
            static_url_path="/static")

# ---------------------------
# Configuration / weights
# ---------------------------
WEIGHT_LEXICAL = 0.6
WEIGHT_STRUCTURE = 0.25
WEIGHT_SEMANTIC = 0.15
TOP_K_TFIDF = 20

# ---------------------------
# Helper functions
# ---------------------------
word_re = re.compile(r"\w+", flags=re.UNICODE)

def tokenize_text(text):
    return [t.lower() for t in word_re.findall(str(text))]

def jaccard_similarity(a, b):
    """Structural similarity menggunakan Jaccard pada set kata"""
    set_a = set(tokenize_text(a))
    set_b = set(tokenize_text(b))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def extract_text_from_pdf(file_stream):
    """Ekstrak teks dari PDF"""
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
    """Ekstrak teks dari DOCX"""
    try:
        doc = Document(file_stream)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Gagal membaca DOCX: {str(e)}")

def extract_text_from_file(file):
    """Deteksi tipe file dan ekstrak teks"""
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
# Global variables untuk models & data
# ---------------------------
tfidf_vectorizer = None
tfidf_matrix = None
df = None
corpus_texts = None
corpus_embeddings = None
model = None

def load_models():
    """Lazy loading untuk models (untuk cold start di Vercel)"""
    global tfidf_vectorizer, tfidf_matrix, df, corpus_texts, corpus_embeddings, model
    
    try:
        if tfidf_vectorizer is None:
            print("Loading TF-IDF vectorizer...")
            tfidf_vectorizer = joblib.load("models/tfidf.pkl")
            
        if tfidf_matrix is None:
            print("Loading TF-IDF matrix...")
            tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
            
        if df is None:
            print("Loading CSV data...")
            df = pd.read_csv("models/corpus_clean.csv")
            df = df.fillna("")
            
            if "abstract" not in df.columns:
                raise ValueError("CSV tidak memiliki kolom 'abstract'.")
            
            corpus_texts = df["abstract"].astype(str).tolist()
            
        if corpus_embeddings is None:
            print("Loading embeddings...")
            corpus_embeddings = np.load("models/corpus_embeddings.npy")
            
        if model is None:
            print("Loading SentenceTransformer model...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        traceback.print_exc()
        raise

# ---------------------------
# Search functions
# ---------------------------
def tfidf_search(query, top_n=TOP_K_TFIDF):
    """Cari dokumen menggunakan TF-IDF (lexical)"""
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx_sorted = scores.argsort()[::-1][:top_n]
    return idx_sorted, scores[idx_sorted]

def compute_semantic_score(query, candidate_indices):
    """Hitung semantic similarity menggunakan SentenceTransformer"""
    q_emb = model.encode([query], show_progress_bar=False)[0]
    subset = corpus_embeddings[candidate_indices]
    sims = cosine_similarity([q_emb], subset).flatten()
    return sims

def check_plagiarism(query, top_k=5):
    """Fungsi utama untuk cek plagiarisme"""
    # Load models jika belum
    load_models()
    
    # 1. LEXICAL (TF-IDF)
    idx_top, lexical_scores = tfidf_search(query, top_n=TOP_K_TFIDF)
    
    # 2. SEMANTIC (SentenceTransformer)
    semantic_scores = compute_semantic_score(query, idx_top)
    
    # 3. STRUCTURAL (Jaccard)
    structural_scores = np.array([jaccard_similarity(query, corpus_texts[i]) for i in idx_top])
    
    # 4. GABUNGKAN SCORE
    final_scores = (
        lexical_scores * WEIGHT_LEXICAL +
        structural_scores * WEIGHT_STRUCTURE +
        semantic_scores * WEIGHT_SEMANTIC
    )
    
    # 5. Ambil top-k hasil
    top_indices = final_scores.argsort()[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_indices):
        doc_idx = idx_top[idx]
        results.append({
            "index": int(doc_idx),
            "title": str(df.loc[doc_idx, "title"]) if "title" in df.columns else f"Dokumen {doc_idx}",
            "url": str(df.loc[doc_idx, "url"]) if "url" in df.columns else "",
            "pdf": str(df.loc[doc_idx, "pdf"]) if "pdf" in df.columns else "",
            "abstract": corpus_texts[doc_idx][:500] + "..." if len(corpus_texts[doc_idx]) > 500 else corpus_texts[doc_idx],
            "semantic": float(semantic_scores[idx]),
            "lexical": float(lexical_scores[idx]),
            "structure": float(structural_scores[idx]),
            "final_score": float(final_scores[idx]),
            "similarity": float(final_scores[idx]) * 100
        })
    
    return results

# ---------------------------
# API endpoints
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    """Endpoint untuk cek kesehatan server"""
    try:
        load_models()
        return jsonify({
            "status": "healthy",
            "message": "Server berjalan normal",
            "models_loaded": all([
                tfidf_vectorizer is not None,
                tfidf_matrix is not None,
                df is not None,
                corpus_embeddings is not None,
                model is not None
            ])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint utama untuk prediksi plagiarisme"""
    try:
        query = ""
        
        # Cek apakah ada file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                if file.content_length > 10 * 1024 * 1024:  # 10MB limit
                    return jsonify({"error": "File terlalu besar (maks 10MB)"}), 400
                
                try:
                    query = extract_text_from_file(file)
                except Exception as e:
                    return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 400
            else:
                # Kalau tidak ada file, gunakan text input
                query = request.form.get("query", "").strip()
                if not query:
                    query = request.form.get("text_input", "").strip()
        else:
            # Kalau tidak ada file upload, gunakan text biasa
            data = request.get_json(silent=True)
            if data:
                query = data.get("query", data.get("text_input", "")).strip()
            else:
                query = request.form.get("query", request.form.get("text_input", "")).strip()
        
        if not query:
            return jsonify({"error": "Teks kosong! Silakan masukkan teks atau unggah file."}), 400
        
        if len(query) < 10:
            return jsonify({"error": "Teks terlalu pendek (minimal 10 karakter)"}), 400
        
        # Gunakan fungsi plagiarisme
        results = check_plagiarism(query, top_k=10)
        
        # Format response
        response_data = {
            "query_length": len(query),
            "query_preview": query[:200] + "..." if len(query) > 200 else query,
            "results": []
        }
        
        for r in results:
            response_data["results"].append({
                "index": r["index"],
                "title": r["title"],
                "url": r["url"],
                "pdf": r["pdf"],
                "abstract": r["abstract"],
                "semantic": round(r["semantic"], 4),
                "lexical": round(r["lexical"], 4),
                "structure": round(r["structure"], 4),
                "final_score": round(r["final_score"], 4),
                "similarity": round(r["similarity"], 2)
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Handler untuk error 404
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint tidak ditemukan"}), 404

# Handler untuk error 500
@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Terjadi kesalahan internal server"}), 500

# Untuk development lokal
if __name__ == "__main__":
    print("Loading models untuk development...")
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    # Untuk production (Vercel/Gunicorn)
    print("Initializing application for production...")