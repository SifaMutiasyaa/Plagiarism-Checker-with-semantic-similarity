# app.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import PyPDF2
from docx import Document
import io

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------------------
# Configuration / weights (SAMA DENGAN NOTEBOOK)
# ---------------------------
WEIGHT_LEXICAL = 0.6      # Sama dengan notebook: 0.6
WEIGHT_STRUCTURE = 0.25   # Sama dengan notebook: 0.25
WEIGHT_SEMANTIC = 0.15    # Sama dengan notebook: 0.15

TOP_K_TFIDF = 20  # candidate pool from TF-IDF before rerank

# ---------------------------
# Helpers
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
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_stream):
    """Ekstrak teks dari DOCX"""
    doc = Document(file_stream)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file):
    """Deteksi tipe file dan ekstrak teks"""
    filename = file.filename.lower()
    file_stream = io.BytesIO(file.read())
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_stream)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file_stream)
    elif filename.endswith('.txt'):
        file_stream.seek(0)
        return file_stream.read().decode('utf-8')
    else:
        raise ValueError("Format file tidak didukung. Gunakan PDF, DOCX, atau TXT.")

# ---------------------------
# Load models & data
# ---------------------------
tfidf_vectorizer = joblib.load("models/tfidf.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

# CSV
df = pd.read_csv("models/corpus_clean.csv")
df = df.fillna("")

if "abstract" not in df.columns:
    raise ValueError("CSV tidak memiliki kolom 'abstract'.")

corpus_texts = df["abstract"].astype(str).tolist()

# Embeddings
corpus_embeddings = np.load("models/corpus_embeddings.npy")

# Gunakan model yang SAMA dengan notebook
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ---------------------------
# Search functions (SAMA DENGAN NOTEBOOK)
# ---------------------------
def tfidf_search(query, top_n=TOP_K_TFIDF):
    """Cari dokumen menggunakan TF-IDF (lexical)"""
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx_sorted = scores.argsort()[::-1][:top_n]
    return idx_sorted, scores[idx_sorted]

def compute_semantic_score(query, candidate_indices):
    """Hitung semantic similarity menggunakan SentenceTransformer"""
    q_emb = model.encode([query])[0]
    subset = corpus_embeddings[candidate_indices]
    sims = cosine_similarity([q_emb], subset).flatten()
    return sims

def check_plagiarism(query, top_k=5):
    """Fungsi utama untuk cek plagiarisme (SAMA DENGAN NOTEBOOK)"""
    # 1. LEXICAL (TF-IDF)
    idx_top, lexical_scores = tfidf_search(query, top_n=TOP_K_TFIDF)
    
    # 2. SEMANTIC (SentenceTransformer)
    semantic_scores = compute_semantic_score(query, idx_top)
    
    # 3. STRUCTURAL (Jaccard)
    structural_scores = np.array([jaccard_similarity(query, corpus_texts[i]) for i in idx_top])
    
    # 4. GABUNGKAN SCORE (SAMA DENGAN NOTEBOOK)
    final_scores = (
        lexical_scores * WEIGHT_LEXICAL +
        structural_scores * WEIGHT_STRUCTURE +
        semantic_scores * WEIGHT_SEMANTIC
    )
    
    # 5. Ambil top-k hasil
    top_indices = final_scores.argsort()[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_indices):
        doc_idx = idx_top[idx]  # Index asli di corpus
        results.append({
            "index": int(doc_idx),
            "title": str(df.loc[doc_idx, "title"]) if "title" in df.columns else "",
            "url": str(df.loc[doc_idx, "url"]) if "url" in df.columns else "",
            "pdf": str(df.loc[doc_idx, "pdf"]) if "pdf" in df.columns else "",
            "abstract": corpus_texts[doc_idx],
            "semantic": float(semantic_scores[idx]),
            "lexical": float(lexical_scores[idx]),
            "structure": float(structural_scores[idx]),
            "final_score": float(final_scores[idx]),
            "similarity": float(final_scores[idx]) * 100  # Persentase untuk frontend
        })
    
    return results

# ---------------------------
# API endpoints
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    query = ""
    
    # Cek apakah ada file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                query = extract_text_from_file(file)
            except Exception as e:
                return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 400
        else:
            return jsonify({"error": "File tidak dipilih!"}), 400
    else:
        # Kalau tidak ada file, gunakan text input
        query = request.form.get("query", "").strip()
        if not query:
            query = request.form.get("text_input", "").strip()
    
    if not query:
        return jsonify({"error": "Teks kosong!"}), 400
    
    try:
        # Gunakan fungsi yang sama dengan notebook
        results = check_plagiarism(query, top_k=10)
        
        # Format respons untuk frontend
        response_data = []
        for r in results:
            response_data.append({
                "index": r["index"],
                "title": r["title"],
                "url": r["url"],
                "pdf": r["pdf"],
                "abstract": r["abstract"],
                "semantic": round(r["semantic"], 4),
                "lexical": round(r["lexical"], 4),
                "structure": round(r["structure"], 4),
                "final_score": round(r["final_score"], 4),
                "similarity": round(r["similarity"], 2)  # Persentase
            })
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)