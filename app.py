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
# Configuration
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
# Global variables untuk models
# ---------------------------
tfidf_vectorizer = None
tfidf_matrix = None
df = None
corpus_texts = None
corpus_embeddings = None
model = None

def load_models():
    global tfidf_vectorizer, tfidf_matrix, df, corpus_texts, corpus_embeddings, model
    try:
        if tfidf_vectorizer is None:
            print("Memuat TF-IDF vectorizer...")
            tfidf_vectorizer = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
            
        if tfidf_matrix is None:
            print("Memuat TF-IDF matrix...")
            tfidf_matrix = joblib.load(os.path.join(models_dir, "tfidf_matrix.pkl"))
            
        if df is None:
            print("Memuat CSV corpus...")
            df = pd.read_csv(os.path.join(models_dir, "corpus_clean.csv"))
            df = df.fillna("")
            corpus_texts = df["abstract"].astype(str).tolist()
            
        if corpus_embeddings is None:
            print("Memuat corpus embeddings...")
            corpus_embeddings = np.load(os.path.join(models_dir, "corpus_embeddings.npy"))
            
        if model is None:
            print("Memuat SentenceTransformer...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        print("Model semua berhasil dimuat!")
        return True
    except Exception as e:
        print("Error load model:", str(e))
        traceback.print_exc()
        return False


# ---------------------------
# Search functions
# ---------------------------
def tfidf_search(query, top_n=TOP_K_TFIDF):
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx_sorted = scores.argsort()[::-1][:top_n]
    return idx_sorted, scores[idx_sorted]

def compute_semantic_score(query, candidate_indices):
    q_emb = model.encode([query], show_progress_bar=False)[0]
    subset = corpus_embeddings[candidate_indices]
    sims = cosine_similarity([q_emb], subset).flatten()
    return sims

def check_plagiarism(query, top_k=5):
    if not load_models():
        raise Exception("Gagal memuat model")
    
    idx_top, lexical_scores = tfidf_search(query, top_n=TOP_K_TFIDF)
    semantic_scores = compute_semantic_score(query, idx_top)
    structural_scores = np.array([jaccard_similarity(query, corpus_texts[i]) for i in idx_top])
    
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
            "abstract": corpus_texts[doc_idx][:500] + "..." if len(corpus_texts[doc_idx]) > 500 else corpus_texts[doc_idx],
            "semantic": float(semantic_scores[idx]),
            "lexical": float(lexical_scores[idx]),
            "structure": float(structural_scores[idx]),
            "final_score": float(final_scores[idx]),
            "similarity": float(final_scores[idx]) * 100
        })
    
    return results


# ---------------------------
# Endpoints
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "running", "models_loaded": load_models()})

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

        results = check_plagiarism(query, top_k=10)

        return jsonify({
            "query_length": len(query),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(static_dir, path)


# ---------------------------
# RUN SERVER (WAJIB UNTUK RAILWAY)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
