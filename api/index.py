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
import sys
import json

# ---------------------------
# Fix path untuk model di Vercel
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, 'models')

app = Flask(__name__, 
            static_folder=os.path.join(parent_dir, 'static'), 
            template_folder=os.path.join(parent_dir, 'templates'),
            static_url_path="/static")

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
        raise ValueError(f"Failed to read PDF: {str(e)}")

def extract_text_from_docx(file_stream):
    try:
        doc = Document(file_stream)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {str(e)}")

def extract_text_from_file(file):
    filename = file.filename.lower()
    file_content = file.read()
    
    if not file_content:
        raise ValueError("Empty file")
    
    file_stream = io.BytesIO(file_content)
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_stream)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file_stream)
    elif filename.endswith('.txt'):
        file_stream.seek(0)
        return file_stream.read().decode('utf-8', errors='ignore').strip()
    else:
        raise ValueError("Unsupported format. Use PDF, DOCX, or TXT.")

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
            print("Loading TF-IDF vectorizer...")
            tfidf_vectorizer = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
            
        if tfidf_matrix is None:
            print("Loading TF-IDF matrix...")
            tfidf_matrix = joblib.load(os.path.join(models_dir, "tfidf_matrix.pkl"))
            
        if df is None:
            print("Loading CSV data...")
            df = pd.read_csv(os.path.join(models_dir, "corpus_clean.csv"))
            df = df.fillna("")
            
            if "abstract" not in df.columns:
                raise ValueError("CSV tidak memiliki kolom 'abstract'.")
            
            corpus_texts = df["abstract"].astype(str).tolist()
            
        if corpus_embeddings is None:
            print("Loading embeddings...")
            corpus_embeddings = np.load(os.path.join(models_dir, "corpus_embeddings.npy"))
            
        if model is None:
            print("Loading SentenceTransformer model...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        traceback.print_exc()
        raise

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
    load_models()
    
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
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error loading template: {str(e)}", 500

@app.route("/health")
def health():
    try:
        load_models()
        return jsonify({
            "status": "healthy",
            "message": "Server is running",
            "models_loaded": True
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return jsonify({
            "message": "Send POST request with 'query' parameter or upload file",
            "example": {
                "POST": "/predict",
                "body": {"query": "text to check"},
                "or": "upload file with key 'file'"
            }
        })
    
    try:
        query = ""
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                if file.content_length and file.content_length > 10 * 1024 * 1024:
                    return jsonify({"error": "File too large (max 10MB)"}), 400
                query = extract_text_from_file(file)
            else:
                query = request.form.get("query", "").strip()
                if not query:
                    query = request.form.get("text_input", "").strip()
        else:
            if request.is_json:
                data = request.get_json()
                query = data.get("query", data.get("text_input", "")).strip()
            else:
                query = request.form.get("query", request.form.get("text_input", "")).strip()
        
        if not query:
            return jsonify({"error": "No text provided"}), 400
        
        if len(query) < 10:
            return jsonify({"error": "Text too short (min 10 characters)"}), 400
        
        results = check_plagiarism(query, top_k=10)
        
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
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "available_endpoints": ["/", "/health", "/predict"]}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Vercel membutuhkan variabel 'app'
app = app