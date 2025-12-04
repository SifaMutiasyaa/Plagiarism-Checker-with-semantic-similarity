from flask import Flask, jsonify
from flask_cors import CORS
from app import app  # import Flask app dari app.py

CORS(app)

# Wrapper untuk Vercel
def handler(request, response):
    return app(request.environ, response.start_response)
