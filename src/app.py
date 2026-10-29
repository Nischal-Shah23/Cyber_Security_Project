"""Minimal Flask scoring API for the trained research model.

This is a research/demo integration surface, not a hardened production WAF.
It does not proxy requests to an upstream target. Keep it on localhost or an
isolated lab network unless you add authentication, TLS, limits and proxy logic.
"""
from __future__ import annotations
import os, time
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.features import extract_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/shadowguard_research_model.joblib"))
app = Flask(__name__)
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return jsonify({"status":"ok", "model_loaded": model is not None})

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error":"Model not loaded. Train first using python -m src.train --data data/raw"}), 503
    body = request.get_json(silent=True) or {}
    raw = body.get("request")
    if not isinstance(raw, str) or not raw.strip():
        return jsonify({"error":"JSON body must contain a non-empty string field named 'request'."}), 400
    if len(raw) > 20000:
        return jsonify({"error":"Request text exceeds 20000 characters."}), 413
    row = {"request": raw, **extract_features(raw)}
    started = time.perf_counter()
    frame = pd.DataFrame([row])
    probability = float(model.predict_proba(frame)[0,1])
    threshold = float(os.getenv("ATTACK_THRESHOLD", "0.5"))
    label = "attack" if probability >= threshold else "normal"
    return jsonify({
        "prediction": label,
        "attack_probability": round(probability, 6),
        "threshold": threshold,
        "decision": "BLOCK" if label == "attack" else "ALLOW",
        "latency_ms": round((time.perf_counter()-started)*1000, 3),
        "disclaimer":"Research demonstration; validate on representative traffic before any deployment."
    })

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
