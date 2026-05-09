#!/usr/bin/env python3
"""
Stock Prediction REST API
─────────────────────────
Endpoints:
  GET  /health                  → liveness check
  POST /predict/nasdaq_price    → Nasdaq next-day Close price
  POST /predict/vn_price        → Vietnam next-day Close price
  POST /predict/vn_signal       → Vietnam BUY / SELL signal probability

POST body (JSON): { "window": [[col1, col2, ...], ...] }
                   WINDOW_SIZE × n_features  (30 rows)

Run:  python app.py
"""

import os, json
import numpy as np
from flask import Flask, request, jsonify, abort
import tensorflow as tf

app = Flask(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {}
MODEL_FILES = {
    "nasdaq_price": ("nasdaq_model_task11.keras",  4),
    "vn_price":     ("vn_model_task21.keras",       3),
    "vn_buy":       ("vn_model_task31_buy.keras",   3),
    "vn_sell":      ("vn_model_task32_sell.keras",  3),
}

def _load_models():
    for name, (path, _) in MODEL_FILES.items():
        if os.path.exists(path):
            MODELS[name] = tf.keras.models.load_model(path)
            print(f"  Loaded: {name}")
        else:
            print(f"  Missing model file: {path}")

def _normalize(X, close_col):
    X_n = X.astype("float32").copy()
    for j in range(X_n.shape[1]):
        mn  = X_n[:, j].min()
        rng = max(float(X_n[:, j].max() - mn), 1e-8)
        X_n[:, j] = (X_n[:, j] - mn) / rng
    c_mn  = float(X[:, close_col].min())
    c_rng = max(float(X[:, close_col].max() - c_mn), 1e-8)
    return X_n[None], c_mn, c_rng

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": sorted(MODELS)})

# ── Nasdaq price ──────────────────────────────────────────────────────────────
@app.route("/predict/nasdaq_price", methods=["POST"])
def predict_nasdaq_price():
    """Body: {"window": [[Low,Open,Volume,High,Close,AdjClose], ...]}  30 rows"""
    data = request.get_json(force=True)
    if "window" not in data:
        abort(400, "Missing field: window")
    w = np.array(data["window"], dtype="float32")
    if w.ndim != 2 or w.shape[1] != 6:
        abort(400, "window must be shape (30, 6)")
    X_n, c_mn, c_rng = _normalize(w, close_col=4)
    y_n = float(MODELS["nasdaq_price"].predict(X_n, verbose=0).flat[0])
    return jsonify({"predicted_close": round(y_n * c_rng + c_mn, 4)})

# ── Vietnam price ─────────────────────────────────────────────────────────────
@app.route("/predict/vn_price", methods=["POST"])
def predict_vn_price():
    """Body: {"window": [[Open,High,Low,Close,Volume], ...]}  30 rows"""
    data = request.get_json(force=True)
    if "window" not in data:
        abort(400, "Missing field: window")
    w = np.array(data["window"], dtype="float32")
    if w.ndim != 2 or w.shape[1] != 5:
        abort(400, "window must be shape (30, 5)")
    X_n, c_mn, c_rng = _normalize(w, close_col=3)
    y_n = float(MODELS["vn_price"].predict(X_n, verbose=0).flat[0])
    return jsonify({"predicted_close": round(y_n * c_rng + c_mn, 4)})

# ── Vietnam BUY / SELL signal ─────────────────────────────────────────────────
@app.route("/predict/vn_signal", methods=["POST"])
def predict_vn_signal():
    """
    Body: {"window": [[Open,High,...,OBV], ...], "signal_type": "buy"|"sell"}
    30 rows × 17 features (VN_FEATURES + 12 technical indicators)
    """
    data = request.get_json(force=True)
    if "window" not in data:
        abort(400, "Missing field: window")
    w = np.array(data["window"], dtype="float32")
    if w.ndim != 2 or w.shape[1] != 17:
        abort(400, "window must be shape (30, 17)")
    key = "vn_buy" if data.get("signal_type", "buy") == "buy" else "vn_sell"
    X_n, _, _ = _normalize(w, close_col=3)
    prob = float(MODELS[key].predict(X_n, verbose=0).flat[0])
    return jsonify({"probability": round(prob, 4), "signal": int(prob >= 0.5)})

if __name__ == "__main__":
    print("Loading models…")
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
