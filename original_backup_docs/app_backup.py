"""
app.py
Flask API serving precomputed JSON files and model predictions
for the flight delay D3 dashboard.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "Outputs"

# ── Load precomputed JSON files once at startup ───────────────────────────────
print("Loading precomputed data ...")

with open(STATIC_DIR / "nodes.json")       as f: NODES       = json.load(f)
with open(STATIC_DIR / "edges.json")       as f: EDGES       = json.load(f)
with open(STATIC_DIR / "breakdown.json")   as f: BREAKDOWN   = json.load(f)
with open(STATIC_DIR / "propagation.json") as f: PROPAGATION = json.load(f)

# Index nodes by airport code for fast lookup
NODES_BY_CODE = {n["id"]: n for n in NODES}

print(f"  Loaded {len(NODES)} nodes, {len(EDGES)} edges")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model ...")
try:
    MODEL = joblib.load(OUTPUT_DIR / "gb_pipeline_50k.pkl")
    print("  Gradient Boosting model loaded")
except Exception as e:
    MODEL = None
    print(f"  Warning: model not loaded — {e}")

print("Ready.\n")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/network")
def network():
    """Returns all nodes and edges for the force-directed graph."""
    return jsonify({
        "nodes": NODES,
        "edges": EDGES
    })

@app.route("/api/breakdown")
def breakdown():
    return jsonify(BREAKDOWN)

@app.route("/api/airport/<code>")
def airport(code):
    """Returns stats + delay breakdown for one airport."""
    code = code.upper()
    node = NODES_BY_CODE.get(code)
    if not node:
        return jsonify({"error": f"Airport {code} not found"}), 404

    return jsonify({
        "code":      code,
        "stats":     node,
        "breakdown": BREAKDOWN.get(code, {})
    })


@app.route("/api/propagation/<code>")
def propagation(code):
    """Returns correlated airports for a given hub."""
    code = code.upper()
    connected = PROPAGATION.get(code, [])
    return jsonify({
        "hub":       code,
        "connected": connected
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts flight conditions as JSON, returns delay probability.

    Expected JSON body example:
    {
        "MONTH": 7,
        "DAY_OF_WEEK": 5,
        "FL_DAY": 15,
        "CRS_DEP_HOUR": 8,
        "CRS_ARR_HOUR": 11,
        "CRS_ELAPSED_TIME": 180,
        "DISTANCE": 1200,
        "OP_UNIQUE_CARRIER": "DL",
        "ORIGIN": "ATL",
        "DEST": "JFK",
        "ORIGIN_STATE_ABR": "GA",
        "DEST_STATE_ABR": "NY",
        "ORIGIN_Total_Operations": 2500,
        "DEST_Total_Operations": 2800
    }
    """
    if MODEL is None:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required_fields = ["MONTH", "DAY_OF_WEEK", "DISTANCE"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing {field}"}), 400

    # Build a single-row DataFrame with all expected feature columns
    # Missing features get None — the model's imputer handles them
    feature_cols = [
        "MONTH", "DAY_OF_WEEK", "FL_DAY", "CRS_DEP_HOUR", "CRS_ARR_HOUR",
        "CRS_ELAPSED_TIME", "DISTANCE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "ORIGIN_STATE_ABR", "DEST_STATE_ABR",
        "ORIGIN_Air_Carrier", "ORIGIN_Air_Taxi", "ORIGIN_General_Aviation",
        "ORIGIN_Military_Itinerant", "ORIGIN_Total_Itinerant",
        "ORIGIN_Local_Civil", "ORIGIN_Local_Military", "ORIGIN_Local_Total",
        "ORIGIN_Total_Operations",
        "DEST_Air_Carrier", "DEST_Air_Taxi", "DEST_General_Aviation",
        "DEST_Military_Itinerant", "DEST_Total_Itinerant",
        "DEST_Local_Civil", "DEST_Local_Military", "DEST_Local_Total",
        "DEST_Total_Operations",
        "ORIGIN_TMAX", "ORIGIN_TMIN", "ORIGIN_TAVG", "ORIGIN_PRCP",
        "ORIGIN_SNWD", "ORIGIN_AWND", "ORIGIN_WSF2",
        "DEST_TMAX", "DEST_TMIN", "DEST_TAVG", "DEST_PRCP",
        "DEST_SNWD", "DEST_AWND", "DEST_WSF2"
    ]

    row = {col: data.get(col, None) for col in feature_cols}
    input_df = pd.DataFrame([row])

    try:
        proba = MODEL.predict_proba(input_df)[0][1]
        delayed = bool(proba >= 0.35)   # threshold from model evaluation
        return jsonify({
            "delay_probability": round(float(proba), 4),
            "predicted_delayed": delayed,
            "threshold_used":    0.35
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/summary")
def summary():
    """Returns high-level dataset summary for dashboard header."""
    total_flights = sum(n["total_flights"] for n in NODES)
    avg_delay_rate = round(
        sum(n["pct_delayed"] for n in NODES) / len(NODES), 4
    )
    return jsonify({
        "total_airports":  len(NODES),
        "total_routes":    len(EDGES),
        "total_flights":   total_flights,
        "avg_delay_rate":  avg_delay_rate,
        "model":           "Gradient Boosting (AUC 0.7065)",
        "years":           "2015-2019, 2022-2024"
    })

@app.route("/health")
def health():
    return "OK"

# ── Run ───────────────────────────────────────────────────────────────────────
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)