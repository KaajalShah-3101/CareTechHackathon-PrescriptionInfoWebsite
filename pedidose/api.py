"""
api.py  —  Flask web server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The "waiter" between your HTML frontend and the ML model.
Receives HTTP requests, calls model.py, sends back JSON.

Endpoints:
  GET  /api/health       — is the server + model ready?
  GET  /api/drugs        — list of supported drugs
  POST /api/check        — check if a dose is safe  ← main one
  POST /api/train        — retrain model (after updating CSV)

Run with:  python api.py
"""

import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from .config import API_HOST, API_PORT, API_RELOAD
from .model import check_dose, is_model_ready, get_supported_drugs

app = Flask(__name__)
CORS(app)   # allows your HTML page to call this server


# ── Helpers ───────────────────────────────────────────────────────
def ok(data):
    return jsonify({"success": True,  "data": data}), 200

def err(msg, code=400):
    return jsonify({"success": False, "error": msg}), code


# ── GET /api/health ───────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    ready = is_model_ready()
    return ok({
        "model_ready": ready,
        "message": "Ready ✓" if ready else "Run `python model.py` first"
    })


# ── GET /api/drugs ────────────────────────────────────────────────
@app.route("/api/drugs", methods=["GET"])
def drugs():
    if not is_model_ready():
        return err("Model not trained yet.", 503)
    return ok({"drugs": get_supported_drugs()})


# ── POST /api/check  ← frontend calls this ────────────────────────
# Required fields:  drug, weight_kg, age_years, prescribed_mg
# Optional fields:  bmi, chronic, adverse, female
#
# Returns:  status ("safe"/"overdose"/"underdose"),
#           color  ("green"/"red"/"yellow"),
#           message, dose ranges, confidence
@app.route("/api/check", methods=["POST"])
def check():
    if not is_model_ready():
        return err("Model not trained. Run `python model.py` first.", 503)

    body = request.get_json(silent=True) or {}

    # Validate required fields
    for field in ["drug", "weight_kg", "age_years", "prescribed_mg"]:
        if field not in body:
            return err(f"Missing required field: '{field}'")

    try:
        weight       = float(body["weight_kg"])
        age          = float(body["age_years"])
        prescribed   = float(body["prescribed_mg"])
    except (ValueError, TypeError):
        return err("weight_kg, age_years, and prescribed_mg must be numbers")

    if not (0 < weight <= 200):
        return err("weight_kg must be between 0 and 200")
    if not (0 <= age <= 120):
        return err("age_years must be between 0 and 120")
    if prescribed <= 0:
        return err("prescribed_mg must be greater than 0")

    # Optional fields
    bmi     = float(body["bmi"])     if "bmi"     in body else None
    chronic = bool(body["chronic"])  if "chronic" in body else False
    adverse = bool(body["adverse"])  if "adverse" in body else False
    female  = bool(body["female"])   if "female"  in body else False

    try:
        result = check_dose(
            drug=body["drug"], weight_kg=weight, age_years=age,
            prescribed_mg=prescribed, bmi=bmi,
            chronic=chronic, adverse=adverse, female=female,
        )
        return ok(result)
    except Exception as e:
        traceback.print_exc()
        return err(f"Prediction error: {str(e)}", 500)


# ── POST /api/train ───────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train_route():
    try:
        from model import train
        train()
        return ok({"message": "Model retrained successfully ✓"})
    except FileNotFoundError as e:
        return err(str(e), 404)
    except Exception as e:
        traceback.print_exc()
        return err(f"Training failed: {str(e)}", 500)


# ── Start server ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  PediDose API  —  http://localhost:" + str(API_PORT))
    print("  Model ready: " + str(is_model_ready()))
    if not is_model_ready():
        print("  ⚠  Run `python model.py` first!")
    print("="*50 + "\n")
    app.run(host=API_HOST, port=API_PORT, debug=API_RELOAD)