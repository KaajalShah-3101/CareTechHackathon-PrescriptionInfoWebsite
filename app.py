"""
app.py  —  Unified Backend Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Serves:
  1. API endpoints for dose safety checking
  2. Frontend HTML interface
  3. ML model inference via pedidose package
"""

import traceback
import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Import pedidose model functions
from pedidose.config import API_HOST, API_PORT, API_RELOAD
from pedidose.model import check_dose, is_model_ready, get_supported_drugs

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for frontend


# ── Helpers ───────────────────────────────────────────────────────
def ok(data):
    """Return success response"""
    return jsonify({"success": True, "data": data}), 200

def err(msg, code=400):
    """Return error response"""
    return jsonify({"success": False, "error": msg}), code


# ── Root: Serve frontend ──────────────────────────────────────────
@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve the frontend HTML"""
    try:
        with open('frontend.html', 'r', encoding='utf-8') as f:
            html = f.read()
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return err("Frontend not found", 404)


# ── GET /api/health ───────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """Check if model is ready"""
    ready = is_model_ready()
    return ok({
        "model_ready": ready,
        "message": "Ready ✓" if ready else "Run `python pedidose/model.py` first"
    })


# ── GET /api/drugs ────────────────────────────────────────────────
@app.route("/api/drugs", methods=["GET"])
def drugs():
    """List supported medications"""
    if not is_model_ready():
        return err("Model not trained yet", 503)
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
    """Check if a prescribed dose is safe for a patient, or lookup safe range"""
    if not is_model_ready():
        return err("Model not trained. Run `python pedidose/model.py` first", 503)

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
        return err("weight_kg must be between 0.1 and 200")
    if not (0 <= age <= 120):
        return err("age_years must be between 0 and 120")
    if prescribed < 0:
        return err("prescribed_mg must be greater than or equal to 0")

    # Optional fields
    bmi     = float(body["bmi"])     if "bmi"     in body else None
    chronic = bool(body["chronic"])  if "chronic" in body else False
    adverse = bool(body["adverse"])  if "adverse" in body else False
    female  = bool(body["female"])   if "female"  in body else False

    try:
        # If prescribed_mg is 0, use 1 for the model but indicate it's a lookup
        dose_for_model = prescribed if prescribed > 0 else 1
        result = check_dose(
            drug=body["drug"], weight_kg=weight, age_years=age,
            prescribed_mg=dose_for_model, bmi=bmi,
            chronic=chronic, adverse=adverse, female=female,
        )
        # Normalize field names for frontend compatibility
        result['drug'] = body["drug"]  # Add the drug name
        if 'guideline_min_mg' in result:
            result['min_dose_mg'] = result.pop('guideline_min_mg')
        if 'guideline_max_mg' in result:
            result['max_dose_mg'] = result.pop('guideline_max_mg')
        return ok(result)
    except Exception as e:
        traceback.print_exc()
        return err(f"Prediction error: {str(e)}", 500)


# ── POST /api/train ───────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train_route():
    """Retrain the model with updated data"""
    try:
        from pedidose.model import train
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
    print(f"  MedCheck Backend  —  http://{API_HOST}:{API_PORT}")
    print(f"  Model ready: {is_model_ready()}")
    if not is_model_ready():
        print("  ⚠  Run `python pedidose/model.py` first!")
    print(f"  Frontend: http://localhost:{API_PORT}/")
    print("="*50 + "\n")
    app.run(host=API_HOST, port=API_PORT, debug=API_RELOAD)
