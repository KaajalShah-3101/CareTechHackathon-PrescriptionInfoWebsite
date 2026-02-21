"""
config.py  —  All project settings in one place.
Change something here and it updates everywhere.
"""

from pathlib import Path

# ── Folder layout ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
CSV_PATH = DATA_DIR / "personalized_medication_dataset.csv"

# ── Saved model files (auto-created when you run: python model.py) ─────────────
MODEL_PATH = MODEL_DIR / "pedidose_model.pkl"
META_PATH  = MODEL_DIR / "pedidose_meta.json"

# ── API server ─────────────────────────────────────────────────────────────────
API_HOST   = "0.0.0.0"
API_PORT   = 8000
API_RELOAD = True

# ── Exact column names from the Kaggle CSV ─────────────────────────────────────
COL_AGE        = "Age"                     # integer years (18-90 in dataset)
COL_WEIGHT     = "Weight_kg"               # float, kilograms
COL_BMI        = "BMI"
COL_MEDICATION = "Recommended_Medication"  # Amoxicillin | Amlodipine | Ibuprofen
COL_DOSAGE     = "Dosage"                  # "5 mg" | "200 mg" | "400 mg" | "500 mg"
COL_EFFECTIVE  = "Treatment_Effectiveness"
COL_ADVERSE    = "Adverse_Reactions"
COL_CHRONIC    = "Chronic_Conditions"
COL_DIAGNOSIS  = "Diagnosis"
COL_GENDER     = "Gender"

# ── Pediatric dosing safety guidelines ────────────────────────────────────────
# Source: AAP / BNF for Children / Taketomo Pediatric Dosage Handbook
# Used to safety-check doses for CHILD patients (the ML model learns
# patterns from the dataset; these caps enforce hard clinical limits).
PEDIATRIC_GUIDELINES = {
    "amoxicillin": dict(min_kg=25,   max_kg=45,  abs_max=500, min_age_mo=0),
    "amlodipine":  dict(min_kg=0.05, max_kg=0.1, abs_max=5,   min_age_mo=12),
    "ibuprofen":   dict(min_kg=5,    max_kg=10,  abs_max=400, min_age_mo=6),
}