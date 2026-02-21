# app.py
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for frontend

# ─── Load the dataset ───
BASE_DIR = os.path.dirname(__file__)  # folder containing app.py
DATA_PATH = os.path.join(BASE_DIR, 'pedidose', 'data', 'personalized_medication_dataset.csv')

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Normalize column names for easier access
df.columns = [col.lower().strip() for col in df.columns]

# Example relevant columns from the dataset:
# 'Recommended_Medication', 'min_dose_mg', 'max_dose_mg', 'recommended_dose_mg', 'condition', 'notes'

@app.route('/api/dosage', methods=['POST'])
def get_dosage():
    data = request.get_json()
    medication = data.get('medication', '').strip().lower()
    weight     = data.get('weight')
    age        = data.get('age')
    sex        = data.get('sex', '').lower()
    condition  = data.get('condition', '').strip()

    # Validate input
    if not medication or weight is None or age is None:
        return jsonify({'error': 'Medication, weight, and age are required'}), 400

    # Lookup medication in dataset
    med_row = df[df['Recommended_Medication'].str.lower() == medication]

    if med_row.empty:
        return jsonify({'error': 'Medication not found in dataset'}), 404

    med_row = med_row.iloc[0]

    # Optional: adjust dose by weight (if dataset is per kg)
    # Here we assume dataset doses are per adult average weight (70 kg)
    weight_factor = weight / 70
    min_dose = round(float(med_row.get('min_dose_mg', 0)) * weight_factor, 1)
    max_dose = round(float(med_row.get('max_dose_mg', 0)) * weight_factor, 1)
    rec_dose = round(float(med_row.get('recommended_dose_mg', 0)) * weight_factor, 1)

    response = {
        "medication": med_row.get('med_name', 'Unknown'),
        "min_dose": min_dose,
        "max_dose": max_dose,
        "recommended_dose": rec_dose,
        "unit": "mg",
        "condition": med_row.get('condition', None),
        "notes": med_row.get('notes', None)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
