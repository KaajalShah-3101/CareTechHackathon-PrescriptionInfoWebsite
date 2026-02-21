"""
data_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads personalized_medication_dataset.csv and returns
a clean DataFrame ready for training.

What this file knows about the dataset:
  - 1000 rows, 17 columns
  - Age in YEARS (18–90), not months
  - Weight_kg (50–120 kg)
  - 3 drugs: Amoxicillin, Amlodipine, Ibuprofen
  - Dosage is a string like "200 mg" — we extract the number
  - 263 rows missing Recommended_Medication → drop them
  - 195 rows missing Dosage → drop them
  - Treatment_Effectiveness: Very Effective / Effective / Neutral / Ineffective
  - Adverse_Reactions: Yes / No
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .config import CSV_PATH, PEDIATRIC_GUIDELINES


def load_clean_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Loads and cleans the Kaggle CSV.
    Returns a DataFrame with these columns ready for training:
        age_years, weight_kg, bmi, drug, dose_mg,
        effective_score, had_adverse, chronic_flag,
        [all the encoded columns the model needs]
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n❌  File not found: {csv_path}"
            f"\n    → Copy your CSV into the data/ folder."
            f"\n    → It should be named: personalized_medication_dataset.csv"
        )

    df = pd.read_csv(csv_path)
    print(f"[data_loader] Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── 1. Drop rows missing the two things we must have ──────────────────
    before = len(df)
    df = df.dropna(subset=["Recommended_Medication", "Dosage"])
    print(f"[data_loader] Dropped {before - len(df)} rows with missing medication/dosage → {len(df)} remain")

    # ── 2. Parse dosage string → float  ("200 mg" → 200.0) ───────────────
    df["dose_mg"] = (
        df["Dosage"]
        .str.extract(r"(\d+\.?\d*)")   # grab the number
        .astype(float)
    )
    df = df[df["dose_mg"] > 0]         # drop any that didn't parse

    # ── 3. Standardise drug names (lowercase, stripped) ───────────────────
    df["drug"] = df["Recommended_Medication"].str.strip().str.lower()

    # ── 4. Age: already in years — keep as-is, also make months column ────
    df["age_years"]  = df["Age"].astype(float)
    df["age_months"] = df["age_years"] * 12   # for pediatric safety checks

    # ── 5. Encode Treatment_Effectiveness → numeric score ─────────────────
    effectiveness_map = {
        "Very Effective": 3,
        "Effective":      2,
        "Neutral":        1,
        "Ineffective":    0,
    }
    df["effective_score"] = df["Treatment_Effectiveness"].map(effectiveness_map).fillna(1)

    # ── 6. Adverse reactions → binary ─────────────────────────────────────
    df["had_adverse"] = (df["Adverse_Reactions"].str.strip() == "Yes").astype(int)

    # ── 7. Chronic conditions → binary flag ───────────────────────────────
    df["chronic_flag"] = df["Chronic_Conditions"].notna().astype(int)

    # ── 8. Gender → binary ────────────────────────────────────────────────
    df["is_female"] = (df["Gender"].str.strip() == "Female").astype(int)

    # ── 9. Drop rows with impossible values ───────────────────────────────
    df = df[(df["Weight_kg"] > 0) & (df["Weight_kg"] <= 200)]
    df = df[(df["age_years"] >= 0) & (df["age_years"] <= 120)]

    df = df.reset_index(drop=True)

    print(f"[data_loader] Clean rows: {len(df):,}")
    print(f"[data_loader] Drugs:  {df['drug'].value_counts().to_dict()}")
    print(f"[data_loader] Doses:  {sorted(df['dose_mg'].unique().tolist())}")
    print()

    return df


if __name__ == "__main__":
    df = load_clean_data()
    print(df[["age_years", "Weight_kg", "drug", "dose_mg",
              "Treatment_Effectiveness", "had_adverse"]].head(10))