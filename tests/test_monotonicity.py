"""Verification of monotonic constraints in trained credit risk model."""

import pytest
import joblib
import pandas as pd
import numpy as np


def test_cibil_monotonicity():
    pipeline = joblib.load("models/calibrated_credit_pipeline.joblib")

    base_profile = {
        'no_of_dependents': 2,
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum': 5000000.0,
        'loan_amount': 15000000.0,
        'loan_term': 10,
        'residential_assets_value': 3000000.0,
        'commercial_assets_value': 2000000.0,
        'luxury_assets_value': 4000000.0,
        'bank_asset_value': 2000000.0
    }

    # Evaluate across ascending CIBIL scores
    cibil_scores = [350, 450, 550, 650, 750, 850]
    predicted_pds = []

    for score in cibil_scores:
        profile = base_profile.copy()
        profile['cibil_score'] = score
        df = pd.DataFrame([profile])
        pd_val = float(pipeline.predict_proba(df)[0, 1])
        predicted_pds.append(pd_val)

    # Risk must be strictly non-increasing as CIBIL score increases
    for i in range(len(predicted_pds) - 1):
        assert predicted_pds[i] >= predicted_pds[i + 1] - 1e-5, (
            f"Monotonicity violation at CIBIL {cibil_scores[i]} -> {cibil_scores[i+1]}: "
            f"PD went from {predicted_pds[i]} to {predicted_pds[i+1]}"
        )
