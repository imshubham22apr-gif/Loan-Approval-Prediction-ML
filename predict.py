"""
predict.py — Loan Approval Prediction CLI
==========================================
Loads the trained model and preprocessing pipeline from the models/ directory
and predicts loan approval status for a new applicant.

Usage:
    python predict.py              (interactive prompt)
    python predict.py --demo       (run with sample data)
"""

import argparse
import pandas as pd
import numpy as np
from model import load_pipeline
from data_cleaner import clean_data, transform_data


# Column order expected by the model (must match training)
FEATURE_COLS = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

CAT_COLS = ['education', 'self_employed']
NUM_COLS = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]


def get_demo_data():
    """Return a sample applicant for quick testing."""
    return {
        'loan_id': 99999,
        'no_of_dependents': 2,
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum': 5000000,
        'loan_amount': 15000000,
        'loan_term': 12,
        'cibil_score': 750,
        'residential_assets_value': 7000000,
        'commercial_assets_value': 3000000,
        'luxury_assets_value': 5000000,
        'bank_asset_value': 4000000,
        'loan_status': 'Approved'  # placeholder — will be ignored
    }


def get_user_input():
    """Interactively collect applicant details from the user."""
    print("\n--- Enter Applicant Details ---\n")
    data = {}
    data['loan_id'] = 0  # placeholder
    data['no_of_dependents'] = int(input("Number of dependents: "))
    data['education'] = input("Education (Graduate / Not Graduate): ").strip()
    data['self_employed'] = input("Self employed (Yes / No): ").strip()
    data['income_annum'] = float(input("Annual income: "))
    data['loan_amount'] = float(input("Loan amount requested: "))
    data['loan_term'] = int(input("Loan term (months): "))
    data['cibil_score'] = int(input("CIBIL score (300-900): "))
    data['residential_assets_value'] = float(input("Residential assets value: "))
    data['commercial_assets_value'] = float(input("Commercial assets value: "))
    data['luxury_assets_value'] = float(input("Luxury assets value: "))
    data['bank_asset_value'] = float(input("Bank asset value: "))
    data['loan_status'] = 'Approved'  # placeholder
    return data


def predict(applicant_dict, model, preprocessors):
    """Run prediction on a single applicant dictionary."""
    df = pd.DataFrame([applicant_dict])
    df = clean_data(df)

    # We need to include 'loan_status' for transform_data but we won't use it
    cat_cols_with_target = CAT_COLS + ['loan_status']
    df_transformed = transform_data(df, preprocessors, cat_cols_with_target, NUM_COLS)

    # Drop helper columns
    X = df_transformed.drop(columns=['loan_id', 'loan_status'], errors='ignore')

    prediction = model.predict(X)[0]
    try:
        probability = model.predict_proba(X)[0]
    except Exception:
        probability = None

    status = "APPROVED" if prediction == 1 else "REJECTED"
    return status, probability


def main():
    parser = argparse.ArgumentParser(description="Loan Approval Prediction CLI")
    parser.add_argument("--demo", action="store_true", help="Run prediction with sample data")
    args = parser.parse_args()

    print("=" * 50)
    print("  Loan Approval Prediction System")
    print("=" * 50)

    # Load saved model and preprocessors
    try:
        model, preprocessors = load_pipeline()
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print("ERROR: No trained model found. Run `python main.py` first to train the model.")
        return

    if args.demo:
        applicant = get_demo_data()
        print("Using demo applicant data:")
        for k, v in applicant.items():
            if k not in ('loan_id', 'loan_status'):
                print(f"  {k}: {v}")
    else:
        applicant = get_user_input()

    status, probability = predict(applicant, model, preprocessors)

    print("\n" + "=" * 50)
    print(f"  Prediction: {status}")
    if probability is not None:
        print(f"  Confidence — Rejected: {probability[0]:.2%}  |  Approved: {probability[1]:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
