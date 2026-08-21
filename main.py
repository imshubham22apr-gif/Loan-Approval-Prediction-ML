"""
main.py — Complete Loan Approval Prediction Pipeline
=====================================================
Runs the entire ML pipeline:
  1. Download dataset (if needed)
  2. Load & clean data
  3. Fit preprocessing (encoders + scaler)
  4. Transform data
  5. Train multiple models (Logistic Regression, Decision Tree, Random Forest)
  6. Evaluate & compare models
  7. Save the best-performing model + preprocessors to disk
"""

import pandas as pd
import data_cleaner as dc
from model import train_models, evaluate_models, save_pipeline
from download_data import download_dataset
import os


def main():
    csv_path = "loan_approval_dataset.csv"

    # -----------------------------------------------------------
    # Step 1: Download dataset if not present or too small
    # -----------------------------------------------------------
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 5000:
        print("=" * 60)
        print("STEP 1: Downloading full dataset...")
        print("=" * 60)
        download_dataset()
    else:
        print("=" * 60)
        print("STEP 1: Dataset already present — skipping download.")
        print("=" * 60)

    # -----------------------------------------------------------
    # Step 2: Load and clean
    # -----------------------------------------------------------
    print("\nSTEP 2: Loading & cleaning data...")
    df = dc.load_data(csv_path)
    df = dc.clean_data(df)

    print(f"  Dataset shape: {df.shape}")
    print(f"  Target distribution:\n{df['loan_status'].value_counts().to_string()}\n")

    # -----------------------------------------------------------
    # Step 3: Fit preprocessing objects
    # -----------------------------------------------------------
    print("STEP 3: Fitting preprocessing pipeline...")
    num_cols = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    cat_cols = ['education', 'self_employed', 'loan_status']

    preprocessors = dc.fit_preprocessing(df, cat_cols, num_cols)

    # -----------------------------------------------------------
    # Step 4: Transform data
    # -----------------------------------------------------------
    print("STEP 4: Transforming data...")
    df_transformed = dc.transform_data(df, preprocessors, cat_cols, num_cols)

    # -----------------------------------------------------------
    # Step 5: Train / Test split + model training
    # -----------------------------------------------------------
    print("\nSTEP 5: Splitting data & training models...")
    X_train, X_test, y_train, y_test = dc.split_data(df_transformed)
    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    trained_models = train_models(X_train, y_train)

    # -----------------------------------------------------------
    # Step 6: Evaluate & compare
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation Results")
    print("=" * 60)
    results = evaluate_models(trained_models, X_test, y_test)

    # Pretty-print comparison table
    print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 58)
    best_name, best_f1 = None, -1
    for name, metrics in results.items():
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['f1_score']:>10.4f} {auc_str:>10}")
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_name = name

    print(f"\n>> Best model: {best_name} (F1 = {best_f1:.4f})")

    # -----------------------------------------------------------
    # Step 7: Save the best model + preprocessors
    # -----------------------------------------------------------
    print("\nSTEP 7: Saving pipeline artifacts...")
    best_model = results[best_name]['model_object']
    save_pipeline(best_model, preprocessors)

    print("\n[DONE] Pipeline complete! You can now run `python predict.py` for inference.")


if __name__ == "__main__":
    main()
