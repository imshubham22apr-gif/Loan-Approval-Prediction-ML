"""Institutional Credit Risk Model Training Pipeline.

Features:
- Leakage-free Scikit-learn Pipeline with ColumnTransformer
- Engineered Financial Ratios (LTV, DTI, Liquidity, Asset-to-Loan)
- Monotonic Constraints (Higher CIBIL strictly non-increases default risk)
- Stratified K-Fold Probability Calibration (Platt Scaling)
- Cost-Sensitive Business Loss Minimization
- Baseline PSI Drift Tracking Serializer
"""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

from ..features.financial_ratios import FinancialRatioTransformer
from .cost_matrix import find_optimal_threshold, ExpectedLossCalculator
from .evaluate import evaluate_calibrated_model
from ..monitoring.drift_detector import PopulationStabilityIndex


def load_and_preprocess_data(data_path: str = "loan_approval_dataset.csv"):
    """Loads raw dataset, sanitizes string headers/values, and maps target for credit risk."""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype in ['object', 'string']:
            df[col] = df[col].astype(str).str.strip()

    # Credit Risk Formulation:
    # 1 = Default / High Risk (Historical 'Rejected')
    # 0 = Good / Creditworthy (Historical 'Approved')
    df['default_risk_target'] = df['loan_status'].map({'Rejected': 1, 'Approved': 0})

    return df


def build_pipeline_and_train(
    config_path: str = "configs/model_config.yaml"
):
    """Executes the end-to-end institutional training workflow."""
    # Load config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_path = cfg.get("data", {}).get("raw_data_path", "loan_approval_dataset.csv")
    model_dir = cfg.get("paths", {}).get("model_dir", "models")
    reports_dir = cfg.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("INSTITUTIONAL CREDIT RISK TRAINING & PROBABILITY CALIBRATION")
    print("=" * 70)

    print(f"Loading raw loan portfolio data from '{data_path}'...")
    df = load_and_preprocess_data(data_path)
    print(f"Portfolio size: {df.shape[0]} applicants, {df.shape[1]} raw attributes.")

    # Target distribution
    target_counts = df['default_risk_target'].value_counts()
    print(f"Default Risk Distribution: Non-Default (0)={target_counts.get(0, 0)}, Default/Rejected (1)={target_counts.get(1, 0)}")

    y = df['default_risk_target']
    X = df.drop(columns=['loan_id', 'loan_status', 'default_risk_target'], errors='ignore')

    # Train / Test split (Stratified by Default Risk)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.get("data", {}).get("test_size", 0.20),
        random_state=cfg.get("data", {}).get("random_state", 42),
        stratify=y
    )
    print(f"Dataset split: Train={len(X_train)} samples, Holdout Test={len(X_test)} samples.")

    # Features breakdown
    numeric_cols = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value',
        'ltv_ratio', 'dti_ratio', 'liquidity_ratio', 'total_assets', 'asset_to_loan_ratio'
    ]
    cat_cols = ['education', 'self_employed']

    # 1. Feature Engineering step
    ratio_transformer = FinancialRatioTransformer()

    # 2. Preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )

    # Map monotonic constraints to transformed column indices
    # Transformed order: numeric_cols (0..13), then cat columns one-hot encoded (14..N)
    # Monotonic constraints: cibil_score (-1), income_annum (-1), ltv_ratio (+1), dti_ratio (+1), asset_to_loan_ratio (-1)
    mono_map = {
        'cibil_score': -1,
        'income_annum': -1,
        'ltv_ratio': 1,
        'dti_ratio': 1,
        'asset_to_loan_ratio': -1
    }
    mono_constraints = []
    for col in numeric_cols:
        mono_constraints.append(mono_map.get(col, 0))
    # Categorical one-hot features get 0 (unconstrained)
    # We will pass this to LightGBM after getting total features or allow LightGBM on transformed array
    # Since education and self_employed will expand to 4 one-hot columns:
    mono_constraints.extend([0, 0, 0, 0])

    print("\nConfiguring LightGBM Classifier with Monotonic Credit Constraints...")
    base_lgbm = LGBMClassifier(
        n_estimators=cfg.get("model", {}).get("n_estimators", 200),
        learning_rate=cfg.get("model", {}).get("learning_rate", 0.05),
        num_leaves=cfg.get("model", {}).get("num_leaves", 31),
        max_depth=cfg.get("model", {}).get("max_depth", 6),
        subsample=cfg.get("model", {}).get("subsample", 0.8),
        colsample_bytree=cfg.get("model", {}).get("colsample_bytree", 0.8),
        monotone_constraints=mono_constraints,
        random_state=cfg.get("model", {}).get("random_state", 42),
        verbose=-1
    )

    # 3. Probability Calibration via Platt Scaling (CalibratedClassifierCV)
    print("Enclosing model in 5-Fold Stratified Probability Calibration (Platt Scaling)...")
    calibrated_lgbm = CalibratedClassifierCV(
        estimator=base_lgbm,
        method=cfg.get("calibration", {}).get("method", "sigmoid"),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )

    # Full Leakage-Proof Scikit-Learn Pipeline
    full_pipeline = Pipeline(steps=[
        ('financial_ratios', ratio_transformer),
        ('preprocessor', preprocessor),
        ('classifier', calibrated_lgbm)
    ])

    print("Fitting full leakage-proof pipeline on training data...")
    full_pipeline.fit(X_train, y_train)
    print("Training and probability calibration complete.")

    # Evaluate on Holdout Test Set
    print("\nRunning Risk Evaluation on holdout test set...")
    cost_fp = cfg.get("credit_risk", {}).get("cost_false_positive", 1.0)
    cost_fn = cfg.get("credit_risk", {}).get("cost_false_negative", 0.08)
    lgd = cfg.get("credit_risk", {}).get("lgd", 0.45)

    eval_metrics = evaluate_calibrated_model(
        model=full_pipeline,
        X_test=X_test,
        y_test=y_test,
        raw_df_test=df.loc[X_test.index],
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        lgd=lgd
    )

    print("\n=== HOLD-OUT TEST RESULTS ===")
    print(f"  ROC-AUC:               {eval_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:                {eval_metrics['pr_auc']:.4f}")
    print(f"  Brier Score Loss:      {eval_metrics['brier_score']:.4f} (Lower = superior calibration)")
    print(f"  Cost-Optimal Cutoff:   {eval_metrics['optimal_threshold']:.4f}")
    print(f"  Business Cost Savings: {eval_metrics['cost_reduction_pct']:.2f}% vs naive 0.5 threshold")
    print(f"  Total Portfolio EL:    INR {eval_metrics['total_portfolio_expected_loss_inr']:,.2f}")

    print("\nFairness & Bias Audit (Four-Fifths Rule):")
    for f in eval_metrics['fairness_audit']:
        print(f"  [{f['status']}] {f['feature']}: DIR={f['disparate_impact_ratio']} (Compliant: {f['four_fifths_compliant']})")

    # Fit Baseline Drift Detector
    print("\nFitting Population Stability Index (PSI) baseline statistics...")
    psi_detector = PopulationStabilityIndex(num_bins=10)
    # Compute derived features on X_train for baseline distribution
    X_train_ratios = ratio_transformer.transform(X_train)
    psi_cols = [
        'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
        'ltv_ratio', 'dti_ratio', 'liquidity_ratio', 'total_assets'
    ]
    psi_detector.fit(X_train_ratios, numerical_cols=psi_cols)
    baseline_path = os.path.join(model_dir, "baseline_stats.json")
    psi_detector.save(baseline_path)
    print(f"Baseline statistics saved to '{baseline_path}'.")

    # Save Pipeline and Metadata
    pipeline_path = os.path.join(model_dir, "calibrated_credit_pipeline.joblib")
    joblib.dump(full_pipeline, pipeline_path)
    print(f"Full pipeline saved to '{pipeline_path}'.")

    # Save threshold & metadata config
    metadata = {
        "optimal_threshold": eval_metrics['optimal_threshold'],
        "lgd": lgd,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "metrics": eval_metrics
    }
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    metrics_report_path = os.path.join(reports_dir, "model_metrics.json")
    with open(metrics_report_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Evaluation report written to '{metrics_report_path}'.")
    print("=" * 70)
    return full_pipeline, eval_metrics


if __name__ == "__main__":
    build_pipeline_and_train()
