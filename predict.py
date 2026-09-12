"""Production CLI Underwriting Inference Tool.

Evaluates an applicant's loan application using the calibrated credit risk pipeline,
displaying institutional metrics:
- Calibrated Probability of Default (PD)
- Expected Financial Loss (EL = PD * LGD * EAD)
- Financial Ratios (LTV, DTI, Liquidity, Asset-to-Loan)
- SHAP Waterfall Attributions (Top Drivers)
- Adverse Action Remedies (if rejected)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

from src.features.financial_ratios import FinancialRatioTransformer
from src.models.cost_matrix import ExpectedLossCalculator
from src.explainability.shap_explainer import CreditRiskExplainer
from src.explainability.adverse_action import AdverseActionGenerator


def predict_single_applicant(features_dict: dict, models_dir: str = "models"):
    """Runs end-to-end inference and explainability on a single applicant profile."""
    pipeline_path = os.path.join(models_dir, "calibrated_credit_pipeline.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(pipeline_path):
        print(f"Error: Pipeline not found at '{pipeline_path}'. Run 'python -m src.models.train' first.")
        return None

    pipeline = joblib.load(pipeline_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    threshold = metadata.get("optimal_threshold", 0.0991)
    lgd = metadata.get("lgd", 0.45)

    df_raw = pd.DataFrame([features_dict])

    # 1. Financial Ratios
    ratios_df = pipeline.named_steps["financial_ratios"].transform(df_raw)
    ltv = float(ratios_df["ltv_ratio"].iloc[0])
    dti = float(ratios_df["dti_ratio"].iloc[0])
    liquidity = float(ratios_df["liquidity_ratio"].iloc[0])
    asset_to_loan = float(ratios_df["asset_to_loan_ratio"].iloc[0])
    total_assets = float(ratios_df["total_assets"].iloc[0])

    # 2. Calibrated Probability of Default
    prob_default = float(pipeline.predict_proba(df_raw)[0, 1])

    # 3. Decision
    decision = "Approved" if prob_default < threshold else "Rejected"

    # 4. Expected Loss
    loan_amt = float(features_dict.get("loan_amount", 0.0))
    el_calc = ExpectedLossCalculator(lgd=lgd)
    expected_loss = float(el_calc.calculate_expected_loss(
        probability_of_default=np.array([prob_default]),
        exposure_at_default=np.array([loan_amt])
    )[0])

    # 5. SHAP Explanations
    explainer = CreditRiskExplainer(pipeline)
    shap_info = explainer.explain_raw_applicant(features_dict, top_k=3)

    # 6. Adverse Action Notice (if rejected)
    remedies = []
    if decision == "Rejected":
        adverse_gen = AdverseActionGenerator(pipeline, optimal_threshold=threshold)
        remedies = adverse_gen.get_recourse_actions(features_dict, max_recommendations=3)

    return {
        "decision": decision,
        "prob_default": prob_default,
        "threshold": threshold,
        "expected_loss": expected_loss,
        "financial_ratios": {
            "ltv": ltv,
            "dti": dti,
            "liquidity": liquidity,
            "asset_to_loan": asset_to_loan,
            "total_assets": total_assets
        },
        "top_risk_drivers": shap_info["top_risk_drivers"],
        "top_approval_drivers": shap_info["top_approval_drivers"],
        "remedies": remedies
    }


def main():
    if len(sys.argv) > 1:
        try:
            sample_features = {
                'no_of_dependents': int(sys.argv[1]),
                'education': sys.argv[2],
                'self_employed': sys.argv[3],
                'income_annum': float(sys.argv[4]),
                'loan_amount': float(sys.argv[5]),
                'loan_term': int(sys.argv[6]),
                'cibil_score': int(sys.argv[7]),
                'residential_assets_value': float(sys.argv[8]),
                'commercial_assets_value': float(sys.argv[9]),
                'luxury_assets_value': float(sys.argv[10]),
                'bank_asset_value': float(sys.argv[11])
            }
        except IndexError:
            print("Usage: python predict.py <dependents> <education> <self_employed> <income> <loan_amount> <term> <cibil> <residential_val> <commercial_val> <luxury_val> <bank_val>")
            sys.exit(1)
    else:
        print("No CLI arguments passed. Running inference on sample applicant profile...\n")
        sample_features = {
            'no_of_dependents': 2,
            'education': 'Graduate',
            'self_employed': 'No',
            'income_annum': 9600000,
            'loan_amount': 29900000,
            'loan_term': 12,
            'cibil_score': 778,
            'residential_assets_value': 2400000,
            'commercial_assets_value': 17600000,
            'luxury_assets_value': 22700000,
            'bank_asset_value': 8000000
        }

    print("=" * 65)
    print("INSTITUTIONAL CREDIT RISK UNDERWRITING ASSESSMENT")
    print("=" * 65)
    print("APPLICANT PROFILE:")
    for k, v in sample_features.items():
        if isinstance(v, float) and v >= 10000:
            print(f"  {k:<26}: INR {v:,.2f}")
        else:
            print(f"  {k:<26}: {v}")

    result = predict_single_applicant(sample_features)
    if not result:
        return

    print("\n" + "-" * 65)
    print("UNDERWRITING DECISION & RISK METRICS:")
    print("-" * 65)
    dec = result["decision"]
    symbol = "[APPROVED]" if dec == "Approved" else "[REJECTED]"
    print(f"  Final Decision:               {symbol} {dec.upper()}")
    print(f"  Calibrated Default Risk (PD): {result['prob_default']:.2%}")
    print(f"  Cost-Optimal Risk Cutoff:     {result['threshold']:.2%}")
    print(f"  Expected Financial Loss (EL): INR {result['expected_loss']:,.2f}")

    print("\nFINANCIAL RATIOS:")
    r = result["financial_ratios"]
    print(f"  Loan-to-Value (LTV):          {r['ltv']:.2f}")
    print(f"  Debt-to-Income (DTI):         {r['dti']:.2f}")
    print(f"  Liquidity Coverage:           {r['liquidity']:.2f}")
    print(f"  Asset-to-Loan Coverage:       {r['asset_to_loan']:.2f}")
    print(f"  Total Assets Net Valuation:   INR {r['total_assets']:,.2f}")

    print("\nSHAP FACTOR ATTRIBUTION:")
    print("  Key Mitigating Strengths (Approval Factors):")
    for d in result["top_approval_drivers"]:
        print(f"   + {d['feature']:<25} (SHAP impact: {d['shap_value']:+.3f})")
    print("  Key Vulnerabilities (Risk Drivers):")
    for d in result["top_risk_drivers"]:
        print(f"   - {d['feature']:<25} (SHAP impact: {d['shap_value']:+.3f})")

    if result["remedies"]:
        print("\nADVERSE ACTION NOTICE (Actionable Remedies for Approval):")
        for i, rem in enumerate(result["remedies"], 1):
            appr = " -> Achieves Approval" if rem["achieves_approval"] else ""
            print(f"  {i}. [{rem['action_type']}] {rem['description']}")
            print(f"     Risk reduced by {rem['risk_reduction_pct']}% (New PD: {rem['new_pd']:.2%}){appr} | Effort: {rem['effort_level']}")

    print("=" * 65)


if __name__ == "__main__":
    main()
