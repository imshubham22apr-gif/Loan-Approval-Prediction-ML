"""FastAPI Production Microservice for Credit Risk Underwriting & Monitoring."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ApplicantInput,
    CreditRiskPredictionResponse,
    FinancialRatiosResponse,
    FeatureDriver,
    AdverseActionRecourse,
    DriftCheckBatchRequest,
    DriftCheckResponse
)
from ..models.cost_matrix import ExpectedLossCalculator
from ..explainability.shap_explainer import CreditRiskExplainer
from ..explainability.adverse_action import AdverseActionGenerator
from ..monitoring.drift_detector import PopulationStabilityIndex


# Global application state
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads model artifacts, explainers, and baseline drift stats on startup."""
    model_path = "models/calibrated_credit_pipeline.joblib"
    meta_path = "models/model_metadata.json"
    baseline_path = "models/baseline_stats.json"

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model pipeline artifact not found at '{model_path}'. Run training first.")

    pipeline = joblib.load(model_path)
    state["pipeline"] = pipeline

    # Load metadata
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    state["metadata"] = metadata
    state["threshold"] = metadata.get("optimal_threshold", 0.0991)
    state["lgd"] = metadata.get("lgd", 0.45)

    # Initialize Explainer and Adverse Action generator
    state["explainer"] = CreditRiskExplainer(pipeline)
    state["adverse_gen"] = AdverseActionGenerator(pipeline, optimal_threshold=state["threshold"])

    # Initialize Drift detector
    drift_detector = PopulationStabilityIndex()
    if os.path.exists(baseline_path):
        drift_detector.load(baseline_path)
    state["drift_detector"] = drift_detector

    print("=" * 60)
    print("FastAPI Credit Risk Microservice Loaded Successfully")
    print(f"Optimal Decision Cutoff: {state['threshold']}")
    print(f"Loss Given Default (LGD): {state['lgd']}")
    print("=" * 60)

    yield
    state.clear()


app = FastAPI(
    title="Institutional Credit Risk Underwriting API",
    description=(
        "Production-grade microservice incorporating Calibrated Probability of Default (PD), "
        "Expected Loss (EL = PD * LGD * EAD), Monotonic Constraints, SHAP XAI attributions, "
        "Regulatory Adverse Action counterfactual recourse, and PSI Drift Monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Institutional Credit Risk Underwriting API",
        "version": "2.0.0",
        "documentation": "/docs",
        "status": "active"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    if "pipeline" not in state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded."
        )
    return {
        "status": "healthy",
        "optimal_threshold": state.get("threshold"),
        "lgd": state.get("lgd")
    }


@app.get("/metrics", tags=["Governance & Auditing"])
def get_model_metrics():
    """Returns holdout validation metrics, cost-curve savings, and fairness audit."""
    if "metadata" not in state or "metrics" not in state["metadata"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics not found in metadata."
        )
    return state["metadata"]["metrics"]


@app.post("/predict", response_model=CreditRiskPredictionResponse, tags=["Inference"])
def predict_credit_risk(applicant: ApplicantInput):
    """Evaluates applicant creditworthiness, computes calibrated PD, Expected Loss,

    SHAP risk drivers, and adverse action remedies if denied.
    """
    pipeline = state["pipeline"]
    threshold = state["threshold"]
    lgd = state["lgd"]

    applicant_dict = applicant.model_dump()
    raw_df = pd.DataFrame([applicant_dict])

    # 1. Transform financial ratios for inspection
    ratios_df = pipeline.named_steps["financial_ratios"].transform(raw_df)
    ratios_resp = FinancialRatiosResponse(
        ltv_ratio=round(float(ratios_df["ltv_ratio"].iloc[0]), 4),
        dti_ratio=round(float(ratios_df["dti_ratio"].iloc[0]), 4),
        liquidity_ratio=round(float(ratios_df["liquidity_ratio"].iloc[0]), 4),
        asset_to_loan_ratio=round(float(ratios_df["asset_to_loan_ratio"].iloc[0]), 4),
        total_assets=round(float(ratios_df["total_assets"].iloc[0]), 2)
    )

    # 2. Calibrated Probability of Default (PD)
    prob_default = float(pipeline.predict_proba(raw_df)[0, 1])

    # 3. Decision based on Cost-Optimal Threshold
    # If PD < threshold -> Low Risk -> Approved
    # If PD >= threshold -> High Risk -> Rejected
    decision = "Approved" if prob_default < threshold else "Rejected"

    # Risk Tier classification
    if prob_default < (threshold * 0.5):
        risk_tier = "Prime / Low Risk"
    elif prob_default < threshold:
        risk_tier = "Near Prime / Moderate Risk"
    elif prob_default < (threshold * 2.0):
        risk_tier = "Subprime / Elevated Risk"
    else:
        risk_tier = "High Risk / Default Prone"

    # 4. Expected Loss (EL = PD * LGD * EAD)
    el_calc = ExpectedLossCalculator(lgd=lgd)
    expected_loss = float(el_calc.calculate_expected_loss(
        probability_of_default=np.array([prob_default]),
        exposure_at_default=np.array([applicant.loan_amount])
    )[0])

    # 5. SHAP Feature Attributions
    shap_results = state["explainer"].explain_raw_applicant(applicant_dict, top_k=3)
    top_risk = [FeatureDriver(**d) for d in shap_results["top_risk_drivers"]]
    top_approval = [FeatureDriver(**d) for d in shap_results["top_approval_drivers"]]

    # 6. Adverse Action Recourse (if rejected)
    remedies = []
    if decision == "Rejected":
        raw_remedies = state["adverse_gen"].get_recourse_actions(applicant_dict, max_recommendations=3)
        remedies = [AdverseActionRecourse(**r) for r in raw_remedies]

    return CreditRiskPredictionResponse(
        decision=decision,
        probability_of_default=round(prob_default, 4),
        risk_tier=risk_tier,
        expected_loss_inr=round(expected_loss, 2),
        decision_threshold=round(threshold, 4),
        financial_ratios=ratios_resp,
        top_risk_drivers=top_risk,
        top_approval_drivers=top_approval,
        adverse_action_remedies=remedies
    )


@app.post("/drift-check", response_model=DriftCheckResponse, tags=["Monitoring"])
def check_population_drift(batch: DriftCheckBatchRequest):
    """Calculates Population Stability Index (PSI) on a batch of live applicants."""
    if len(batch.applicants) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch of applicants must contain at least 1 record."
        )

    pipeline = state["pipeline"]
    batch_dicts = [a.model_dump() for a in batch.applicants]
    raw_df = pd.DataFrame(batch_dicts)
    df_transformed = pipeline.named_steps["financial_ratios"].transform(raw_df)

    drift_report = state["drift_detector"].evaluate_batch_drift(df_transformed)
    return DriftCheckResponse(**drift_report)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
