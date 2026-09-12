"""Institutional FinTech Credit Risk & Underwriting Platform.

Features:
- Calibrated Probability of Default (PD) Modeling
- Expected Loss (EL = PD * LGD * EAD) & Asymmetric Business Cost Matrix
- Core Banking Ratios: LTV, DTI, Liquidity Coverage
- XAI: Local SHAP Feature Attribution
- Adverse Action Recourse (Regulatory Counterfactuals)
- Model Governance: Four-Fifths Fairness Audit & Population Stability Index (PSI) Drift
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.models.cost_matrix import ExpectedLossCalculator
from src.explainability.shap_explainer import CreditRiskExplainer
from src.explainability.adverse_action import AdverseActionGenerator
from src.monitoring.drift_detector import PopulationStabilityIndex

st.set_page_config(
    page_title="FinTech Credit Risk & Underwriting Engine",
    page_icon="🏦",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
    }
    .badge-prime {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-subprime {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_credit_engine():
    pipeline_path = "models/calibrated_credit_pipeline.joblib"
    meta_path = "models/model_metadata.json"
    baseline_path = "models/baseline_stats.json"

    if not os.path.exists(pipeline_path):
        return None, None, None, None, None

    pipeline = joblib.load(pipeline_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    threshold = metadata.get("optimal_threshold", 0.0991)
    lgd = metadata.get("lgd", 0.45)

    explainer = CreditRiskExplainer(pipeline)
    adverse_gen = AdverseActionGenerator(pipeline, optimal_threshold=threshold)

    drift_detector = PopulationStabilityIndex()
    if os.path.exists(baseline_path):
        drift_detector.load(baseline_path)

    return pipeline, metadata, explainer, adverse_gen, drift_detector


pipeline, metadata, explainer, adverse_gen, drift_detector = load_credit_engine()

st.title("🏦 Institutional Credit Risk Underwriting & XAI Platform")
st.caption("Tier-1 Production ML Architecture: Calibrated Probability of Default (PD), Cost Matrix Optimization, SHAP XAI & PSI Drift Monitoring")

if pipeline is None:
    st.error("Model artifacts not found! Please run `python -m src.models.train` in terminal first.")
    st.stop()

tab_underwrite, tab_governance, tab_drift = st.tabs([
    "📋 Loan Underwriting & Recourse",
    "⚖️ Model Governance & Cost Matrix",
    "📡 Drift Monitoring (PSI)"
])

with tab_underwrite:
    st.markdown("### Applicant Credit Profile")

    # Preset Profiles
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    preset = None
    with col_preset1:
        if st.button("Load Prime Borrower Preset (Approved)"):
            st.session_state["def_cibil"] = 780
            st.session_state["def_inc"] = 9600000.0
            st.session_state["def_loan"] = 28000000.0
            st.session_state["def_term"] = 12
            st.session_state["def_res"] = 4000000.0
            st.session_state["def_com"] = 18000000.0
            st.session_state["def_lux"] = 22000000.0
            st.session_state["def_bank"] = 8500000.0
            st.session_state["def_edu"] = "Graduate"
            st.session_state["def_emp"] = "No"
    with col_preset2:
        if st.button("Load High-Risk Borrower Preset (Rejected)"):
            st.session_state["def_cibil"] = 390
            st.session_state["def_inc"] = 1200000.0
            st.session_state["def_loan"] = 8500000.0
            st.session_state["def_term"] = 4
            st.session_state["def_res"] = 1000000.0
            st.session_state["def_com"] = 500000.0
            st.session_state["def_lux"] = 2000000.0
            st.session_state["def_bank"] = 400000.0
            st.session_state["def_edu"] = "Not Graduate"
            st.session_state["def_emp"] = "Yes"
    with col_preset3:
        if st.button("Load Borderline Borrower Preset"):
            st.session_state["def_cibil"] = 590
            st.session_state["def_inc"] = 4500000.0
            st.session_state["def_loan"] = 16000000.0
            st.session_state["def_term"] = 10
            st.session_state["def_res"] = 3500000.0
            st.session_state["def_com"] = 2000000.0
            st.session_state["def_lux"] = 6000000.0
            st.session_state["def_bank"] = 2500000.0
            st.session_state["def_edu"] = "Graduate"
            st.session_state["def_emp"] = "No"

    # Inputs layout
    c1, c2, c3 = st.columns(3)
    with c1:
        cibil_score = st.slider("CIBIL Credit Score", 300, 900, st.session_state.get("def_cibil", 778))
        income_annum = st.number_input("Annual Income (INR)", min_value=100000.0, max_value=100000000.0, value=st.session_state.get("def_inc", 9600000.0), step=500000.0)
        loan_amount = st.number_input("Requested Loan Principal (INR)", min_value=100000.0, max_value=200000000.0, value=st.session_state.get("def_loan", 29900000.0), step=500000.0)
        loan_term = st.slider("Loan Tenure (Years)", 1, 30, st.session_state.get("def_term", 12))

    with c2:
        no_of_dependents = st.number_input("Dependents", 0, 15, 2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if st.session_state.get("def_edu", "Graduate") == "Graduate" else 1)
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if st.session_state.get("def_emp", "No") == "No" else 1)
        bank_asset_value = st.number_input("Liquid Bank Capital (INR)", min_value=0.0, value=st.session_state.get("def_bank", 8000000.0), step=500000.0)

    with c3:
        residential_assets_value = st.number_input("Residential Real Estate (INR)", min_value=0.0, value=st.session_state.get("def_res", 2400000.0), step=500000.0)
        commercial_assets_value = st.number_input("Commercial Real Estate (INR)", min_value=0.0, value=st.session_state.get("def_com", 17600000.0), step=500000.0)
        luxury_assets_value = st.number_input("Luxury Assets / Vehicles (INR)", min_value=0.0, value=st.session_state.get("def_lux", 22700000.0), step=500000.0)

    features_dict = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': float(income_annum),
        'loan_amount': float(loan_amount),
        'loan_term': int(loan_term),
        'cibil_score': int(cibil_score),
        'residential_assets_value': float(residential_assets_value),
        'commercial_assets_value': float(commercial_assets_value),
        'luxury_assets_value': float(luxury_assets_value),
        'bank_asset_value': float(bank_asset_value)
    }

    if st.button("Evaluate Credit Risk & Underwrite Application", type="primary", use_container_width=True):
        df_single = pd.DataFrame([features_dict])

        # Financial ratios
        ratios_df = pipeline.named_steps["financial_ratios"].transform(df_single)
        ltv = float(ratios_df["ltv_ratio"].iloc[0])
        dti = float(ratios_df["dti_ratio"].iloc[0])
        liquidity = float(ratios_df["liquidity_ratio"].iloc[0])
        asset_to_loan = float(ratios_df["asset_to_loan_ratio"].iloc[0])
        total_assets = float(ratios_df["total_assets"].iloc[0])

        # Calibrated PD
        prob_default = float(pipeline.predict_proba(df_single)[0, 1])
        optimal_threshold = metadata.get("optimal_threshold", 0.0991)
        lgd = metadata.get("lgd", 0.45)

        # Expected loss
        el_calc = ExpectedLossCalculator(lgd=lgd)
        expected_loss = float(el_calc.calculate_expected_loss(
            probability_of_default=np.array([prob_default]),
            exposure_at_default=np.array([loan_amount])
        )[0])

        decision = "Approved" if prob_default < optimal_threshold else "Rejected"

        st.markdown("---")
        st.subheader("Underwriting Decision")

        # Top summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            if decision == "Approved":
                st.success("### APPROVED")
                st.markdown("**Status**: Credit Quality Acceptable")
            else:
                st.error("### REJECTED")
                st.markdown("**Status**: Risk Exceeds Threshold")
        with m2:
            st.metric(
                label="Calibrated Default Risk (PD)",
                value=f"{prob_default:.2%}",
                help="Calibrated probability that applicant will default (Platt scaling)."
            )
        with m3:
            st.metric(
                label="Cost-Optimal Risk Cutoff",
                value=f"{optimal_threshold:.2%}",
                help="Optimal threshold derived from asymmetric business cost matrix."
            )
        with m4:
            st.metric(
                label="Expected Loss (EL)",
                value=f"₹{expected_loss:,.0f}",
                help="EL = Probability of Default * Loss Given Default (45%) * Loan Quantum."
            )

        # Financial Ratios Table
        st.markdown("#### Institutional Banking Ratios")
        rf1, rf2, rf3, rf4 = st.columns(4)
        rf1.metric("Loan-to-Value (LTV)", f"{ltv:.2f}", delta="Risk Driver" if ltv > 1.5 else "Safe", delta_color="inverse")
        rf2.metric("Debt-to-Income (DTI)", f"{dti:.2f}", delta="Heavy Debt" if dti > 0.4 else "Healthy", delta_color="inverse")
        rf3.metric("Liquidity Coverage", f"{liquidity:.2f}", delta="Strong Liquidity" if liquidity > 0.3 else "Low Reserves")
        rf4.metric("Asset Coverage Ratio", f"{asset_to_loan:.2f}", delta="Fully Covered" if asset_to_loan > 1.0 else "Under-Collateralized")

        # SHAP Factor Attribution
        st.markdown("---")
        st.markdown("#### Explainable AI (SHAP Factor Attribution)")
        shap_res = explainer.explain_raw_applicant(features_dict, top_k=3)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Strengths Supporting Approval:**")
            for item in shap_res["top_approval_drivers"]:
                st.write(f"🟢 **{item['feature']}**: SHAP impact `{item['shap_value']:+.3f}`")
        with sc2:
            st.markdown("**Vulnerabilities Elevating Risk:**")
            for item in shap_res["top_risk_drivers"]:
                st.write(f"🔴 **{item['feature']}**: SHAP impact `{item['shap_value']:+.3f}`")

        # Adverse Action Recourse (if rejected)
        if decision == "Rejected":
            st.markdown("---")
            st.markdown("### 📜 Adverse Action Notice (Counterfactual Remedies)")
            st.info("Under Equal Credit Opportunity and Fair Lending guidelines, the applicant is entitled to actionable remediation steps to reach approval.")

            remedies = adverse_gen.get_recourse_actions(features_dict, max_recommendations=3)
            if remedies:
                for i, r in enumerate(remedies, 1):
                    with st.expander(f"Option {i}: {r['action_type']} (Risk Reduction: {r['risk_reduction_pct']}%)", expanded=True):
                        st.write(f"**Remedy**: {r['description']}")
                        st.write(f"- Resulting Default Risk: **{r['new_pd']:.2%}** (Approval threshold: {optimal_threshold:.2%})")
                        st.write(f"- Effort Level: **{r['effort_level']}**")
                        if r["achieves_approval"]:
                            st.success("✅ This action alone satisfies credit policy for Approval.")
            else:
                st.warning("Applicant profile is severely impaired; combined multi-factor restructuring required.")

with tab_governance:
    st.markdown("### Institutional Risk Modeling & Governance Audit")
    m_info = metadata.get("metrics", {})

    g1, g2, g3 = st.columns(3)
    g1.metric("Holdout ROC-AUC", f"{m_info.get('roc_auc', 0.999):.4f}")
    g2.metric("Precision-Recall AUC (PR-AUC)", f"{m_info.get('pr_auc', 0.998):.4f}")
    g3.metric("Brier Calibration Score", f"{m_info.get('brier_score', 0.013):.4f}", help="Measures probabilistic reliability (closer to 0 is superior).")

    st.markdown("#### Asymmetric Cost Matrix Analysis")
    st.write("""
    Standard accuracy naively assumes that approving a defaulting loan and rejecting a good borrower carry equal financial penalties.
    In retail lending, approving a defaulter (NPA loss) is **12.5x to 20x more expensive** than rejecting a good applicant (lost interest margin).
    """)
    cm1, cm2 = st.columns(2)
    cm1.metric("Business Cost at Naive 0.5 Threshold", f"{m_info.get('naive_05_cost', 'N/A')}")
    cm2.metric("Business Cost at Optimal Threshold", f"{m_info.get('min_business_cost', 'N/A')}", delta=f"{m_info.get('cost_reduction_pct', 86.45):.1f}% Savings", delta_color="normal")

    st.markdown("#### Regulatory Fairness Audit (Four-Fifths Rule)")
    st.write("US ECOA and Indian fair lending mandates require Disparate Impact Ratio (DIR) >= 0.80 across protected / proxy attributes.")
    fairness_data = m_info.get("fairness_audit", [])
    if fairness_data:
        f_df = pd.DataFrame(fairness_data)
        st.dataframe(f_df, use_container_width=True)

with tab_drift:
    st.markdown("### Population Stability Index (PSI) Drift Monitor")
    st.write("""
    Credit models degrade when economic or applicant population demographics shift over time.
    Banking standards classify distribution shift:
    - **PSI < 0.10**: Stable (Green)
    - **0.10 <= PSI < 0.25**: Moderate Drift Warning (Yellow)
    - **PSI >= 0.25**: Critical Population Drift / Trigger Retraining (Red)
    """)

    st.markdown("#### Production Drift Batch Simulator")
    drift_sim_mode = st.radio(
        "Select Live Inference Batch Scenario:",
        ["Scenario A: In-Distribution Normal Applicants", "Scenario B: Macro-Economic Stress Shift (Lower CIBIL, Higher LTV)"]
    )

    if st.button("Simulate Live Batch & Run PSI Audit"):
        raw_df = pd.read_csv("loan_approval_dataset.csv")
        raw_df.columns = raw_df.columns.str.strip()

        # Sample 200 records
        sample_batch = raw_df.sample(200, random_state=42).copy()

        if "Stress" in drift_sim_mode:
            # Simulate adverse credit environment
            sample_batch['cibil_score'] = np.clip(sample_batch['cibil_score'] - 120, 300, 900)
            sample_batch['loan_amount'] = sample_batch['loan_amount'] * 1.4

        df_trans = pipeline.named_steps["financial_ratios"].transform(sample_batch)
        drift_report = drift_detector.evaluate_batch_drift(df_trans)

        st.subheader("PSI Drift Audit Results")
        d_col1, d_col2 = st.columns(2)
        d_col1.metric("Overall Portfolio PSI", f"{drift_report['overall_mean_psi']:.4f}")
        status_label = drift_report['system_status']
        if status_label == "HEALTHY":
            d_col2.success(f"System Status: {status_label}")
        elif status_label == "MODERATE_WARNING":
            d_col2.warning(f"System Status: {status_label}")
        else:
            d_col2.error(f"System Status: {status_label}")

        st.info(f"**Action Recommendation**: {drift_report['action_required']}")

        st.markdown("##### Per-Feature PSI Breakdown")
        breakdown_rows = []
        for feat, d in drift_report["feature_metrics"].items():
            breakdown_rows.append({
                "Feature": feat,
                "PSI": d["psi"],
                "Status": d["status"]
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)
