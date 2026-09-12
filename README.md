# Institutional Credit Risk & Underwriting ML System
### Production-Grade FinTech Machine Learning Architecture (Tier-1 / IIT & MIT Benchmark)

[![CI Pipeline](https://github.com/imshubham22apr-gif/Loan-Approval-Prediction-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/imshubham22apr-gif/Loan-Approval-Prediction-ML/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏛️ Executive Summary & Architectural Evolution

In retail lending and commercial banking, achieving a "98% accuracy score" on a symmetric binary classification setup is often a red flag indicating data leakage, uncalibrated probability thresholds, or synthetic dataset quirks. 

In actual credit risk underwriting, **approving a defaulting loan (Non-Performing Asset / NPA)** can cost an institution $10\times$ to $25\times$ more than **rejecting a creditworthy applicant (lost interest margin)**. Furthermore, US ECOA/FCRA and Indian RBI regulatory statutes legally mandate **Explainable AI (Adverse Action Notices)**, **Fair Lending Bias Audits (Four-Fifths Rule)**, and **Population Stability Index (PSI) Drift Monitoring**.

This repository transforms loan underwriting into an **Institutional-Grade FinTech Decision Engine**.

### 📊 Comparative Paradigm Matrix

| Architectural Domain | Typical Tutorial-Level ML Repo | Tier-1 Institutional (IIT / MIT) Production Standard |
| :--- | :--- | :--- |
| **Problem Formulation** | Symmetric Binary Classification ($0/1$) | **Probability of Default ($PD$)** & Expected Loss ($EL = PD \times LGD \times EAD$) |
| **Optimization Objective** | Symmetric Accuracy / F1-Score | **Asymmetric Business Cost Curve Minimization** ($C_{FP} \gg C_{FN}$) |
| **Feature Engineering** | Raw column MinMax scaling & Label Encoding | **Core Banking Ratios** (LTV, DTI, Liquidity Coverage) + **Monotonic Constraints** |
| **Validation & Leakage** | Single 80/20 train-test split | **Stratified K-Fold CV** inside strict Leakage-Proof Scikit-Learn Pipeline |
| **Model Calibration** | Uncalibrated Tree Probabilities | **Platt Scaling (Sigmoid Calibration)** evaluated via **Brier Score Loss** |
| **Regulatory & XAI** | Global feature importances / Correlation plot | **SHAP TreeExplainer Waterfall** + **Adverse Action Counterfactual Recourse** |
| **Fairness & Bias** | None | **Disparate Impact Ratio (DIR)** across demographic / proxy variables (80% rule) |
| **Serving & Architecture** | Standalone script or bare Streamlit | **Containerized FastAPI Microservice** + Pydantic Bounds Validation + Docker |
| **Drift Monitoring** | None | **Population Stability Index (PSI)** tracking distribution shift & triggering alerts |
| **Quality & CI/CD** | Untested ad-hoc scripts | **Automated Pytest Suite (12 unit/integration tests)** + **GitHub Actions CI** |

---

## 📐 Financial Risk Theory & Mathematics

### 1. Basel Expected Loss (EL) Formulation
The continuous credit loss incurred by a lending institution is modeled as:
$$\text{Expected Loss (EL)} = PD \times LGD \times EAD$$
- **$PD$ (Probability of Default)**: Calibrated model prediction $\in [0, 1]$.
- **$LGD$ (Loss Given Default)**: Basel standard retail proxy fixed at $45\%$ ($0.45$).
- **$EAD$ (Exposure at Default)**: Requested principal `loan_amount` in INR.

### 2. Asymmetric Business Cost Matrix & Threshold Optimization
Unlike standard $0.5$ cutoffs, the decision threshold $\tau^*$ is derived by minimizing the total financial damage function:
$$\text{Total Business Cost}(\tau) = C_{\text{bad\_loan}} \cdot FN(\tau) + C_{\text{lost\_customer}} \cdot FP(\tau)$$
where:
- Approving a bad borrower ($FN$ where Positive=Default) costs $C_{\text{bad\_loan}} = 1.0 \times \text{LGD}$.
- Rejecting a good borrower ($FP$) costs the lost net interest margin $C_{\text{lost\_customer}} \approx 0.08$.
- **Cost Reduction Achieved**: **86.45% financial savings** over naive $0.5$ threshold!

```
Optimal Decision Cutoff (τ*): 0.0991 (9.91% default probability)
If PD < 9.91%  -> APPROVED (Prime / Near Prime)
If PD >= 9.91% -> REJECTED (Elevated Risk / Subprime)
```

---

## ⚙️ Advanced Banking Feature Engineering & Monotonic Constraints

Raw balance-sheet features are mapped into core banking underwriting ratios:

1. **Loan-to-Value (LTV) Ratio**:
   $$\text{LTV} = \frac{\text{loan\_amount}}{\text{residential\_assets} + \text{commercial\_assets} + 10^{-5}}$$
2. **Debt-to-Income (DTI) / Burden Ratio**:
   $$\text{DTI} = \frac{\text{loan\_amount} / \text{loan\_term}}{\text{income\_annum} + 10^{-5}}$$
3. **Asset Liquidity Coverage Ratio**:
   $$\text{Liquidity} = \frac{\text{bank\_asset\_value}}{\text{luxury\_assets} + \text{residential\_assets} + \text{commercial\_assets} + 10^{-5}}$$
4. **Asset-to-Loan Coverage**:
   $$\text{Asset Coverage} = \frac{\text{Total Assets}}{\text{loan\_amount} + 10^{-5}}$$

### Monotonic Constraints in Tree Ensembles (LightGBM)
To satisfy financial common sense and audit compliance, models must obey non-decreasing/non-increasing domain bounds:
```python
monotone_constraints = {
    'cibil_score': -1,         # Increasing CIBIL strictly non-increases default risk
    'income_annum': -1,        # Increasing income strictly non-increases default risk
    'ltv_ratio': +1,           # Higher leverage strictly increases default risk
    'dti_ratio': +1,           # Higher debt burden strictly increases default risk
    'asset_to_loan_ratio': -1  # Higher asset coverage strictly decreases risk
}
```

---

## 📈 Model Performance, Calibration & Fairness Audit

Trained using 5-Fold Stratified Cross-Validation with Platt Scaling probability calibration:

- **ROC-AUC**: `0.9990`
- **PR-AUC**: `0.9983`
- **Brier Score Loss**: `0.0131` *(Near-zero Brier indicates well-calibrated probabilistic risk)*
- **Cost Savings vs Naive Cutoff**: `86.45%`

### Regulatory Fair Lending Audit (Four-Fifths / 80% Rule)
$$\text{Disparate Impact Ratio (DIR)} = \frac{P(\text{Approved} \mid \text{Unprivileged})}{P(\text{Approved} \mid \text{Privileged})}$$

| Attribute Audited | Privileged Group | Unprivileged Group | Disparate Impact Ratio | Regulatory Compliance |
| :--- | :--- | :--- | :---: | :---: |
| **Education** | Graduate | Not Graduate | **0.9216** | **PASS** (Compliant $\ge 0.80$) |
| **Employment** | Salaried (No) | Self-Employed (Yes) | **1.0019** | **PASS** (Compliant $\ge 0.80$) |

---

## 🔍 Explainable AI (XAI) & Adverse Action Notices

Under US Equal Credit Opportunity Act (ECOA) and RBI guidelines, when credit is denied, applicants are legally entitled to:
1. **SHAP Factor Attribution**: Top positive strengths supporting approval and top negative drivers increasing risk.
2. **Actionable Adverse Action Counterfactual Recourse**: Exact, actionable remedies calculating how minimal changes in CIBIL, loan quantum, or tenure flip rejection into approval:
   ```text
   Option 1: [LOAN_AMOUNT_REDUCTION] Reduce loan request by 25% (reduce by INR 2,125,000)
             -> Lowers default risk from 77.34% to 8.42% (Achieves Approval).
   Option 2: [CIBIL_SCORE_IMPROVEMENT] Improve CIBIL score from 390 to 750 (+360 pts)
             -> Reduces default risk by 31.76%.
   ```

---

## 📡 Population Stability Index (PSI) Drift Monitoring

Production models degrade when macroeconomic or demographic shifts occur. This engine includes real-time PSI tracking:
$$PSI = \sum_{i=1}^{B} \left( \% \text{Actual}_i - \% \text{Expected}_i \right) \times \ln\left( \frac{\% \text{Actual}_i}{\% \text{Expected}_i} \right)$$
- **$PSI < 0.10$**: Stable Distribution (Green)
- **$0.10 \le PSI < 0.25$**: Moderate Warning (Yellow)
- **$PSI \ge 0.25$**: Critical Distribution Drift $\rightarrow$ Mandatory Retraining Alert (Red)

---

## 🏗️ Repository Architecture

```text
Loan-Approval-Prediction-ML/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated Pytest & CI workflow
├── configs/
│   └── model_config.yaml              # Risk hyperparameters, cost matrix & feature config
├── src/
│   ├── api/
│   │   ├── main.py                    # Production FastAPI REST microservice
│   │   └── schemas.py                 # Pydantic v2 domain-validated input/output models
│   ├── features/
│   │   └── financial_ratios.py        # Scikit-Learn transformer for LTV, DTI, Liquidity
│   ├── models/
│   │   ├── train.py                   # Calibrated LightGBM pipeline with monotonic constraints
│   │   ├── evaluate.py                # Brier score, ROC-AUC, Cost Curve & Fairness metrics
│   │   └── cost_matrix.py             # Expected Loss (EL = PD * LGD * EAD) & cost optimizer
│   ├── explainability/
│   │   ├── shap_explainer.py          # SHAP TreeExplainer local waterfall attribution
│   │   ├── adverse_action.py          # Counterfactual adverse action recourse generator
│   │   └── fairness_audit.py          # Disparate Impact Ratio auditor
│   └── monitoring/
│       └── drift_detector.py          # Population Stability Index (PSI) drift monitoring
├── tests/
│   ├── test_financial_features.py     # Unit tests for LTV, DTI, Liquidity formulas
│   ├── test_pipeline_leakage.py       # Data leakage checks across transformers
│   ├── test_monotonicity.py           # Verification of non-decreasing credit score constraints
│   ├── test_api.py                    # FastAPI endpoint tests & Pydantic validation
│   └── test_drift.py                  # PSI calculation tests
├── app.py                             # Interactive FinTech Streamlit Underwriting & XAI App
├── predict.py                         # Production CLI Underwriting Tool
├── Dockerfile                         # Containerization for FastAPI & Streamlit
├── docker-compose.yml                 # Multi-service local composition
├── pyproject.toml                     # Modern package & build specification
└── requirements.txt                   # Frozen production dependencies
```

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Setup
```bash
git clone https://github.com/imshubham22apr-gif/Loan-Approval-Prediction-ML.git
cd Loan-Approval-Prediction-ML
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Training Pipeline
```bash
python -m src.models.train
```

### 3. Run Automated Pytest Suite
```bash
pytest tests/ -v
```

### 4. Run CLI Underwriting Assessment
```bash
# Evaluate sample profile
python predict.py

# Evaluate custom applicant:
# python predict.py <dependents> <education> <self_employed> <income> <loan_amount> <term> <cibil> <residential_val> <commercial_val> <luxury_val> <bank_val>
python predict.py 2 Graduate No 9600000 29900000 12 778 2400000 17600000 22700000 8000000
```

### 5. Launch FastAPI REST Microservice
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Endpoint: `GET http://localhost:8000/health`
- Governance Metrics: `GET http://localhost:8000/metrics`
- Underwriting Decision: `POST http://localhost:8000/predict`
- Drift Monitoring: `POST http://localhost:8000/drift-check`

### 6. Launch Interactive Streamlit Platform
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

### 7. Run with Docker Compose
```bash
docker-compose up --build
```
- API served at `http://localhost:8000`
- Web Dashboard served at `http://localhost:8501`

---

## 📜 License & Citation

Distributed under the MIT License. Developed to embody Tier-1 (IIT / MIT) quantitative finance and production ML engineering rigor.
