"""Model Evaluation and Financial Credit Risk Metrics.

Computes:
- ROC-AUC & PR-AUC
- Brier Score Loss (Probabilistic Calibration Metric)
- Cost-optimal threshold evaluation
- Expected Loss comparison
- Confusion Matrix at business-optimal threshold
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from .cost_matrix import find_optimal_threshold, ExpectedLossCalculator
from ..explainability.fairness_audit import audit_model_fairness


def evaluate_calibrated_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_df_test: pd.DataFrame = None,
    cost_fp: float = 1.0,
    cost_fn: float = 0.08,
    lgd: float = 0.45
) -> Dict[str, Any]:
    """Evaluates the full credit risk pipeline.

    Here, positive class (1) = Default / High Risk.
    """
    # Predicted probabilities for class 1 (Default)
    y_prob_default = model.predict_proba(X_test)[:, 1]

    # Probabilistic & Discriminative metrics
    auc = float(roc_auc_score(y_test, y_prob_default))
    pr_auc = float(average_precision_score(y_test, y_prob_default))
    brier = float(brier_score_loss(y_test, y_prob_default))

    # Optimal threshold based on business cost matrix
    best_thresh, min_cost, cost_curve = find_optimal_threshold(
        y_true=y_test,
        y_prob_default=y_prob_default,
        cost_fp=cost_fp,
        cost_fn=cost_fn
    )

    # Cost at naive 0.5 threshold
    y_pred_naive = (y_prob_default >= 0.5).astype(int)
    fp_naive = np.sum((y_test == 0) & (y_pred_naive == 1))
    fn_naive = np.sum((y_test == 1) & (y_pred_naive == 0))
    naive_cost = (fn_naive * cost_fp) + (fp_naive * cost_fn)
    cost_reduction_pct = float(max(0, (naive_cost - min_cost) / (naive_cost + 1e-6) * 100))

    # Business Optimal Predictions
    y_pred_optimal = (y_prob_default >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_optimal).tolist()
    cr = classification_report(y_test, y_pred_optimal, output_dict=True)

    # Expected Loss calculation
    loan_amounts = X_test['loan_amount'].values if 'loan_amount' in X_test.columns else np.full(len(y_test), 1e7)
    el_calc = ExpectedLossCalculator(lgd=lgd)
    expected_losses = el_calc.calculate_expected_loss(
        probability_of_default=y_prob_default,
        exposure_at_default=loan_amounts
    )
    total_portfolio_el = float(np.sum(expected_losses))

    # Fairness audit if test features are present
    fairness_results = []
    if raw_df_test is not None:
        audit_df = raw_df_test.copy()
        # In audit: 1 = Approved, 0 = Rejected
        # An applicant is approved if y_prob_default < best_thresh
        audit_df['predicted_approval'] = (y_prob_default < best_thresh).astype(int)
        fairness_results = audit_model_fairness(audit_df, prediction_col='predicted_approval')

    return {
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "optimal_threshold": round(best_thresh, 4),
        "min_business_cost": round(min_cost, 2),
        "naive_05_cost": round(naive_cost, 2),
        "cost_reduction_pct": round(cost_reduction_pct, 2),
        "confusion_matrix": cm,
        "classification_report": cr,
        "total_portfolio_expected_loss_inr": round(total_portfolio_el, 2),
        "fairness_audit": fairness_results
    }
