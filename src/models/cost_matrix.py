"""Financial Risk & Cost-Sensitive Optimization Module.

Implements Credit Risk Basel formulation:
- Expected Loss: EL = PD * LGD * EAD
- Asymmetric Business Cost Matrix Optimization
- Cost Curve Evaluation across thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class ExpectedLossCalculator:
    """Calculates Credit Risk Expected Loss (EL = PD * LGD * EAD).

    Attributes:
        lgd (float): Loss Given Default (fraction of exposure lost upon default, typically 0.45).
    """

    def __init__(self, lgd: float = 0.45):
        self.lgd = lgd

    def calculate_expected_loss(
        self,
        probability_of_default: np.ndarray,
        exposure_at_default: np.ndarray
    ) -> np.ndarray:
        """Computes continuous expected loss in currency units (INR).

        Args:
            probability_of_default: Calibrated PD array (0.0 to 1.0).
            exposure_at_default: Requested loan amount (EAD).

        Returns:
            np.ndarray: Expected financial loss for each applicant.
        """
        pd_clean = np.clip(np.asarray(probability_of_default), 0.0, 1.0)
        ead_clean = np.maximum(np.asarray(exposure_at_default), 0.0)
        return pd_clean * self.lgd * ead_clean


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob_default: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 0.08,
    num_thresholds: int = 100
) -> Tuple[float, float, pd.DataFrame]:
    """Finds the optimal decision threshold that minimizes total asymmetric business cost.

    Here, positive class (1) = Default / High Risk.
    - False Positive (FP): Predicted Default, Actual Good (Opportunity loss of interest margin = cost_fn)
      Wait! In standard definitions:
      If True = Default (1):
      - FP = Predicted Default (1), Actual Good (0) -> Opportunity loss of interest margin.
      - FN = Predicted Good (0), Actual Default (1) -> Actual Default loss (NPA, loss of principal * LGD).
      Notice: Defaulting on a loan is catastrophic. An actual defaulter approved is an FN when Default=1.
      To prevent ambiguity, let:
      - Cost of approving a bad borrower (Actual=1, Predicted=0) = cost_bad_approval (e.g. 1.0)
      - Cost of rejecting a good borrower (Actual=0, Predicted=1) = cost_opportunity_loss (e.g. 0.08)

    Args:
        y_true: Ground truth default indicator (1 for Default, 0 for Non-Default)
        y_prob_default: Calibrated probability of default
        cost_fp: Opportunity loss weight (rejecting a good customer)
        cost_fn: Default loss weight (approving a bad customer)
        num_thresholds: Number of threshold evaluation points

    Returns:
        best_threshold (float): Threshold minimizing total cost
        min_cost (float): The minimum business cost achieved
        cost_curve_df (pd.DataFrame): DataFrame with thresholds, costs, FP, FN, precision, recall
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob_default)

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    curve_data = []

    best_threshold = 0.5
    min_cost = float('inf')

    # Baseline cost at naive 0.5 threshold
    for thresh in thresholds:
        # Predict 1 (Default/Reject) if prob >= thresh
        y_pred = (y_prob >= thresh).astype(int)

        # FP: Actual 0 (Good), Predicted 1 (Rejected) -> Lost customer opportunity
        fp = np.sum((y_true == 0) & (y_pred == 1))
        # FN: Actual 1 (Default), Predicted 0 (Approved) -> Bad loan approved (NPA loss)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        # TP: Actual 1 (Default), Predicted 1 (Rejected) -> Correct risk avoidance
        tp = np.sum((y_true == 1) & (y_pred == 1))
        # TN: Actual 0 (Good), Predicted 0 (Approved) -> Profitable loan
        tn = np.sum((y_true == 0) & (y_pred == 0))

        # Total business cost = (loss from bad loans) + (opportunity cost of rejected good loans)
        # Here: FN * cost_bad_loan + FP * cost_lost_customer
        cost = (fn * cost_fp) + (fp * cost_fn)

        curve_data.append({
            'threshold': thresh,
            'total_cost': cost,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0
        })

        if cost < min_cost:
            min_cost = cost
            best_threshold = thresh

    cost_curve_df = pd.DataFrame(curve_data)
    return float(best_threshold), float(min_cost), cost_curve_df
