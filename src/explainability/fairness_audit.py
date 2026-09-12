"""Fair Lending & Bias Audit Module.

Calculates Disparate Impact Ratio and Demographic Parity metrics across
protected / proxy features (e.g. education, self_employed) to verify
adherence to US ECOA / Indian Fair Lending regulations (Four-Fifths / 80% rule).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def calculate_disparate_impact(
    df: pd.DataFrame,
    protected_col: str,
    privileged_group: Any,
    unprivileged_group: Any,
    prediction_col: str
) -> Dict[str, Any]:
    """Calculates Disparate Impact Ratio:

    DIR = P(Approval | Unprivileged) / P(Approval | Privileged)

    Under the 4/5ths (80%) rule:
    - DIR >= 0.80: Compliant / No adverse impact detected
    - DIR < 0.80: Potential adverse impact / regulatory scrutiny required
    """
    priv_mask = df[protected_col] == privileged_group
    unpriv_mask = df[protected_col] == unprivileged_group

    priv_total = priv_mask.sum()
    unpriv_total = unpriv_mask.sum()

    if priv_total == 0 or unpriv_total == 0:
        return {
            "feature": protected_col,
            "disparate_impact_ratio": None,
            "status": "INSUFFICIENT_DATA"
        }

    # Approvals (1 for Approved, 0 for Rejected)
    priv_approvals = df.loc[priv_mask, prediction_col].sum()
    unpriv_approvals = df.loc[unpriv_mask, prediction_col].sum()

    rate_priv = priv_approvals / priv_total
    rate_unpriv = unpriv_approvals / unpriv_total

    dir_ratio = rate_unpriv / (rate_priv + 1e-7)

    is_fair = dir_ratio >= 0.80

    return {
        "feature": protected_col,
        "privileged_group": str(privileged_group),
        "unprivileged_group": str(unprivileged_group),
        "privileged_approval_rate": round(float(rate_priv), 4),
        "unprivileged_approval_rate": round(float(rate_unpriv), 4),
        "disparate_impact_ratio": round(float(dir_ratio), 4),
        "four_fifths_compliant": bool(is_fair),
        "status": "PASS" if is_fair else "WARNING_ADVERSE_IMPACT"
    }


def audit_model_fairness(
    df: pd.DataFrame,
    prediction_col: str
) -> List[Dict[str, Any]]:
    """Runs a complete demographic audit over available proxy attributes."""
    audits = []

    # 1. Audit on Education: Graduate (Privileged) vs Not Graduate (Unprivileged)
    if 'education' in df.columns:
        res = calculate_disparate_impact(
            df=df,
            protected_col='education',
            privileged_group='Graduate',
            unprivileged_group='Not Graduate',
            prediction_col=prediction_col
        )
        audits.append(res)

    # 2. Audit on Employment: Salaried / Not Self-Employed (Privileged) vs Self-Employed (Unprivileged)
    if 'self_employed' in df.columns:
        res = calculate_disparate_impact(
            df=df,
            protected_col='self_employed',
            privileged_group='No',
            unprivileged_group='Yes',
            prediction_col=prediction_col
        )
        audits.append(res)

    return audits
