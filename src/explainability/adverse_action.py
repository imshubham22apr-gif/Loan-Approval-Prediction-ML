"""Adverse Action Notice & Counterfactual Recourse Module.

Under FCRA and ECOA regulatory guidelines, loan denial requires clear,
actionable recourse instructions explaining how the applicant can alter
their profile (e.g. increase credit score, reduce loan quantum, extend tenure, pledge collateral)
to reduce risk and reach approval.
"""

import copy
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class AdverseActionGenerator:
    """Generates actionable counterfactual recommendations for rejected loan applicants."""

    def __init__(self, full_pipeline, optimal_threshold: float = 0.15):
        """Args:

        full_pipeline: Complete Scikit-learn Pipeline (with preprocessing +
        calibrated classifier)
        optimal_threshold: Cost-optimal threshold for default risk (if PD <
        thresh -> Approve)
        """
        self.pipeline = full_pipeline
        self.threshold = optimal_threshold

    def get_recourse_actions(
        self,
        applicant_raw_dict: Dict[str, Any],
        max_recommendations: int = 4
    ) -> List[Dict[str, Any]]:
        """Evaluates actionable counterfactual scenarios to reduce risk and achieve loan approval.

        Tests realistic financial interventions:
        1. Increasing CIBIL score
        2. Lowering loan principal amount requested
        3. Extending loan repayment term (lowering DTI burden)
        4. Increasing liquid bank deposit assets
        5. Combined adjustments
        """
        base_df = pd.DataFrame([applicant_raw_dict])
        base_prob_default = float(self.pipeline.predict_proba(base_df)[0, 1])

        # If already below threshold, loan is already approved
        if base_prob_default < self.threshold:
            return []

        candidates = []

        # Intervention 1: CIBIL Score Improvement
        curr_cibil = int(applicant_raw_dict.get('cibil_score', 600))
        for target_cibil in [curr_cibil + 50, curr_cibil + 100, 750, 800, 850]:
            candidate_cibil = min(900, target_cibil)
            if candidate_cibil > curr_cibil:
                test_dict = copy.deepcopy(applicant_raw_dict)
                test_dict['cibil_score'] = candidate_cibil
                test_df = pd.DataFrame([test_dict])
                new_pd = float(self.pipeline.predict_proba(test_df)[0, 1])
                diff_pts = candidate_cibil - curr_cibil
                risk_reduction = (base_prob_default - new_pd) * 100
                achieves_approval = new_pd < self.threshold
                candidates.append({
                    "action_type": "CIBIL_SCORE_IMPROVEMENT",
                    "description": f"Improve CIBIL score from {curr_cibil} to {candidate_cibil} (+{diff_pts} pts) by clearing outstanding balances.",
                    "new_pd": round(new_pd, 4),
                    "risk_reduction_pct": round(risk_reduction, 2),
                    "achieves_approval": achieves_approval,
                    "effort_level": "Medium" if diff_pts <= 50 else "High"
                })
                if achieves_approval:
                    break

        # Intervention 2: Reducing Requested Loan Amount
        curr_loan = float(applicant_raw_dict.get('loan_amount', 10000000))
        for pct in [0.15, 0.30, 0.50, 0.70]:
            new_loan = curr_loan * (1.0 - pct)
            test_dict = copy.deepcopy(applicant_raw_dict)
            test_dict['loan_amount'] = new_loan
            test_df = pd.DataFrame([test_dict])
            new_pd = float(self.pipeline.predict_proba(test_df)[0, 1])
            reduction_amt = curr_loan - new_loan
            risk_reduction = (base_prob_default - new_pd) * 100
            achieves_approval = new_pd < self.threshold
            candidates.append({
                "action_type": "LOAN_AMOUNT_REDUCTION",
                "description": f"Reduce loan amount request by {int(pct * 100)}% (reduce by INR {reduction_amt:,.0f} to INR {new_loan:,.0f}) to reduce LTV leverage.",
                "new_pd": round(new_pd, 4),
                "risk_reduction_pct": round(risk_reduction, 2),
                "achieves_approval": achieves_approval,
                "effort_level": "Immediate"
            })
            if achieves_approval:
                break

        # Intervention 3: Extending Loan Tenure (Lowering DTI / Annual burden)
        curr_term = int(applicant_raw_dict.get('loan_term', 5))
        for ext in [2, 5, 10]:
            new_term = min(30, curr_term + ext)
            if new_term > curr_term:
                test_dict = copy.deepcopy(applicant_raw_dict)
                test_dict['loan_term'] = new_term
                test_df = pd.DataFrame([test_dict])
                new_pd = float(self.pipeline.predict_proba(test_df)[0, 1])
                risk_reduction = (base_prob_default - new_pd) * 100
                achieves_approval = new_pd < self.threshold
                candidates.append({
                    "action_type": "TENURE_EXTENSION",
                    "description": f"Extend loan tenure from {curr_term} to {new_term} years to decrease annual Debt-to-Income (DTI) debt burden.",
                    "new_pd": round(new_pd, 4),
                    "risk_reduction_pct": round(risk_reduction, 2),
                    "achieves_approval": achieves_approval,
                    "effort_level": "Immediate"
                })
                if achieves_approval:
                    break

        # Intervention 4: Joint Action (CIBIL + Loan Quantum reduction)
        test_dict = copy.deepcopy(applicant_raw_dict)
        test_dict['cibil_score'] = min(800, curr_cibil + 100)
        test_dict['loan_amount'] = curr_loan * 0.75
        test_df = pd.DataFrame([test_dict])
        new_pd = float(self.pipeline.predict_proba(test_df)[0, 1])
        risk_reduction = (base_prob_default - new_pd) * 100
        candidates.append({
            "action_type": "COMBINED_RESTRUCTURING",
            "description": f"Co-sign or improve CIBIL to {test_dict['cibil_score']} combined with a 25% lower loan request (INR {test_dict['loan_amount']:,.0f}).",
            "new_pd": round(new_pd, 4),
            "risk_reduction_pct": round(risk_reduction, 2),
            "achieves_approval": new_pd < self.threshold,
            "effort_level": "Medium"
        })

        # Sort: items that achieve approval first, then by highest risk reduction
        candidates.sort(key=lambda x: (x["achieves_approval"], x["risk_reduction_pct"]), reverse=True)

        # Remove duplicate action types to present diverse choices
        seen_types = set()
        unique_recs = []
        for c in candidates:
            if c["action_type"] not in seen_types and c["risk_reduction_pct"] > 0:
                unique_recs.append(c)
                seen_types.add(c["action_type"])

        return unique_recs[:max_recommendations]
