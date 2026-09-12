"""SHAP Explainability Module for Credit Risk Models.

Provides:
- Local Waterfall attributions (individual applicant drivers)
- Top positive risk drivers & top mitigating approval drivers
- Global Feature Importance
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List


class CreditRiskExplainer:
    """Computes Shapley values using TreeExplainer for tree ensembles."""

    def __init__(self, pipeline_or_model, feature_names: List[str] = None):
        """Args:

        pipeline_or_model: Either full Pipeline or trained classifier
        feature_names: Optional list of processed feature names
        """
        self.pipeline = None
        if hasattr(pipeline_or_model, 'named_steps'):
            self.pipeline = pipeline_or_model
            model = pipeline_or_model.named_steps['classifier']
            preprocessor = pipeline_or_model.named_steps.get('preprocessor', None)
            if preprocessor and feature_names is None:
                try:
                    num_feats = preprocessor.transformers_[0][2]
                    cat_feats = list(preprocessor.transformers_[1][1].get_feature_names_out())
                    self.feature_names = list(num_feats) + cat_feats
                except Exception:
                    self.feature_names = feature_names
            else:
                self.feature_names = feature_names
        else:
            model = pipeline_or_model
            self.feature_names = feature_names

        self.model = model

        # If calibrated classifier, extract the fitted base estimator from calibrated_classifiers_
        if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            base_estimator = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, 'estimator_'):
            base_estimator = model.estimator_
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'n_classes_'):
            base_estimator = model.estimator
        else:
            base_estimator = model

        self.base_estimator = base_estimator
        self.explainer = shap.TreeExplainer(self.base_estimator)

    def explain_raw_applicant(self, applicant_dict: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """Directly explains a raw applicant dict using the enclosed pipeline."""
        if self.pipeline is None:
            raise ValueError("Explainer was initialized without a full Pipeline; cannot transform raw applicant.")

        df = pd.DataFrame([applicant_dict])
        df_ratios = self.pipeline.named_steps['financial_ratios'].transform(df)
        X_proc = self.pipeline.named_steps['preprocessor'].transform(df_ratios)
        return self.explain_instance(X_proc, top_k=top_k)

    def explain_instance(self, processed_features: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """Explains a single applicant's risk score."""
        if isinstance(processed_features, pd.DataFrame):
            feature_names = list(processed_features.columns)
            feats_arr = processed_features.values
        else:
            feature_names = self.feature_names or [f"f_{i}" for i in range(processed_features.shape[1])]
            feats_arr = np.asarray(processed_features)

        if feats_arr.ndim == 1:
            feats_arr = feats_arr.reshape(1, -1)

        shap_vals = self.explainer.shap_values(feats_arr)
        if isinstance(shap_vals, list):
            values = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                values = shap_vals[0, :, 1]
            elif shap_vals.ndim == 2:
                values = shap_vals[0]
            else:
                values = shap_vals
        else:
            values = np.asarray(shap_vals)

        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
            base_value = float(expected_val[1])
        else:
            base_value = float(np.ravel(expected_val)[0])

        feature_contribs = []
        for name, val, raw_input in zip(feature_names, values, feats_arr[0]):
            feature_contribs.append({
                "feature": name,
                "shap_value": float(val),
                "feature_value": float(raw_input)
            })

        sorted_by_risk = sorted(feature_contribs, key=lambda x: x["shap_value"], reverse=True)
        top_risk_drivers = [x for x in sorted_by_risk if x["shap_value"] > 0][:top_k]

        sorted_by_mitigation = sorted(feature_contribs, key=lambda x: x["shap_value"])
        top_approval_drivers = [x for x in sorted_by_mitigation if x["shap_value"] < 0][:top_k]

        return {
            "base_value": base_value,
            "top_risk_drivers": top_risk_drivers,
            "top_approval_drivers": top_approval_drivers,
            "all_attributions": {x["feature"]: x["shap_value"] for x in feature_contribs}
        }
