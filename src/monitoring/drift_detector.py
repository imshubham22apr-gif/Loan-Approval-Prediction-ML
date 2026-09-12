"""Population Stability Index (PSI) & Distribution Drift Detection Module.

Calculates standard banking credit drift metrics:
- PSI < 0.10: Stable (Green)
- 0.10 <= PSI < 0.25: Slight Drift / Warning (Yellow)
- PSI >= 0.25: Significant Drift / Mandatory Retraining (Red)
"""

import json
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd


class PopulationStabilityIndex:
    """Calculates Population Stability Index (PSI) between baseline (training)

    and production target (live inference) data distributions.
    """

    def __init__(self, num_bins: int = 10, epsilon: float = 1e-4):
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.baseline_stats_ = {}

    def fit(self, baseline_df: pd.DataFrame, numerical_cols: List[str] = None):
        """Extracts quantile bin edges and baseline proportions."""
        if numerical_cols is None:
            numerical_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numerical_cols:
            if col in baseline_df.columns:
                series = baseline_df[col].dropna()
                # Compute quantile bin edges
                percentiles = np.linspace(0, 100, self.num_bins + 1)
                bin_edges = np.percentile(series, percentiles)
                # Ensure strictly increasing bins
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 2:
                    bin_edges = np.array([series.min() - 1, series.max() + 1])

                # Calculate baseline distribution
                counts, _ = np.histogram(series, bins=bin_edges)
                proportions = counts / len(series)
                # Smooth zero bins
                proportions = np.where(proportions == 0, self.epsilon, proportions)
                proportions = proportions / np.sum(proportions)

                self.baseline_stats_[col] = {
                    "bin_edges": bin_edges.tolist(),
                    "expected_proportions": proportions.tolist()
                }
        return self

    def calculate_feature_psi(self, actual_series: pd.Series, col_name: str) -> float:
        """Calculates PSI for a single feature against learned baseline."""
        if col_name not in self.baseline_stats_:
            return 0.0

        bin_edges = np.array(self.baseline_stats_[col_name]["bin_edges"])
        expected_prop = np.array(self.baseline_stats_[col_name]["expected_proportions"])

        # Force edges to span -inf to +inf for unseen test extremes
        bins_extended = bin_edges.copy()
        bins_extended[0] = -np.inf
        bins_extended[-1] = np.inf

        counts, _ = np.histogram(actual_series.dropna(), bins=bins_extended)
        actual_prop = counts / max(len(actual_series.dropna()), 1)

        # Smooth zeros
        actual_prop = np.where(actual_prop == 0, self.epsilon, actual_prop)
        actual_prop = actual_prop / np.sum(actual_prop)

        # PSI = sum((Actual - Expected) * ln(Actual / Expected))
        psi_val = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
        return float(np.round(psi_val, 5))

    def evaluate_batch_drift(self, live_df: pd.DataFrame) -> Dict[str, Any]:
        """Audits an entire production batch for data drift."""
        feature_psis = {}
        high_drift_features = []
        moderate_drift_features = []

        for col in self.baseline_stats_:
            if col in live_df.columns:
                psi = self.calculate_feature_psi(live_df[col], col)
                status = "STABLE"
                if psi >= 0.25:
                    status = "HIGH_DRIFT"
                    high_drift_features.append(col)
                elif psi >= 0.10:
                    status = "MODERATE_DRIFT"
                    moderate_drift_features.append(col)

                feature_psis[col] = {
                    "psi": psi,
                    "status": status
                }

        overall_psi = np.mean([v["psi"] for v in feature_psis.values()]) if feature_psis else 0.0

        if overall_psi >= 0.25 or len(high_drift_features) > 0:
            system_status = "CRITICAL_DRIFT_ALERT"
            action = "Trigger model retraining pipeline immediately."
        elif overall_psi >= 0.10 or len(moderate_drift_features) > 0:
            system_status = "MODERATE_WARNING"
            action = "Monitor closely; sample drift detected."
        else:
            system_status = "HEALTHY"
            action = "No intervention needed."

        return {
            "overall_mean_psi": float(np.round(overall_psi, 4)),
            "system_status": system_status,
            "action_required": action,
            "high_drift_features": high_drift_features,
            "moderate_drift_features": moderate_drift_features,
            "feature_metrics": feature_psis
        }

    def save(self, filepath: str):
        """Serializes baseline statistics to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.baseline_stats_, f, indent=2)

    def load(self, filepath: str):
        """Loads baseline statistics from JSON."""
        with open(filepath, 'r') as f:
            self.baseline_stats_ = json.load(f)
        return self
