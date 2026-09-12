"""Unit tests for Population Stability Index (PSI) drift detector."""

import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift_detector import PopulationStabilityIndex


def test_psi_identical_distributions():
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        "feature_a": np.random.normal(50, 10, 1000),
        "feature_b": np.random.uniform(10, 100, 1000)
    })

    detector = PopulationStabilityIndex(num_bins=10)
    detector.fit(baseline_data)

    # Test on new sample from exact same distribution
    live_sample = pd.DataFrame({
        "feature_a": np.random.normal(50, 10, 500),
        "feature_b": np.random.uniform(10, 100, 500)
    })

    report = detector.evaluate_batch_drift(live_sample)
    assert report["overall_mean_psi"] < 0.10
    assert report["system_status"] == "HEALTHY"


def test_psi_drastic_drift():
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        "feature_a": np.random.normal(50, 10, 1000)
    })

    detector = PopulationStabilityIndex(num_bins=10)
    detector.fit(baseline_data)

    # Shifted distribution (mean shifts from 50 to 120)
    drifted_sample = pd.DataFrame({
        "feature_a": np.random.normal(120, 10, 500)
    })

    report = detector.evaluate_batch_drift(drifted_sample)
    assert report["overall_mean_psi"] > 0.25
    assert report["system_status"] == "CRITICAL_DRIFT_ALERT"
