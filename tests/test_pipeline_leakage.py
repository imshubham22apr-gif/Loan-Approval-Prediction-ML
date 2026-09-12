"""Test to verify absence of data leakage across pipeline steps."""

import pytest
import os
import joblib
import pandas as pd
import numpy as np


def test_pipeline_artifact_integrity():
    pipeline_path = "models/calibrated_credit_pipeline.joblib"
    assert os.path.exists(pipeline_path), "Trained pipeline artifact missing."

    pipeline = joblib.load(pipeline_path)

    # Verify pipeline components
    step_names = [step[0] for step in pipeline.steps]
    assert "financial_ratios" in step_names
    assert "preprocessor" in step_names
    assert "classifier" in step_names


def test_no_target_leakage_in_features():
    pipeline = joblib.load("models/calibrated_credit_pipeline.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]

    # Ensure target column is not in transformed features
    num_cols = preprocessor.transformers_[0][2]
    assert "loan_status" not in num_cols
    assert "default_risk_target" not in num_cols
    assert "loan_id" not in num_cols
