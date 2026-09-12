"""Unit tests for financial feature engineering formulas."""

import pytest
import pandas as pd
import numpy as np
from src.features.financial_ratios import FinancialRatioTransformer


def test_financial_ratio_formulas():
    raw_data = {
        'loan_amount': [10000000.0],
        'loan_term': [10],
        'income_annum': [5000000.0],
        'residential_assets_value': [4000000.0],
        'commercial_assets_value': [6000000.0],
        'luxury_assets_value': [2000000.0],
        'bank_asset_value': [3000000.0],
        'cibil_score': [750]
    }
    df = pd.DataFrame(raw_data)
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.fit_transform(df)

    # Expected:
    # Collateral = 4M + 6M = 10M
    # LTV = 10M / 10M = 1.0
    # DTI = (10M / 10) / 5M = 1M / 5M = 0.20
    # Liquidity = 3M / (2M + 4M + 6M) = 3M / 12M = 0.25
    # Total Assets = 3M + 2M + 4M + 6M = 15M
    # Asset to Loan = 15M / 10M = 1.50

    np.testing.assert_almost_equal(transformed_df['ltv_ratio'].iloc[0], 1.0, decimal=3)
    np.testing.assert_almost_equal(transformed_df['dti_ratio'].iloc[0], 0.20, decimal=3)
    np.testing.assert_almost_equal(transformed_df['liquidity_ratio'].iloc[0], 0.25, decimal=3)
    np.testing.assert_almost_equal(transformed_df['asset_to_loan_ratio'].iloc[0], 1.50, decimal=3)
    np.testing.assert_almost_equal(transformed_df['total_assets'].iloc[0], 15000000.0, decimal=1)


def test_zero_division_guard():
    raw_data = {
        'loan_amount': [1000000.0],
        'loan_term': [0],  # Zero term edge case
        'income_annum': [0.0],  # Zero income edge case
        'residential_assets_value': [0.0],
        'commercial_assets_value': [0.0],
        'luxury_assets_value': [0.0],
        'bank_asset_value': [0.0],
        'cibil_score': [500]
    }
    df = pd.DataFrame(raw_data)
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.fit_transform(df)

    # Must not contain inf or NaN
    assert not np.isnan(transformed_df['ltv_ratio'].iloc[0])
    assert not np.isinf(transformed_df['ltv_ratio'].iloc[0])
    assert not np.isnan(transformed_df['dti_ratio'].iloc[0])
    assert not np.isinf(transformed_df['dti_ratio'].iloc[0])
