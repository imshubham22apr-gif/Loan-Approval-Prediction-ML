"""Financial Feature Engineering Transformer.

Implements core banking and credit underwriting ratios:
- Loan-to-Value (LTV) Ratio
- Debt-to-Income / Debt Burden (DTI) Ratio
- Asset Liquidity Coverage Ratio
- Asset to Loan Coverage Ratio
- Total Collateral & Net Worth
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """Generates institutional banking ratios from raw loan and applicant asset data.

    Ensures zero data leakage and conforms to Scikit-learn Transformer API.
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        # Stateless transformer, but records column names
        if isinstance(X, pd.DataFrame):
            self.input_cols_ = list(X.columns)
        else:
            self.input_cols_ = None
        return self

    def transform(self, X):
        """Calculates domain financial ratios."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.input_cols_)

        # Ensure required numerical columns are converted to float
        numeric_cols = [
            'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
            'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 1. Total Collateral (Pledged Physical Assets)
        total_collateral = df['residential_assets_value'] + df['commercial_assets_value']

        # 2. Loan-to-Value (LTV) Ratio
        # Higher LTV = higher credit risk
        df['ltv_ratio'] = df['loan_amount'] / (total_collateral + self.epsilon)

        # 3. Debt-to-Income / Annual Debt Burden (DTI)
        # Annual EMI approximation: loan_amount / loan_term
        # Ratio of annual debt obligation to annual gross income
        annual_debt_obligation = df['loan_amount'] / np.maximum(df['loan_term'], 1)
        df['dti_ratio'] = annual_debt_obligation / (df['income_annum'] + self.epsilon)

        # 4. Liquid vs. Illiquid Asset Ratio (Asset Liquidity Coverage)
        # Ratio of immediately accessible bank capital to physical/luxury assets
        illiquid_assets = (
            df['luxury_assets_value']
            + df['residential_assets_value']
            + df['commercial_assets_value']
        )
        df['liquidity_ratio'] = df['bank_asset_value'] / (illiquid_assets + self.epsilon)

        # 5. Total Net Worth & Total Assets
        total_assets = (
            df['bank_asset_value']
            + df['luxury_assets_value']
            + df['residential_assets_value']
            + df['commercial_assets_value']
        )
        df['total_assets'] = total_assets

        # 6. Asset-to-Loan Coverage Ratio
        # Measures total balance-sheet coverage relative to requested loan amount
        df['asset_to_loan_ratio'] = total_assets / (df['loan_amount'] + self.epsilon)

        # Cap extreme outlier ratios for numerical stability in tree splits
        df['ltv_ratio'] = np.clip(df['ltv_ratio'], 0.0, 50.0)
        df['dti_ratio'] = np.clip(df['dti_ratio'], 0.0, 20.0)
        df['liquidity_ratio'] = np.clip(df['liquidity_ratio'], 0.0, 50.0)
        df['asset_to_loan_ratio'] = np.clip(df['asset_to_loan_ratio'], 0.0, 50.0)

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
