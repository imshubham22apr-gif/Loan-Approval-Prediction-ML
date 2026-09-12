"""Pydantic Request/Response Schemas with Credit Domain Constraints."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class ApplicantInput(BaseModel):
    """Raw loan applicant application data with strict banking bounds."""

    no_of_dependents: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Number of financial dependents"
    )
    education: Literal["Graduate", "Not Graduate"] = Field(
        default="Graduate",
        description="Educational qualification"
    )
    self_employed: Literal["Yes", "No"] = Field(
        default="No",
        description="Employment status: self-employed or salaried"
    )
    income_annum: float = Field(
        default=9600000.0,
        gt=0.0,
        description="Annual gross income in INR"
    )
    loan_amount: float = Field(
        default=29900000.0,
        gt=0.0,
        description="Requested loan principal amount in INR"
    )
    loan_term: int = Field(
        default=12,
        ge=1,
        le=30,
        description="Loan tenure in years"
    )
    cibil_score: int = Field(
        default=778,
        ge=300,
        le=900,
        description="Official CIBIL credit score (300 to 900)"
    )
    residential_assets_value: float = Field(
        default=2400000.0,
        ge=0.0,
        description="Market valuation of residential property owned in INR"
    )
    commercial_assets_value: float = Field(
        default=17600000.0,
        ge=0.0,
        description="Market valuation of commercial real estate owned in INR"
    )
    luxury_assets_value: float = Field(
        default=22700000.0,
        ge=0.0,
        description="Valuation of luxury assets (vehicles, art, jewelry) in INR"
    )
    bank_asset_value: float = Field(
        default=8000000.0,
        ge=0.0,
        description="Liquid capital and cash bank deposits in INR"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 9600000.0,
                "loan_amount": 29900000.0,
                "loan_term": 12,
                "cibil_score": 778,
                "residential_assets_value": 2400000.0,
                "commercial_assets_value": 17600000.0,
                "luxury_assets_value": 22700000.0,
                "bank_asset_value": 8000000.0
            }
        }
    )


class FeatureDriver(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class AdverseActionRecourse(BaseModel):
    action_type: str
    description: str
    new_pd: float
    risk_reduction_pct: float
    achieves_approval: bool
    effort_level: str


class FinancialRatiosResponse(BaseModel):
    ltv_ratio: float
    dti_ratio: float
    liquidity_ratio: float
    asset_to_loan_ratio: float
    total_assets: float


class CreditRiskPredictionResponse(BaseModel):
    """Comprehensive institutional underwriting decision response."""

    decision: Literal["Approved", "Rejected"]
    probability_of_default: float = Field(..., description="Calibrated default probability (0 to 1)")
    risk_tier: str = Field(..., description="Risk categorization: Low Risk, Moderate Risk, High Risk")
    expected_loss_inr: float = Field(..., description="Expected Financial Loss = PD * LGD * EAD")
    decision_threshold: float = Field(..., description="Cost-optimal decision threshold used")
    financial_ratios: FinancialRatiosResponse
    top_risk_drivers: List[FeatureDriver]
    top_approval_drivers: List[FeatureDriver]
    adverse_action_remedies: List[AdverseActionRecourse] = Field(
        default_factory=list,
        description="Actionable remediation steps if loan was rejected"
    )


class DriftCheckBatchRequest(BaseModel):
    applicants: List[ApplicantInput]


class DriftCheckResponse(BaseModel):
    overall_mean_psi: float
    system_status: str
    action_required: str
    high_drift_features: List[str]
    moderate_drift_features: List[str]
    feature_metrics: Dict[str, Any]
