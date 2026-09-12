"""FastAPI endpoint integration and Pydantic validation tests."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "optimal_threshold" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "roc_auc" in data
    assert "brier_score" in data
    assert "fairness_audit" in data


def test_predict_approved_applicant(client):
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "Approved"
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert len(data["top_approval_drivers"]) > 0
    assert "financial_ratios" in data


def test_predict_rejected_applicant_has_remedies(client):
    bad_payload = {
        "no_of_dependents": 3,
        "education": "Not Graduate",
        "self_employed": "Yes",
        "income_annum": 1200000.0,
        "loan_amount": 8500000.0,
        "loan_term": 4,
        "cibil_score": 390,
        "residential_assets_value": 1000000.0,
        "commercial_assets_value": 500000.0,
        "luxury_assets_value": 2000000.0,
        "bank_asset_value": 400000.0
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "Rejected"
    assert len(data["adverse_action_remedies"]) > 0
    assert "action_type" in data["adverse_action_remedies"][0]


def test_pydantic_bounds_validation_error(client):
    # CIBIL score < 300 should fail Pydantic validation
    invalid_payload = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 5000000.0,
        "loan_amount": 10000000.0,
        "loan_term": 10,
        "cibil_score": 250,  # Below 300 minimum
        "residential_assets_value": 1000000.0,
        "commercial_assets_value": 1000000.0,
        "luxury_assets_value": 1000000.0,
        "bank_asset_value": 1000000.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
