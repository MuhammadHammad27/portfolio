try:
    import pytest
except ImportError:
    pytest = None

from fastapi.testclient import TestClient
from backend.app.main import app

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Online"
        assert "endpoints" in data

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Online"
        assert "model_loaded" in data
        assert "features_count" in data

def test_predict_endpoint_valid_low_risk():
    with TestClient(app) as client:
        payload = {
            "person_age": 30,
            "person_income": 85000.0,
            "person_emp_length": 6.0,
            "loan_amnt": 8000.0,
            "loan_int_rate": 8.5,
            "loan_percent_income": 0.09,
            "cb_person_cred_hist_length": 7,
            "person_home_ownership": "MORTGAGE",
            "loan_intent": "VENTURE",
            "loan_grade": "A",
            "cb_person_default_on_file": "N"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "approved" in data
        assert "calibrated_probability" in data
        assert "risk_level" in data
        assert "risk_score" in data
        assert "shap_explanations" in data
        assert isinstance(data["shap_explanations"], list)
        assert len(data["shap_explanations"]) > 0

def test_predict_endpoint_validation_error():
    with TestClient(app) as client:
        # Age under 18, negative income
        invalid_payload = {
            "person_age": 15,
            "person_income": -100.0,
            "person_emp_length": 2.0,
            "loan_amnt": 1000.0,
            "loan_int_rate": 10.0,
            "loan_percent_income": 0.1,
            "cb_person_cred_hist_length": 1,
            "person_home_ownership": "INVALID_STATUS",
            "loan_intent": "PERSONAL",
            "loan_grade": "A",
            "cb_person_default_on_file": "N"
        }
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

if __name__ == "__main__":
    print("Running backend tests...")
    test_root_endpoint()
    print("[PASS] test_root_endpoint passed")
    test_health_endpoint()
    print("[PASS] test_health_endpoint passed")
    test_predict_endpoint_validation_error()
    print("[PASS] test_predict_endpoint_validation_error passed")
    test_predict_endpoint_valid_low_risk()
    print("[PASS] test_predict_endpoint_valid_low_risk passed")
    print("All tests passed successfully!")

