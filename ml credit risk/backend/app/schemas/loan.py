from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class LoanRequest(BaseModel):
    person_age: int = Field(28, ge=18, le=100, description="Age of the loan applicant in years")
    person_income: float = Field(55000.0, ge=0.0, description="Annual income of the applicant in USD")
    person_emp_length: float = Field(4.0, ge=0.0, le=60.0, description="Employment duration in years")
    loan_amnt: float = Field(12000.0, ge=500.0, description="Requested loan amount in USD")
    loan_int_rate: float = Field(10.5, ge=1.0, le=40.0, description="Loan interest rate percentage")
    loan_percent_income: float = Field(0.22, ge=0.0, le=1.0, description="Ratio of loan amount to annual income (0.0 to 1.0)")
    cb_person_cred_hist_length: int = Field(5, ge=0, le=50, description="Credit history length in years")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        "RENT", description="Home ownership status"
    )
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ] = Field("PERSONAL", description="Purpose of the loan")
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        "B", description="Internal risk grade assigned to the loan"
    )
    cb_person_default_on_file: Literal["N", "Y"] = Field(
        "N", description="Historical default on file indicator"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 28,
                "person_income": 65000.0,
                "person_emp_length": 5.0,
                "loan_amnt": 10000.0,
                "loan_int_rate": 9.5,
                "loan_percent_income": 0.15,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "MORTGAGE",
                "loan_intent": "PERSONAL",
                "loan_grade": "A",
                "cb_person_default_on_file": "N"
            }
        }
    }

class ShapExplanation(BaseModel):
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP attribution value")
    impact: str = Field(..., description="Human-readable impact direction on default risk")

class PredictionResponse(BaseModel):
    approved: bool = Field(..., description="Whether the loan is approved (default prediction == 0)")
    calibrated_probability: float = Field(..., description="Calibrated probability of default (0.0 - 1.0)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, or High Risk")
    risk_score: int = Field(..., description="Scaled risk score from 0 to 1000 (lower is safer)")
    message: str = Field(..., description="Decision summary message")
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")
    shap_explanations: List[ShapExplanation] = Field(..., description="SHAP feature importance attributions")

class HealthCheckResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str = Field("Online", description="Health status of the API")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Indicator if model artifact is loaded in memory")
    features_count: int = Field(..., description="Number of model features")
