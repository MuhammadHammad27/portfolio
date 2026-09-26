import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Explainable Credit Risk API")

MODEL_PATH = os.path.join("model", "loan_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
calibrated_model = artifact["calibrated_model"]
explainer = artifact["explainer"]
feature_names = artifact["feature_names"]

class LoanRequest(BaseModel):
    person_age: int = 25
    person_income: float = 60000.0
    person_emp_length: float = 3.0
    loan_amnt: float = 10000.0
    loan_int_rate: float = 11.0
    loan_percent_income: float = 0.15
    cb_person_cred_hist_length: int = 3
    person_home_ownership: str = "RENT"
    loan_intent: str = "PERSONAL"
    loan_grade: str = "B"
    cb_person_default_on_file: str = "N"

@app.get("/")
def health_check():
    return {"status": "Online", "message": "Credit Risk API is running"}

@app.post("/predict")
def predict_loan(data: LoanRequest):
    try:
        input_df = pd.DataFrame([data.dict()])
        transformed_input = preprocessor.transform(input_df)

        # 1. Prediction extraction with numpy flattening
        prediction_raw = calibrated_model.predict(transformed_input)
        prediction = int(np.ravel(prediction_raw)[0])

        # 2. Probability extraction safely
        probabilities = calibrated_model.predict_proba(transformed_input)
        prob_array = np.ravel(probabilities)
        # Handle 2-class or 1-class probability array
        calibrated_prob = round(float(prob_array[1] if len(prob_array) > 1 else prob_array[0]), 4)

        # 3. SHAP Values Extraction safely
        shap_values = explainer.shap_values(transformed_input)

        if isinstance(shap_values, list):
            # Binary classification usually returns list of 2 arrays [class_0, class_1]
            raw_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            raw_vals = shap_values

        vals = np.array(raw_vals).reshape(-1)

        explanations = [
            {"feature": str(feat), "shap_value": round(float(v), 4)}
            for feat, v in zip(feature_names, vals)
        ]

        return {
            "approved": bool(prediction == 0),
            "calibrated_probability": calibrated_prob,
            "risk_level": "Low Risk" if calibrated_prob < 0.4 else "High Risk",
            "message": "Loan Approved" if prediction == 0 else "Loan Rejected",
            "shap_explanations": explanations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))