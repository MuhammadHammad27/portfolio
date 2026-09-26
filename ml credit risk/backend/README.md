# ⚙️ Credit Risk Assessment Backend API

Production-grade RESTful API built with **FastAPI**, **Pydantic V2**, **Scikit-Learn**, and **SHAP** for calibrated loan default prediction and feature explainability.

---

## 🏗️ Architecture & Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                # Application factory, lifespan, CORS, error handling
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Settings & environment variables
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── loan.py            # Pydantic schemas (LoanRequest, PredictionResponse, etc.)
│   ├── services/
│   │   ├── __init__.py
│   │   └── model_service.py   # Model artifact loader, cache & inference engine
│   └── api/
│       ├── __init__.py
│       ├── router.py          # Master API router
│       └── v1/
│           ├── router.py
│           └── endpoints/
│               ├── health.py  # Health and diagnostic checks
│               └── predict.py # Loan risk inference & SHAP explainability
├── tests/
│   ├── __init__.py
│   └── test_api.py            # Automated test suite using TestClient
├── api/
│   └── index.py               # Vercel serverless entrypoint
├── Dockerfile                 # Production multi-stage Docker container
├── requirements.txt           # Backend-specific Python dependencies
├── vercel.json                # Optional Vercel serverless configuration
└── .env.example               # Template environment configuration
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
cp .env.example .env
```
Default configuration values:
- `HOST=0.0.0.0`
- `PORT=8000`
- `MODEL_PATH=model/loan_model.pkl`

### 3. Run Development Server
From the workspace root:
```bash
python run.py --backend
# Or directly with uvicorn:
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive Documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Testing

Run automated unit and integration tests:
```bash
# Run via test runner
python -m backend.tests.test_api

# Or with pytest (if installed)
pytest backend/tests/test_api.py -v
```

---

## 📡 API Specification

### `GET /health` or `GET /api/v1/health`
Returns system status, active version, model memory state, and feature count.

**Response**:
```json
{
  "status": "Online",
  "version": "1.0.0",
  "model_loaded": true,
  "features_count": 26
}
```

### `POST /predict` or `POST /api/v1/predict`
Calculates default probability, risk tier, decision message, and SHAP feature importance.

**Payload**:
```json
{
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
```

**Response**:
```json
{
  "approved": true,
  "calibrated_probability": 0.0823,
  "risk_level": "Low Risk",
  "risk_score": 82,
  "message": "Loan Approved: Applicant profile meets risk thresholds.",
  "inference_time_ms": 12.4,
  "shap_explanations": [
    {
      "feature": "loan_percent_income",
      "shap_value": -0.421,
      "impact": "Decreases Default Risk"
    }
  ]
}
```

---

## 🐳 Docker Deployment

Build and run standalone backend container:
```bash
# From workspace root
docker build -t credit-risk-backend -f backend/Dockerfile .
docker run -p 8000:8000 credit-risk-backend
```
