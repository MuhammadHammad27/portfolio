# 🎨 Credit Risk Intelligence Frontend

Interactive underwriting and explainability dashboard built with **Streamlit** and **Plotly**.

---

## 🌟 Key Features

- **Real-Time Backend Diagnostics**: Automatically verifies connectivity with the FastAPI backend and displays an active status badge with loaded features count.
- **Applicant Profile Presets**: One-click selection of:
  - *Prime Borrower (Low Risk)*
  - *Average Borrower (Moderate Risk)*
  - *Distressed Borrower (High Risk)*
- **Decision & Risk Metrics**: Clear approval indicator, calibrated default probability, 0–1000 risk index, and inference latency.
- **Explainable AI (XAI)**: Diverging Plotly bar charts visualizing positive (risk-increasing) and negative (risk-reducing) SHAP attributions.
- **Dynamic Configuration**: Connects to any backend instance via environment variables or `st.secrets`.

---

## 📁 Structure

```text
frontend/
├── .streamlit/
│   └── config.toml      # Theme definition (dark mode, typography, colors)
├── app.py               # Main Streamlit dashboard application
├── Dockerfile           # Standalone production container
├── requirements.txt     # Frontend-specific Python dependencies
└── .env.example         # Template environment configuration
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd frontend
pip install -r requirements.txt
```

### 2. Configure Backend URL (Optional)
By default, the frontend connects to `http://127.0.0.1:8000`. To point to a custom or production backend:
- Create a `.env` file or export `BACKEND_API_URL`:
```bash
export BACKEND_API_URL="https://your-backend.onrender.com"
```
Or directly adjust the endpoint in the sidebar input box at runtime!

### 3. Run Application
From workspace root:
```bash
python run.py --frontend
# Or directly:
streamlit run frontend/app.py
```
Open browser at: [http://localhost:8501](http://localhost:8501)

---

## ☁️ Streamlit Community Cloud Deployment (100% Free)

Streamlit Community Cloud is the standard hosting environment for Streamlit apps:
1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **"New App"** and configure:
   - **Repository**: `<your-username>/ml-credit-risk`
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`
4. Under **Advanced settings**, set your environment variable:
   ```toml
   BACKEND_API_URL = "https://your-backend.onrender.com"
   ```
5. Click **Deploy**!

---

## 🐳 Docker Deployment

Build and run standalone frontend container:
```bash
# From workspace root
docker build -t credit-risk-frontend -f frontend/Dockerfile .
docker run -p 8501:8501 -e BACKEND_API_URL="http://host.docker.internal:8000" credit-risk-frontend
```
