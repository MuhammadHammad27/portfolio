import os
import requests
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
    }
    .status-badge-online {
        background-color: #065F46;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-offline {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Backend API Configuration
# ---------------------------------------------------------
default_api = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
try:
    if "BACKEND_API_URL" in st.secrets:
        default_api = st.secrets["BACKEND_API_URL"]
except Exception:
    pass

# Sidebar Backend Config
st.sidebar.markdown("### ⚙️ System Configuration")
backend_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=default_api.rstrip("/"),
    help="Target endpoint of the running FastAPI server"
)

# Live Backend Health Diagnostic
def check_backend_health(url: str):
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
        return False, None
    except Exception:
        return False, None

is_healthy, health_info = check_backend_health(backend_url)

if is_healthy:
    st.sidebar.markdown(
        f'<div class="status-badge-online">● Backend Online (v{health_info.get("version", "1.0.0")})</div>',
        unsafe_allow_html=True
    )
    if health_info.get("model_loaded"):
        st.sidebar.caption(f"✓ Model loaded with {health_info.get('features_count', 26)} active features.")
    else:
        st.sidebar.warning("⚠ Model file not yet loaded in backend.")
else:
    st.sidebar.markdown(
        '<div class="status-badge-offline">● Backend Offline</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption("Run `uvicorn backend.app.main:app --reload` to start backend.")

st.sidebar.markdown("---")

# ---------------------------------------------------------
# Applicant Presets for Rapid Evaluation
# ---------------------------------------------------------
st.sidebar.markdown("### ⚡ Quick Presets")

PRESETS = {
    "Prime Borrower (Low Risk)": {
        "person_age": 34,
        "person_income": 95000.0,
        "person_emp_length": 8.0,
        "loan_amnt": 10000.0,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.10,
        "cb_person_cred_hist_length": 8,
        "person_home_ownership": "MORTGAGE",
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "cb_person_default_on_file": "N"
    },
    "Average Borrower (Moderate Risk)": {
        "person_age": 27,
        "person_income": 48000.0,
        "person_emp_length": 3.0,
        "loan_amnt": 12000.0,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.25,
        "cb_person_cred_hist_length": 4,
        "person_home_ownership": "RENT",
        "loan_intent": "EDUCATION",
        "loan_grade": "C",
        "cb_person_default_on_file": "N"
    },
    "Distressed Borrower (High Risk)": {
        "person_age": 22,
        "person_income": 24000.0,
        "person_emp_length": 0.5,
        "loan_amnt": 15000.0,
        "loan_int_rate": 16.8,
        "loan_percent_income": 0.62,
        "cb_person_cred_hist_length": 2,
        "person_home_ownership": "RENT",
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "E",
        "cb_person_default_on_file": "Y"
    }
}

selected_preset = st.sidebar.selectbox("Load Profile Preset", ["Custom"] + list(PRESETS.keys()))

# Manage session state based on preset
if "current_preset" not in st.session_state or st.session_state.current_preset != selected_preset:
    st.session_state.current_preset = selected_preset
    if selected_preset in PRESETS:
        for k, v in PRESETS[selected_preset].items():
            st.session_state[f"form_{k}"] = v

# Header
st.markdown('<div class="main-title">💳 Enterprise Credit Risk Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Calibrated default probability assessment with transparent TreeExplainer SHAP attribution.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Applicant Data Form
# ---------------------------------------------------------
with st.container():
    st.subheader("📋 Applicant & Loan Parameters")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Background**")
        person_age = st.number_input(
            "Age (Years)",
            min_value=18,
            max_value=100,
            value=int(st.session_state.get("form_person_age", 28)),
            key="input_age"
        )
        person_income = st.number_input(
            "Annual Income ($)",
            min_value=1000.0,
            max_value=2000000.0,
            value=float(st.session_state.get("form_person_income", 55000.0)),
            step=1000.0,
            key="input_income"
        )
        person_emp_length = st.number_input(
            "Employment Duration (Years)",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state.get("form_person_emp_length", 4.0)),
            step=0.5,
            key="input_emp"
        )
        home_options = ["RENT", "OWN", "MORTGAGE", "OTHER"]
        default_home = st.session_state.get("form_person_home_ownership", "RENT")
        person_home_ownership = st.selectbox(
            "Home Ownership",
            home_options,
            index=home_options.index(default_home) if default_home in home_options else 0,
            key="input_home"
        )

    with col2:
        st.markdown("**Loan Specifications**")
        loan_amnt = st.number_input(
            "Requested Loan Amount ($)",
            min_value=500.0,
            max_value=1000000.0,
            value=float(st.session_state.get("form_loan_amnt", 12000.0)),
            step=500.0,
            key="input_amount"
        )
        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=1.0,
            max_value=35.0,
            value=float(st.session_state.get("form_loan_int_rate", 10.5)),
            step=0.1,
            key="input_rate"
        )
        calculated_ratio = round(loan_amnt / max(person_income, 1.0), 3)
        st.caption(f"Calculated Loan-to-Income Ratio: **{calculated_ratio * 100:.1f}%**")
        
        loan_percent_income = st.number_input(
            "Loan % of Annual Income (0.0 - 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=min(1.0, float(st.session_state.get("form_loan_percent_income", min(0.99, calculated_ratio)))),
            step=0.01,
            key="input_ratio"
        )
        intent_options = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        default_intent = st.session_state.get("form_loan_intent", "PERSONAL")
        loan_intent = st.selectbox(
            "Loan Intent",
            intent_options,
            index=intent_options.index(default_intent) if default_intent in intent_options else 0,
            key="input_intent"
        )

    with col3:
        st.markdown("**Credit Bureau History**")
        grade_options = ["A", "B", "C", "D", "E", "F", "G"]
        default_grade = st.session_state.get("form_loan_grade", "B")
        loan_grade = st.selectbox(
            "Assigned Risk Grade",
            grade_options,
            index=grade_options.index(default_grade) if default_grade in grade_options else 1,
            key="input_grade"
        )
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            max_value=50,
            value=int(st.session_state.get("form_cb_person_cred_hist_length", 5)),
            key="input_cred_hist"
        )
        default_options = ["N", "Y"]
        default_val = st.session_state.get("form_cb_person_default_on_file", "N")
        cb_person_default_on_file = st.selectbox(
            "Prior Default Recorded",
            default_options,
            index=default_options.index(default_val) if default_val in default_options else 0,
            key="input_default"
        )

analyze_button = st.button("🚀 Run Risk Underwriting Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Inference & Results Presentation
# ---------------------------------------------------------
if analyze_button:
    payload = {
        "person_age": int(person_age),
        "person_income": float(person_income),
        "person_emp_length": float(person_emp_length),
        "loan_amnt": float(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file
    }

    with st.spinner("Submitting request to FastAPI inference engine..."):
        try:
            api_endpoint = f"{backend_url}/predict"
            response = requests.post(api_endpoint, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.markdown("---")
                st.subheader("🎯 Decision & Risk Overview")

                # Decision Banner
                if result["approved"]:
                    st.success(f"### ✅ {result['message']}")
                else:
                    st.error(f"### ❌ {result['message']}")

                # Metric Cards
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric(
                        label="Approval Status",
                        value="Approved" if result["approved"] else "Rejected",
                        delta="Eligible" if result["approved"] else "-High Risk"
                    )
                with mcol2:
                    st.metric(
                        label="Calibrated Default Prob.",
                        value=f"{result['calibrated_probability'] * 100:.1f}%",
                        delta=f"Risk Score: {result['risk_score']}/1000",
                        delta_color="inverse"
                    )
                with mcol3:
                    st.metric(
                        label="Assigned Risk Tier",
                        value=result["risk_level"]
                    )
                with mcol4:
                    st.metric(
                        label="Model Latency",
                        value=f"{result.get('inference_time_ms', 0):.1f} ms"
                    )

                # ---------------------------------------------------------
                # SHAP Feature Attribution
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader("🔍 Explainable AI (SHAP Feature Attribution)")
                st.markdown("Features with **positive values** push towards default risk, while **negative values** protect against default.")

                shap_list = result.get("shap_explanations", [])
                if shap_list:
                    df_shap = pd.DataFrame(shap_list)
                    
                    # Sort top 12 features by absolute impact for clean visualization
                    df_shap["abs_val"] = df_shap["shap_value"].abs()
                    top_shap = df_shap.sort_values(by="abs_val", ascending=True).tail(12)

                    # Plotly chart
                    try:
                        import plotly.graph_objects as go
                        
                        colors = ['#EF4444' if x > 0 else '#10B981' for x in top_shap["shap_value"]]
                        
                        fig = go.Figure(
                            go.Bar(
                                x=top_shap["shap_value"],
                                y=top_shap["feature"],
                                orientation='h',
                                marker=dict(color=colors),
                                text=[f"{v:+.3f}" for v in top_shap["shap_value"]],
                                textposition="auto"
                            )
                        )
                        fig.update_layout(
                            title="Top Feature Contributions to Default Risk",
                            xaxis_title="SHAP Value (Impact on Default Probability)",
                            yaxis_title="Feature",
                            template="plotly_dark",
                            height=480,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        # Fallback to standard Streamlit chart if plotly issue
                        st.bar_chart(top_shap.set_index("feature")["shap_value"])

                    # Detailed Explanations Table
                    with st.expander("📊 Complete Feature Breakdown Table", expanded=False):
                        display_df = df_shap[["feature", "shap_value", "impact"]].sort_values(
                            by="shap_value", ascending=False
                        )
                        st.dataframe(display_df, use_container_width=True)

            elif response.status_code == 422:
                st.error("Validation Error: Please verify all inputs match expected ranges.")
                st.json(response.json())
            else:
                st.error(f"Inference Engine Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"🚨 Unable to connect to FastAPI backend at `{backend_url}`. "
                "Ensure the backend server is running via `python backend/app/main.py` or Docker."
            )
        except Exception as ex:
            st.error(f"An unexpected error occurred: {ex}")