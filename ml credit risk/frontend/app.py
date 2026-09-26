import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Explainable Credit Risk Assessment System")
st.markdown("Enter applicant details below to predict loan approval and view SHAP explainability analysis.")

# Backend API URL
API_URL = "http://127.0.0.1:8000/predict"

st.sidebar.header("📋 Applicant Inputs")

# Sidebar Form Controls
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=28)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, value=55000)
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0.0, value=4.0, step=0.5)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=500, value=12000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, value=10.5, step=0.1)
loan_percent_income = st.sidebar.number_input("Loan Percent of Income (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, value=5)

person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.sidebar.selectbox("Historical Default on File", ["N", "Y"])

if st.button("🚀 Analyze Credit Risk"):
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file
    }

    try:
        with st.spinner("Processing prediction via FastAPI backend..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result["approved"]:
                    st.success(f"### Status: {result['message']}")
                else:
                    st.error(f"### Status: {result['message']}")

            with col2:
                st.metric("Risk Level", result["risk_level"])
                
            with col3:
                st.metric("Calibrated Default Probability", f"{result['calibrated_probability'] * 100:.1f}%")

            # SHAP Feature Explanations
            st.subheader("🔍 Feature Impact Analysis (SHAP Explanations)")
            shap_df = pd.DataFrame(result["shap_explanations"])
            shap_df = shap_df.sort_values(by="shap_value", ascending=False)

            st.bar_chart(shap_df.set_index("feature")["shap_value"])
            st.dataframe(shap_df, use_container_width=True)

        else:
            st.error(f"Error from Backend API: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")