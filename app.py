"""
Customer Churn Prediction - Streamlit Web App
Author: Maria Anwar
Description: Interactive web app that predicts whether a telecom customer
is likely to churn, based on a trained ML model (Logistic Regression /
Random Forest / XGBoost - any sklearn-API compatible model works).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 0px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .result-box-churn {
            background-color: #ffe5e5;
            border: 2px solid #ff4b4b;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }
        .result-box-safe {
            background-color: #e5ffe9;
            border: 2px solid #21c354;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }
        .stButton>button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            font-weight: 600;
            padding: 10px;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #145a86;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD MODEL AND SCALER (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Make sure 'churn_prediction_model.pkl' "
        "and 'scaler.pkl' are in the same folder as this app."
    )
    st.stop()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter customer details below to predict the likelihood of churn</p>',
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------
# INPUT FORM
# ----------------------------------------------------------------------------
with st.form("churn_form"):

    st.subheader("👤 Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
    with col2:
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

    st.subheader("📞 Services")
    col3, col4 = st.columns(2)
    with col3:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with col4:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.subheader("💳 Account & Billing")
    col5, col6 = st.columns(2)
    with col5:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Bank transfer (automatic)", "Credit card (automatic)",
             "Electronic check", "Mailed check"]
        )
    with col6:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

    submitted = st.form_submit_button("🔮 Predict Churn")

# ----------------------------------------------------------------------------
# BUILD FEATURE VECTOR (must match the exact columns used during training)
# ----------------------------------------------------------------------------
def build_input_dataframe():
    """
    Recreates the same one-hot-encoded structure used during training
    (pd.get_dummies(..., drop_first=True)). Column order MUST match
    the training data exactly.
    """
    data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,

        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,

        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,

        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,

        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,

        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,

        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,

        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,

        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,

        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,

        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    }

    # Column order must exactly match X_train.columns from training
    column_order = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
        "MultipleLines_No phone service", "MultipleLines_Yes",
        "InternetService_Fiber optic", "InternetService_No",
        "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
        "OnlineBackup_No internet service", "OnlineBackup_Yes",
        "DeviceProtection_No internet service", "DeviceProtection_Yes",
        "TechSupport_No internet service", "TechSupport_Yes",
        "StreamingTV_No internet service", "StreamingTV_Yes",
        "StreamingMovies_No internet service", "StreamingMovies_Yes",
        "Contract_One year", "Contract_Two year",
        "PaperlessBilling_Yes",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
    ]

    df_input = pd.DataFrame([data])[column_order]
    return df_input

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
if submitted:
    input_df = build_input_dataframe()

    # Scale numeric features using the SAME scaler fitted on training data
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn (class 1)

    if prediction == 1:
        st.markdown(f"""
            <div class="result-box-churn">
                <h2>⚠️ High Risk of Churn</h2>
                <p style="font-size:1.2rem;">Churn Probability: <b>{probability*100:.1f}%</b></p>
                <p>This customer is likely to leave. Consider proactive retention offers.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box-safe">
                <h2>✅ Low Risk of Churn</h2>
                <p style="font-size:1.2rem;">Churn Probability: <b>{probability*100:.1f}%</b></p>
                <p>This customer is likely to stay.</p>
            </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Customer Churn Prediction Project by Maria Anwar")
