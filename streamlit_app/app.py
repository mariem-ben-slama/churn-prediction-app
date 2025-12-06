import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_single_input

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF4B4B, #1F77B4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #1F77B4;
        border-bottom: 2px solid #1F77B4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Prediction result boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .churn-risk {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        color: white;
        border: 3px solid #ff4444;
    }
    
    .no-churn-risk {
        background: linear-gradient(135deg, #56ab2f, #a8e6cf);
        color: white;
        border: 3px solid #44ff44;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #1F77B4, #FF4B4B);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1F77B4;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL AND ARTIFACTS
# =============================================================================
# Path of folder containing this file (streamlit_app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path of the project root (one folder up)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Correct model paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(PROJECT_ROOT, "models", "training_columns.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    training_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    st.error(f" Error loading model files: {e}")
    st.stop()

# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown('<div class="main-title">📱 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        Predict which customers are at risk of leaving your service. 
        Enter customer details below to get instant churn predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR FOR ADDITIONAL INFO
# =============================================================================
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This churn prediction app uses machine learning to identify customers 
    who are likely to cancel their subscription.
    
    **How it works:**
    - Enter customer details in the form
    - Click 'Predict Churn'
    - Get instant risk assessment
    """)
    
    st.markdown("## Model Info")
    st.markdown("""
    - **Algorithm**: Logistic Regression
    - **Training Data**: Telco Customer Churn Dataset
    - **Features**: 20+ customer attributes
    """)

# =============================================================================
# MAIN CONTENT - CUSTOMER INPUT FORM
# =============================================================================

# -------- FIRST ROW: DEMOGRAPHICS + INTERNET --------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown('<div class="section-header">👤 Customer Demographics</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=100, value=12)

with row1_col2:
    st.markdown('<div class="section-header">🌐 Internet Services</div>', unsafe_allow_html=True)
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])


# -------- SECOND ROW: PHONE + BILLING --------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown('<div class="section-header">📞 Phone Services</div>', unsafe_allow_html=True)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with row2_col2:
    st.markdown('<div class="section-header">💰 Billing Information</div>', unsafe_allow_html=True)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)



# =============================================================================
# DATA VALIDATION
# =============================================================================
if tenure < 0:
    st.warning("⚠️ Tenure cannot be negative")
    st.stop()

if MonthlyCharges < 0 or TotalCharges < 0:
    st.warning("⚠️ Charges cannot be negative")
    st.stop()

# =============================================================================
# PREDICTION SECTION
# =============================================================================
st.markdown("---")
st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

# Create user input dictionary
user_input = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# Convert to DataFrame and preprocess
input_df = preprocess_single_input(user_input, scaler, training_columns)

# Prediction button and results
if st.button("Predict Churn Risk"):
    with st.spinner('Analyzing customer data...'):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display results with styling
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box churn-risk">
                ⚠️ HIGH CHURN RISK ⚠️<br>
                <span style="font-size: 2rem;">Probability: {probability:.1%}</span><br>
                <small>This customer is likely to churn. Consider retention strategies.</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Show some retention suggestions
            with st.expander("Retention Suggestions"):
                st.markdown("""
                - **Offer loyalty discount**
                - **Proactive customer service call**
                - **Personalized retention offer**
                - **Survey to understand pain points**
                """)
        else:
            st.markdown(f"""
            <div class="prediction-box no-churn-risk">
                ✅ LOW CHURN RISK ✅<br>
                <span style="font-size: 2rem;">Probability: {probability:.1%}</span><br>
                <small>This customer is likely to stay. Focus on satisfaction.</small>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; margin-top: 3rem;'>"
    "Built with ❤️ using Streamlit & Scikit-learn"
    "</div>", 
    unsafe_allow_html=True
)