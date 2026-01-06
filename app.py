import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write("Demo deployment (model inference disabled due to cloud constraints)")

st.divider()

# -------------------------------------------------
# Load Scaler (ONLY)
# -------------------------------------------------
SCALER_PATH = "saved model/scaler.pkl"

if not os.path.exists(SCALER_PATH):
    st.error("Scaler file not found")
    st.stop()

scaler = joblib.load(SCALER_PATH)
st.success("Scaler loaded successfully")

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
tenure = st.number_input("Tenure in Months", 0, 72, 12)
monthly_charge = st.number_input("Monthly Charge", 0.0, 500.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
age = st.number_input("Age", 18, 100, 35)
dependents = st.number_input("Number of Dependents", 0, 10, 0)

# -------------------------------------------------
# Create NumPy Input (NO FEATURE NAMES)
# -------------------------------------------------
input_array = np.array([[tenure, monthly_charge, total_charges, age, dependents]])

# Pad with zeros to match scaler dimensions
expected_features = scaler.mean_.shape[0]
current_features = input_array.shape[1]

if current_features < expected_features:
    padding = np.zeros((1, expected_features - current_features))
    input_array = np.hstack([input_array, padding])

# Trim if extra (safety)
input_array = input_array[:, :expected_features]

# Scale (NO FEATURE NAME CHECK)
input_scaled = scaler.transform(input_array)

# -------------------------------------------------
# Demo Prediction
# -------------------------------------------------
if st.button("Predict Churn"):
    probability = np.random.uniform(0.2, 0.8)

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** `{probability:.2f}`")

    if probability > 0.5:
        st.error(" Customer is likely to CHURN")
    else:
        st.success("Customer is likely to STAY")

st.divider()
st.caption("Demo UI | Model trained offline | Streamlit Cloud deployment")
