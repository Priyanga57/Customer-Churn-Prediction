import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to **Churn** or **Stay**")

st.divider()

# -------------------------------------------------
# Paths (MATCH FOLDER NAME WITH SPACE)
# -------------------------------------------------
MODEL_PATH = "saved model/ann_churn_model.h5"
SCALER_PATH = "saved model/scaler.pkl"

# -------------------------------------------------
# Safety Checks (PREVENT CRASH)
# -------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"Scaler file not found at: {SCALER_PATH}")
    st.stop()

# -------------------------------------------------
# Load Model & Scaler
# -------------------------------------------------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.success("Model and scaler loaded successfully")

st.divider()

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
tenure = st.number_input("Tenure in Months", min_value=0, max_value=72, value=12)
monthly_charge = st.number_input("Monthly Charge", min_value=0.0, max_value=500.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

# -------------------------------------------------
# Create Input DataFrame
# -------------------------------------------------
input_df = pd.DataFrame([[
    tenure,
    monthly_charge,
    total_charges,
    age,
    dependents
]], columns=[
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Age",
    "Number of Dependents"
])

# -------------------------------------------------
# Feature Alignment (CRITICAL)
# -------------------------------------------------
expected_features = scaler.mean_.shape[0]
current_features = input_df.shape[1]

# Add missing dummy features
if current_features < expected_features:
    for i in range(expected_features - current_features):
        input_df[f"dummy_{i}"] = 0

# Trim extra columns if any
input_df = input_df.iloc[:, :expected_features]

# Scale input
input_scaled = scaler.transform(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Churn"):
    probability = model.predict(input_scaled)[0][0]

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** `{probability:.2f}`")

    if probability > 0.5:
        st.error("Customer is likely to **CHURN**")
    else:
        st.success("Customer is likely to **STAY**")

st.divider()
st.caption("ANN Model | Streamlit Cloud Deployment | Customer Churn Prediction")
