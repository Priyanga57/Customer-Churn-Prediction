import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Demo deployment (model inference disabled due to cloud constraints)")

# Paths
SCALER_PATH = "saved model/scaler.pkl"

if not os.path.exists(SCALER_PATH):
    st.error("❌ Scaler file not found")
    st.stop()

scaler = joblib.load(SCALER_PATH)
st.success("✅ Scaler loaded successfully")

# Inputs
tenure = st.number_input("Tenure in Months", 0, 72, 12)
monthly_charge = st.number_input("Monthly Charge", 0.0, 500.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
age = st.number_input("Age", 18, 100, 35)
dependents = st.number_input("Number of Dependents", 0, 10, 0)

input_df = pd.DataFrame([[tenure, monthly_charge, total_charges, age, dependents]])

# Feature alignment
expected_features = scaler.mean_.shape[0]
while input_df.shape[1] < expected_features:
    input_df[f"dummy_{input_df.shape[1]}"] = 0

input_df = input_df.iloc[:, :expected_features]
input_scaled = scaler.transform(input_df)

# Fake prediction for demo
if st.button("🔍 Predict Churn"):
    probability = np.random.uniform(0.2, 0.8)

    st.write(f"**Churn Probability:** `{probability:.2f}`")
    if probability > 0.5:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

st.caption("⚠️ TensorFlow models are demonstrated offline due to cloud limitations")
