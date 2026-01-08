import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload transaction CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    data["Fraud Prediction"] = predictions
    st.write(data)
