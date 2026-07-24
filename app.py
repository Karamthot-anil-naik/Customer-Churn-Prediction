import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Platform")

st.write(
    "Predict customer churn using Machine Learning."
)
model = joblib.load("customer_churn_model.pkl")

scaler = joblib.load("scaler.pkl")

feature_columns = joblib.load("feature_columns.pkl")

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)
if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Preprocessing
    data["TotalCharges"] = data["TotalCharges"].replace(" ", np.nan)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])
    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )

    # Remove customerID if present
    if "customerID" in data.columns:
        data.drop("customerID", axis=1, inplace=True)

    # One-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Match training columns
    data = data.reindex(columns=feature_columns, fill_value=0)

    # Scale
    scaled_data = scaler.transform(data)

    # Predict
    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)[:, 1]

    # Results
    results = data.copy()

    results["Prediction"] = prediction

    results["Probability"] = probability

    results["Prediction"] = results["Prediction"].map({
        0: "No",
        1: "Yes"
    })

    st.subheader("Prediction Results")
    st.dataframe(results)

    csv = results.to_csv(index=False)

    st.download_button(
        "Download Results",
        csv,
        "customer_predictions.csv",
        "text/csv"
    )## 



st.sidebar.title("About")

st.sidebar.info("""
Customer Churn Prediction Platform

Developed using

• Python

• Scikit-learn

• Streamlit

• Machine Learning
""")

