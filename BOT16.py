
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import os

# --- Streamlit App ---
st.title("Weather Forecast Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Data Preprocessing
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Handle non-numeric 'Rain' column
        if pd.api.types.is_string_dtype(df['Rain']):
            le = LabelEncoder()
            df['Rain'] = le.fit_transform(df['Rain'])

        X = df.drop('Rain', axis=1)
        y = df['Rain']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Load the saved model and scaler
        model_path = "best_model.joblib"
        scaler_path = "scaler.joblib"

        try:
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            st.error("Error: Model files not found. Train the model first and save it as 'best_model.joblib' and 'scaler.joblib'")
            st.stop()

        # Input fields (moved BEFORE the button)
        temp = st.number_input("Temperature", min_value=-20, max_value=50)
        humidity = st.number_input("Humidity", min_value=0, max_value=100)
        wind_speed = st.number_input("Wind Speed", min_value=0, max_value=50)
        cloud_cover = st.number_input("Cloud Cover", min_value=0, max_value=100)
        pressure = st.number_input("Pressure", min_value=900, max_value=1100)

        if st.button("Predict"):
            new_data = np.array([[temp, humidity, wind_speed, cloud_cover, pressure]])
            new_data_scaled = scaler.transform(new_data)
            prediction = best_model.predict(new_data_scaled)[0]
            probability = best_model.predict_proba(new_data_scaled)[0, 1]
            st.write(f"Rain Prediction: {'Rain' if prediction == 1 else 'No rain'}")
            st.write(f"Probability of Rain: {probability:.2f}")

    except FileNotFoundError:
        st.error("Error: Could not find the CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
