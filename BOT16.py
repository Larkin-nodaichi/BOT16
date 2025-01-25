import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

        # Data Preprocessing ... (same as before) ...

        X = df.drop('Rain', axis=1)
        y = df['Rain']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Load the saved model and scaler using ABSOLUTE paths
        model_dir = "C:\Users\Lenovo\Downloads\weather_forecast_data.csv"  # **REPLACE THIS WITH THE ABSOLUTE PATH**
        best_model = joblib.load(os.path.join(model_dir, 'best_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

        # Input fields ... (same as before) ...

        if st.button("Predict"):
            new_data = np.array([[temp, humidity, wind_speed, cloud_cover, pressure]])
            new_data_scaled = scaler.transform(new_data)
            prediction = best_model.predict(new_data_scaled)[0]
            probability = best_model.predict_proba(new_data_scaled)[0, 1]
            st.write(f"Rain Prediction: {'Rain' if prediction == 1 else 'No rain'}")
            st.write(f"Probability of Rain: {probability:.2f}")

    except FileNotFoundError:
        st.error("Error: Could not find the saved model file. Please check the absolute path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
