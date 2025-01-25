import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st
import os

# --- Streamlit App ---
st.title("Weather Forecast Prediction")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)

        # Ensure 'Rain' column exists
        if 'Rain' not in df.columns:
            st.error("Error: The dataset must contain a 'Rain' column.")
            st.stop()

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

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save the model and scaler
        model_file = "best_model.joblib"
        scaler_file = "scaler.joblib"
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)

        st.success(f"Model and scaler trained and saved successfully as '{model_file}' and '{scaler_file}'.")

        # Predict new inputs
        st.header("Predict Weather Conditions")
        temp = st.number_input("Temperature (°C)", min_value=-20, max_value=50)
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=50)
        cloud_cover = st.number_input("Cloud Cover (%)", min_value=0, max_value=100)
        pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100)

        if st.button("Predict"):
            new_data = np.array([[temp, humidity, wind_speed, cloud_cover, pressure]])
            new_data_scaled = scaler.transform(new_data)
            prediction = model.predict(new_data_scaled)[0]
            probability = model.predict_proba(new_data_scaled)[0, 1]

            st.write(f"Rain Prediction: {'Rain' if prediction == 1 else 'No Rain'}")
            st.write(f"Probability of Rain: {probability:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("Please upload a CSV file.")

