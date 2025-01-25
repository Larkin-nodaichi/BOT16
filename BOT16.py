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

# --- Streamlit App ---
st.title("Weather Forecast Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Data Preprocessing (same as before, but adapt to handle potential errors)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        if df['Rain'].dtype == object:
            df['Rain'] = df['Rain'].str.lower().replace({'rain': 1, 'no rain': 0})
            df['Rain'] = pd.to_numeric(df['Rain'])

        X = df.drop('Rain', axis=1)
        y = df['Rain']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Model training (same as before, but you'll need to load the model instead of training it here)
        best_model = joblib.load('best_model.joblib') #Load the saved model
        scaler = joblib.load('scaler.joblib') #Load the saved scaler


        # Input fields
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

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
