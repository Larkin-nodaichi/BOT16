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


        # Load the saved model and scaler (using relative paths)
        best_model = joblib.load('best_model.joblib')
        scaler = joblib.load('scaler.joblib')

        # Input fields ... (same as before) ...

        if st.button("Predict"):
            #Prediction code (same as before)...

    except FileNotFoundError:
        st.error("Error: Could not find the saved model file. Please ensure 'best_model.joblib' and 'scaler.joblib' are in the same directory as your Streamlit script.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
