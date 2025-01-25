import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Cybersecurity Incident Prediction App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')

        # --- Create Incident_Type column ---
        attack_mapping = {
            'email': 'Phishing',
            'malware_download': 'Malware',
            'dos_attack': 'DoS',
            'ransomware_attack': 'Ransomware',
            # Add more mappings as needed based on your data
            'unknown': 'Unknown' # Handle cases where the attack vector is not clear
        }
        df['Incident_Type'] = df['Attack_Vector'].map(attack_mapping).fillna('Unknown')


        # Preprocessing and Feature Engineering
        df.fillna(method='ffill', inplace=True)  # Fill remaining NaN values

        numerical_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        X = df.drop('Incident_Type', axis=1)  #Incident_Type is now a feature!
        y = df['Incident_Type']

        X_processed = preprocessor.fit_transform(X)

        # Model Training and Evaluation (rest of your code remains the same)
        # ... (your existing model training and evaluation code) ...

    except Exception as e:
        st.error(f"An error occurred: {e}")
