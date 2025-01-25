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

        # --- Choose ONE mapping and uncomment it ---
        # Make sure the keys in the mapping match the values in your 'Attack_Vector' column!

        attack_mapping = {  # Mapping 1 (Example)
            'email': 'Phishing',
            'malware_download': 'Malware',
            'dos_attack': 'DoS',
            'ransomware_attack': 'Ransomware',
            'unknown': 'Unknown'
        }

        # attack_mapping = {  # Mapping 2 (Example)
        #     'phishing_email': 'Phishing',
        #     'malware_infection': 'Malware',
        #     'dos_attack': 'DoS',
        #     'ransomware': 'Ransomware',
        #     'data_breach': 'Data Breach',
        #     'other': 'Other'
        # }

        if 'Attack_Vector' in df.columns:
            df['Incident_Type'] = df['Attack_Vector'].map(attack_mapping).fillna('Unknown')
        else:
            st.error("Error: 'Attack_Vector' column not found. Check your data or choose a different column.")
            st.stop()

        # ... (rest of your preprocessing and model training code) ...

    except Exception as e:
        st.error(f"An error occurred: {e}")
