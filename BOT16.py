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

        #Check if 'Attack Type' column exists (case-insensitive)
        if 'attack type' not in [col.lower() for col in df.columns]:
            st.error("Error: 'Attack Type' column not found in the dataset. Please check your data.")
            st.stop()

        # Find the closest matching column name (case-insensitive)
        attack_type_column = next((col for col in df.columns if col.lower() == 'attack type'), None)
        if attack_type_column is None:
            st.error("Error: Could not find a column matching 'Attack Type'. Please check your data.")
            st.stop()


        attack_mapping = {
            'phishing': 'Phishing',
            'malware': 'Malware',
            'dos': 'DoS',
            'ransomware': 'Ransomware',
            'sql injection': 'SQL Injection',
            'brute force': 'Brute Force',
            'unknown': 'Unknown'
        }


        try:
            df['Incident_Type'] = df[attack_type_column].map(attack_mapping).fillna('Unknown')
        except KeyError as e:
            st.error(f"Error: Key '{e}' not found in attack_mapping. Check your 'Attack Type' column values.")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during mapping: {e}")
            st.stop()

        # ... (rest of your preprocessing and model training code) ...

    except Exception as e:
        st.error(f"An error occurred: {e}")
        unique_attack_types = df['Attack Type'].unique()
st.write(unique_attack_types) #This will display the unique values in your Streamlit app
