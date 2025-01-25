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

        #Check if 'Attack Type' column exists; if not, handle the error gracefully
        if 'Attack Type' not in df.columns:
            st.error("Error: 'Attack Type' column not found in the dataset. Please check your data.")
            st.stop()

        #Improved mapping with better error handling
        attack_mapping = {
            'phishing': 'Phishing',
            'malware': 'Malware',
            'dos': 'DoS',
            'ransomware': 'Ransomware',
            'sql injection': 'SQL Injection',
            'brute force': 'Brute Force',
            'unknown': 'Unknown'
        }

        #Create Incident_Type column, handling potential errors
        try:
            df['Incident_Type'] = df['Attack Type'].map(attack_mapping).fillna('Unknown')
        except KeyError as e:
            st.error(f"Error: Key '{e}' not found in attack_mapping. Check your 'Attack Type' column values.")
            st.stop()

        # Preprocessing and Feature Engineering
        df.fillna(method='ffill', inplace=True)

        numerical_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        X = df.drop('Incident_Type', axis=1)
        y = df['Incident_Type']

        X_processed = preprocessor.fit_transform(X)

        # Model Training and Evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("Classification Report:\n", report)
        st.write("Confusion Matrix:\n", cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
