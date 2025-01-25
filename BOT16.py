import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.title("Weather Forecast Visualization and Rain Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        #Check for necessary columns. Handle errors gracefully.
        required_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            st.error(f"Error: Missing required columns: {missing_cols}")
            st.stop()

        #Convert 'Rain' column to numeric (0 and 1) if it's not already
        if df['Rain'].dtype == object: #Check if it's string type
            rain_mapping = {'rain': 1, 'no rain': 0}
            df['Rain'] = df['Rain'].map(rain_mapping)

        # --- Data Visualization --- (same as before) ...

        # --- Rain Prediction ---
        st.header("Rain Prediction")

        X = df.drop('Rain', axis=1)
        y = df['Rain']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)

        #Threshold at 0.5 for prediction
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"Rain Prediction Accuracy (threshold 0.5): {accuracy:.2f}")
        st.write("Classification Report:\n", report)
        st.write("Confusion Matrix:\n", cm)

        # Display probabilities for each prediction
        for i in range(len(y_test)):
            rain_probability = y_pred_proba[i, 1] * 100
            st.write(f"Prediction {i+1}: Rain probability = {rain_probability:.2f}%")

        # Example prediction (you'll need to provide new data)
        # ... (prediction code as before) ...

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
