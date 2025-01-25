import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.title("Weather Forecast Visualization and Rain Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        #Check for necessary columns.  Handle errors gracefully
        required_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            st.error(f"Error: Missing required columns: {missing_cols}")
            st.stop()

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
        y_pred_proba = model.predict_proba(X_test_scaled) #Get probabilities

        #Calculate accuracy using predicted probabilities (threshold of 0.5)
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int) #Threshold at 0.5
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Rain Prediction Accuracy (threshold 0.5): {accuracy:.2f}")


        # Display probabilities for each prediction
        for i in range(len(y_test)):
            rain_probability = y_pred_proba[i, 1] * 100
            st.write(f"Prediction {i+1}: Rain probability = {rain_probability:.2f}%")


        # Example prediction (you'll need to provide new data)
        new_weather = pd.DataFrame({
            'Temperature': [26],
            'Humidity': [72],
            'Wind_Speed': [10],
            'Cloud_Cover': [40],
            'Pressure': [1011]
        })
        new_weather_scaled = scaler.transform(new_weather)
        new_prediction_proba = model.predict_proba(new_weather_scaled)[0,1] * 100
        st.write(f"New weather data: Rain probability = {new_prediction_proba:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
