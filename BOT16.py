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
        df = pd.read_csv(uploaded_file)  #Read the uploaded file

        #Check for necessary columns.  Handle errors gracefully
        required_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            st.error(f"Error: Missing required columns: {missing_cols}")
            st.stop()

        # --- Data Visualization ---
        st.header("Weather Data Visualization")

        fig_temp = px.line(df, x=df.index, y='Temperature', title='Temperature')
        st.plotly_chart(fig_temp)

        fig_humidity = px.line(df, x=df.index, y='Humidity', title='Humidity')
        st.plotly_chart(fig_humidity)

        fig_wind = px.line(df, x=df.index, y='Wind_Speed', title='Wind Speed')
        st.plotly_chart(fig_wind)

        fig_cloud = px.line(df, x=df.index, y='Cloud_Cover', title='Cloud Cover')
        st.plotly_chart(fig_cloud)

        fig_pressure = px.line(df, x=df.index, y='Pressure', title='Pressure')
        st.plotly_chart(fig_pressure)


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
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Rain Prediction Accuracy: {accuracy:.2f}")

        # Example prediction (you'll need to provide new data)
        # ... (prediction code as before) ...

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
