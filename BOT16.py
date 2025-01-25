import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data_path = "C:\\Users\\Lenovo\\Downloads\\weather_forecast_data.csv"  # Replace with your dataset path
df = pd.read_csv(data_path)

# Ensure 'Rain' column exists
if 'Rain' not in df.columns:
    raise ValueError("The dataset must contain a 'Rain' column.")

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
joblib.dump(model, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully.")

