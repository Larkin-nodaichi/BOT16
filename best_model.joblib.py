import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("weather_forecast_data.csv")

# Preprocess the data
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

if pd.api.types.is_string_dtype(df['Rain']):
    le = LabelEncoder()
    df['Rain'] = le.fit_transform(df['Rain'])

X = df.drop('Rain', axis=1)
y = df['Rain']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully!")
