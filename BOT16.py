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

# --- Streamlit App ---
st.title("Weather Forecast Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Data Preprocessing (same as before, but adapt to handle potential errors)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        if df['Rain'].dtype == object:
            df['Rain'] = df['Rain'].str.lower().replace({'rain': 1, 'no rain': 0})
            df['Rain'] = pd.to_numeric(df['Rain'])

        X = df.drop('Rain', axis=1)
        y = df['Rain']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Model training (same as before, but you'll need to load the model instead of training it here)
        best_model = joblib.load('best_model.joblib') #Load the saved model
        scaler = joblib.load('scaler.joblib') #Load the saved scaler


        # Input fields
        temp = st.number_input("Temperature", min_value=-20, max_value=50)
        humidity = st.number_input("Humidity", min_value=0, max_value=100)
        wind_speed = st.number_input("Wind Speed", min_value=0, max_value=50)
        cloud_cover = st.number_input("Cloud Cover", min_value=0, max_value=100)
        pressure = st.number_input("Pressure", min_value=900, max_value=1100)

        if st.button("Predict"):
            new_data = np.array([[temp, humidity, wind_speed, cloud_cover, pressure]])
            new_data_scaled = scaler.transform(new_data)
            prediction = best_model.predict(new_data_scaled)[0]
            probability = best_model.predict_proba(new_data_scaled)[0, 1]
            st.write(f"Rain Prediction: {'Rain' if prediction == 1 else 'No rain'}")
            st.write(f"Probability of Rain: {probability:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")
#Split data
X = df.drop('Rain', axis=1)
y = df['Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Data Visualization (same as before) ---

# --- Model Training ---
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = scores.mean()

print(results)
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)

#Save the model and scaler
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')


# --- Model Evaluation ---
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#ROC Curve
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# --- Streamlit App ---
import streamlit as st

st.title("Weather Forecast Prediction")

# Load the saved model and scaler
best_model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Input fields
temp = st.number_input("Temperature", min_value=-20, max_value=50)
humidity = st.number_input("Humidity", min_value=0, max_value=100)
wind_speed = st.number_input("Wind Speed", min_value=0, max_value=50)
cloud_cover = st.number_input("Cloud Cover", min_value=0, max_value=100)
pressure = st.number_input("Pressure", min_value=900, max_value=1100)

if st.button("Predict"):
    new_data = np.array([[temp, humidity, wind_speed, cloud_cover, pressure]])
    new_data_scaled = scaler.transform(new_data)
    prediction = best_model.predict(new_data_scaled)[0]
    probability = best_model.predict_proba(new_data_scaled)[0, 1]
    st.write(f"Rain Prediction: {'Rain' if prediction == 1 else 'No rain'}")
    st.write(f"Probability of Rain: {probability:.2f}")
