import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from google.colab import files
uploaded = files.upload()  # Upload `mushrooms.csv` here
df = pd.read_csv('weather_forecast_data.csv')
df = pd.DataFrame(data)


#Handle Missing Values (Example: Imputation with mean for numerical features)
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True) #Mode for categorical
..

#Split data into features (X) and target (y)
X = df.drop('Rain', axis=1)
y = df['Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

#Histograms
df.hist(figsize=(10, 8))
plt.show()

#Scatter Plots (Example)
sns.pairplot(df)
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#ROC Curve (if applicable for your target variable)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
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





import streamlit as st

st.title("Weather Forecast Prediction")

# Input fields for new weather data
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
