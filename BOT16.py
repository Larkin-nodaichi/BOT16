import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Streamlit Title and Instructions
st.title("Weather Forecast Prediction")
st.write("This app predicts rain based on historical weather data.")

# File uploader in Streamlit
uploaded = st.file_uploader("Upload a CSV file", type="csv")

if uploaded is not None:
    # Load the dataset
    df = pd.read_csv(uploaded)
    
    # Handle Missing Values (Imputation with mean for numerical features, mode for categorical)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)  # Mode for categorical

    # Display dataset preview
    st.write("Dataset Preview:")
    st.write(df.head())

    # Data Splitting
    X = df.drop('Rain', axis=1)
    y = df['Rain']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Displaying Histograms of the features
    st.header("Histograms of Features")
    df.hist(figsize=(10, 8))
    st.pyplot()

    # Displaying Scatter Plots
    st.header("Pairwise Plot of Features")
    sns.pairplot(df)
    st.pyplot()

    # Model Training and Comparison
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results[name] = scores.mean()

    st.write("Model Comparison (Cross-Validation Scores):")
    st.write(results)

    # Select the best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)

    # Making Predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    # ROC Curve
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    st.header("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot()

else:
    st.write("Please upload a CSV file.")

