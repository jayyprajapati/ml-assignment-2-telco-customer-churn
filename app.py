import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Telco Churn ML App", layout="wide")

st.title("Telco Customer Churn Prediction")
st.write("Upload test dataset (CSV format) to generate predictions using selected model.")

# Load saved objects
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("saved_models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("saved_models/decision_tree.pkl"),
        "KNN": joblib.load("saved_models/knn.pkl"),
        "Naive Bayes": joblib.load("saved_models/naive_bayes.pkl"),
        "Random Forest": joblib.load("saved_models/random_forest.pkl"),
        "XGBoost": joblib.load("saved_models/xgboost.pkl")
    }
    scaler = joblib.load("saved_models/scaler.pkl")
    feature_columns = joblib.load("saved_models/feature_columns.pkl")
    return models, scaler, feature_columns

models, scaler, feature_columns = load_models()

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

# Model selection
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    # Check if target exists
    if "Churn" in data.columns:
        if data["Churn"].dtype == object:
            y_true = data["Churn"].map({"No": 0, "Yes": 1})
        else:
            y_true = data["Churn"]
        X_input = data.drop("Churn", axis=1)

    else:
        y_true = None
        X_input = data

    # Convert TotalCharges if present
    if "TotalCharges" in X_input.columns:
        X_input["TotalCharges"] = pd.to_numeric(X_input["TotalCharges"], errors="coerce")
        X_input = X_input.fillna(0)

    # One-hot encode
    X_input = pd.get_dummies(X_input, drop_first=True)

    # Align columns with training
    for col in feature_columns:
        if col not in X_input.columns:
            X_input[col] = 0

    X_input = X_input[feature_columns]

    # Scale for required models
    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_processed = scaler.transform(X_input)
    else:
        X_processed = X_input

    # Predict
    y_pred = selected_model.predict(X_processed)
    y_prob = selected_model.predict_proba(X_processed)[:, 1]

    st.subheader("Predictions")
    st.write(pd.DataFrame({"Prediction": y_pred}))

    if y_true is not None:
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"],
            "Value": [accuracy, precision, recall, f1, auc, mcc]
        })

        st.table(metrics_df)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
