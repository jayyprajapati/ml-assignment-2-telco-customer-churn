## STUDENT INFORMATION

### BITS ID: 2025AA05018

### Name: Prajapati Jay Hiteshkumar

### Email: 2025aa05018@wilp.bits-pilani.ac.in

### Date: 8th Feb, 2026

# Telco Customer Churn Prediction using Machine Learning

## 1. Problem Statement

Customer churn refers to customers discontinuing their service with a company. In the telecommunications industry, customer retention is critical because acquiring new customers is significantly more expensive than retaining existing ones.

The objective of this project is to build and compare multiple machine learning classification models to predict whether a customer will churn based on demographic, service usage, and billing information.

This is a binary classification problem where:
- 0 → Customer does not churn
- 1 → Customer churns


---

## 2. Dataset Description

The Telco Customer Churn dataset contains information about over 7,000 customers of a telecommunications company.

After preprocessing:
- Total instances: 7,032
- Total features after encoding: 30
- Target variable: `Churn`

### Feature Categories

1. Demographic Features:
   - Gender
   - SeniorCitizen
   - Partner
   - Dependents

2. Account Information:
   - Tenure
   - Contract type
   - Payment method
   - Paperless billing

3. Service Information:
   - Internet service
   - Phone service
   - Streaming services
   - Online security
   - Tech support

4. Billing Information:
   - MonthlyCharges
   - TotalCharges

Categorical features were one-hot encoded.  
Missing values in `TotalCharges` were removed.  
The dataset was split into 80% training and 20% testing using stratified sampling.


---

## 3. Machine Learning Models Implemented

The following classification models were implemented:

1. Logistic Regression  
2. Decision Tree (Tuned)  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Tuned)  
6. XGBoost  

Light hyperparameter tuning was applied to Decision Tree and Random Forest models to improve generalization and reduce overfitting.


---

## 4. Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

MCC is particularly useful for imbalanced classification problems as it considers all four confusion matrix components.


---

## 5. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|------|-----------|--------|----------|------|
| Random Forest (Tuned) | 0.7448 | 0.8381 | 0.5131 | 0.7834 | 0.6201 | 0.4627 |
| Logistic Regression | 0.8038 | 0.8357 | 0.6476 | 0.5749 | 0.6091 | 0.4803 |
| XGBoost | 0.7832 | 0.8341 | 0.6075 | 0.5214 | 0.5612 | 0.4205 |
| Decision Tree (Tuned) | 0.7783 | 0.8197 | 0.5807 | 0.5963 | 0.5884 | 0.4367 |
| Naive Bayes | 0.6446 | 0.8102 | 0.4184 | 0.8636 | 0.5637 | 0.3808 |
| KNN | 0.7534 | 0.7666 | 0.5362 | 0.5348 | 0.5355 | 0.3676 |


---

## 6. Observations and Analysis

### Logistic Regression
Logistic Regression achieved strong overall performance with high AUC and the highest MCC, indicating balanced predictive capability. The dataset exhibits significant linear separability.

### Decision Tree (Tuned)
The untuned decision tree overfit heavily. After controlling depth and minimum sample constraints, generalization improved substantially. However, a single tree remains less robust than ensemble methods.

### KNN
KNN achieved moderate performance but did not outperform linear or ensemble models, suggesting churn patterns are not strongly defined by local distance clusters.

### Naive Bayes
Naive Bayes achieved very high recall but low precision, aggressively predicting churn. While it captures most churners, it generates many false positives.

### Random Forest (Tuned)
The tuned Random Forest achieved the highest AUC and strongest F1 score. Its high recall makes it particularly suitable for churn detection scenarios where identifying at-risk customers is critical.

### XGBoost
XGBoost performed competitively but did not significantly outperform Logistic Regression or the tuned Random Forest, indicating that extreme boosting complexity is not required for this dataset.


---

## 7. Streamlit Application

The project includes an interactive Streamlit web application that allows:

- Uploading test CSV files
- Selecting a trained model
- Generating predictions
- Viewing evaluation metrics
- Displaying confusion matrix visualization

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
