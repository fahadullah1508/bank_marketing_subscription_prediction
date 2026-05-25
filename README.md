# Task 1: Term Deposit Subscription Prediction

## Overview
Predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign using the Bank Marketing Dataset from UCI Machine Learning Repository.

## Objective
Build classification models to predict term deposit subscriptions and use Explainable AI (SHAP/LIME) to interpret model predictions.

## Dataset
- **Source:** UCI Machine Learning Repository - Bank Marketing Dataset
- **Samples:** 45,211 bank customers
- **Features:** Age, job, marital status, education, balance, housing loan, personal loan, contact type, campaign details, previous outcomes
- **Target:** Whether the customer subscribed to a term deposit (yes/no)

## Approach
1. **Data Exploration:** Analyzed target distribution, feature relationships, and correlation patterns
2. **Data Preprocessing:** Handled 'unknown' values, encoded categorical variables, scaled numerical features
3. **Model Building:** Trained Logistic Regression and Random Forest classifiers with class balancing
4. **Evaluation:** Used Confusion Matrix, F1-Score, and ROC Curve for model assessment
5. **Explainable AI:** Applied SHAP for global and local feature importance, LIME for individual prediction explanations

## Models Used
- Logistic Regression (with class_weight='balanced')
- Random Forest Classifier (with class_weight='balanced')

## Key Results
- Random Forest achieved higher ROC-AUC and F1-Score
- Call duration (duration) is the most important predictive feature
- SHAP analysis revealed clear patterns in customer behavior

## Files
- `Task1_Term_Deposit_Prediction.ipynb` - Main analysis notebook

## Requirements
```
pandas, numpy, matplotlib, seaborn, scikit-learn, shap, lime
```

## How to Run
1. Open the Jupyter notebook
2. Run all cells sequentially
3. Visualizations and results will be displayed inline

## Key Insights
- Longer call durations significantly increase subscription likelihood
- Customers with higher account balances are more likely to subscribe
- Previous campaign success is a strong predictor
- Age and campaign frequency also influence outcomes
