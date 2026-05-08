# Loan Default Risk Prediction System with Explainable AI

## 1. Introduction

Loan default prediction is an important machine learning problem in financial risk management. Banks and lending companies need to estimate whether an applicant is likely to repay a loan or default. A wrong approval can create direct financial loss, while a wrong rejection can reduce customer trust and business opportunity.

This project builds a supervised machine learning system that predicts loan default risk from applicant, financial, and credit-related attributes. The system also includes explainability so that the model decision is not treated as a black box.

## 2. Problem Statement

The objective is to predict whether a loan applicant will default based on features such as income, credit score, debt-to-income ratio, previous defaults, employment history, loan amount, loan purpose, and home ownership.

This is a binary classification problem:

- `0`: No default
- `1`: Default

## 3. Dataset Description

The project uses a realistic loan risk dataset generated with financial-domain logic. It contains 5,000 applicant records with both numerical and categorical features.

Important features include:

- Annual income
- Credit score
- Previous defaults
- Loan amount
- Loan term
- Monthly debt
- Debt-to-income ratio
- Interest rate
- Employment years
- Home ownership
- Education
- Loan purpose

Missing values were intentionally introduced in selected columns to demonstrate real preprocessing.

## 4. Data Preprocessing

The preprocessing pipeline includes:

- Median imputation for numerical missing values
- Most frequent imputation for categorical missing values
- Standard scaling for numerical variables
- One-hot encoding for categorical variables
- Stratified train-test split to preserve class distribution

Using a scikit-learn `Pipeline` ensures preprocessing is applied consistently during training and prediction.

## 5. Exploratory Data Analysis

EDA focused on understanding the relationship between applicant profile and default risk.

Key analysis:

- Target class distribution
- Correlation heatmap
- Credit score comparison by default status
- Debt-to-income ratio comparison by default status
- Default rate across loan purposes

Expected observations:

- Lower credit score increases default risk.
- Higher debt-to-income ratio increases default risk.
- Previous default history is a strong negative indicator.
- Loan amount relative to income impacts repayment risk.

## 6. Models Implemented

Four models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

Logistic Regression was used as a baseline model. Decision Tree was used for interpretability. Random Forest and Gradient Boosting were used for stronger predictive performance.

## 7. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Recall is especially important in this project because missing an actual defaulter is costly for a lender. Therefore, the final model selection gives strong importance to recall along with F1-score.

## 8. Hyperparameter Tuning

Random Forest was tuned using GridSearchCV with 5-fold cross-validation.

Tuned parameters:

- Number of trees
- Maximum tree depth
- Minimum samples split
- Minimum samples leaf

The tuning objective was recall because the business goal is to detect risky borrowers.

## 9. Explainable AI

The final system explains model behavior using:

- Random Forest feature importance
- Permutation importance based on recall
- Applicant-level reason codes in the Streamlit app

Example explanation:

If an applicant is marked high risk, the app may show reasons such as low credit score, high debt-to-income ratio, previous defaults, or loan amount being high compared with income.

## 10. Final System

The final deliverable includes:

- Trained ML model
- Evaluation reports
- EDA and model plots
- Feature importance reports
- Streamlit prediction app

The app accepts applicant details and returns:

- Default probability
- High risk or low risk prediction
- Main explanation for the decision
- Top global model features

## 11. Conclusion

The project successfully demonstrates the complete machine learning pipeline for a real-world financial risk problem. By combining model comparison, hyperparameter tuning, and explainability, the system is more useful than a simple prediction model. It supports both accurate risk prediction and human-understandable decision reasoning.

