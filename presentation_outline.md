# Presentation Outline

## Slide 1: Title

Loan Default Risk Prediction System with Explainable AI  
CSE247 Applied Machine Learning

## Slide 2: Problem Statement

Predict whether a loan applicant is likely to default based on financial and personal data.

## Slide 3: Why This Problem Matters

- Banks lose money when risky borrowers are approved.
- Manual risk decisions can be slow and inconsistent.
- ML can support faster and more data-driven decisions.

## Slide 4: Dataset

- 5,000 applicant records
- Numerical and categorical features
- Target variable: defaulted or not defaulted
- Features include credit score, income, loan amount, debt-to-income ratio, employment years, and previous defaults

## Slide 5: Preprocessing

- Missing value handling
- One-hot encoding
- Feature scaling
- Train-test split
- Pipeline-based implementation

## Slide 6: EDA

Show:

- Target distribution
- Correlation heatmap
- Credit score vs default
- Debt-to-income ratio vs default

## Slide 7: Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Slide 8: Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Main point: Recall is important because missing a defaulter is costly.

## Slide 9: Hyperparameter Tuning

Random Forest tuned using GridSearchCV.

Parameters:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

## Slide 10: Results

Show `outputs/reports/model_comparison.csv`.

Explain which model performed best and why.

## Slide 11: Explainable AI

Show:

- Feature importance plot
- Permutation importance plot
- Example applicant-level explanation

## Slide 12: Streamlit App

Demo:

- Enter applicant details
- Get default probability
- See high risk or low risk result
- Read explanation

## Slide 13: Conclusion

- Built a complete ML pipeline
- Compared multiple models
- Tuned final model
- Added explainability
- Created a usable prediction app

## Slide 14: Viva Points

- Why recall matters more than accuracy
- Why pipeline prevents data leakage
- Why Random Forest performed well
- How feature importance improves trust
