# Viva Questions and Strong Answers

## 1. Why did you choose loan default prediction?

Loan default prediction is a real-world financial risk problem. Banks need to identify risky applicants before approving loans, so this is a practical use case of classification in machine learning.

## 2. What type of ML problem is this?

It is a supervised binary classification problem because the target has two classes: default and no default.

## 3. Why is recall important here?

Recall is important because it measures how many actual defaulters the model catches. In lending, missing a defaulter can cause direct financial loss, so recall is more business-critical than plain accuracy.

## 4. Why did you use multiple models?

Multiple models help compare performance and behavior. Logistic Regression gives a baseline, Decision Tree is interpretable, and Random Forest/Gradient Boosting can capture more complex patterns.

## 5. What is the role of preprocessing?

Preprocessing handles missing values, scales numerical features, and converts categorical variables into numeric form using one-hot encoding. Without preprocessing, many ML algorithms cannot learn properly.

## 6. What is data leakage?

Data leakage happens when information from the test set or future data is accidentally used during training. I avoided this by using a scikit-learn Pipeline, where preprocessing is learned only from training data.

## 7. Why use Random Forest?

Random Forest combines many decision trees, which improves stability and reduces overfitting compared with a single tree. It also provides feature importance, which helps explain the model.

## 8. What is hyperparameter tuning?

Hyperparameter tuning means searching for the best model settings, such as number of trees and maximum depth in Random Forest. I used GridSearchCV with cross-validation.

## 9. What is explainable AI in this project?

Explainable AI means making the prediction understandable. I used feature importance, permutation importance, and applicant-level reason codes to explain why a prediction is high risk or low risk.

## 10. Which features are most important?

The most important features are usually credit score, debt-to-income ratio, previous defaults, interest rate, loan amount, and annual income. The exact ranking is available in the feature importance output.

