# Loan Default Risk Prediction System with Explainable AI

This project predicts whether a loan applicant is likely to default using a complete machine learning pipeline:

- Problem statement and business framing
- Data generation/loading
- Preprocessing and exploratory data analysis
- Four machine learning models
- Model evaluation and comparison
- Hyperparameter tuning
- Explainable AI through feature importance, permutation importance, and applicant-level reason codes
- Streamlit app for live prediction

## Project Structure

```text
.
├── app.py
├── data/
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── presentation_outline.md
├── project_report.md
├── requirements.txt
└── src/
    ├── data_generator.py
    └── train.py
```

## How To Run

1. Generate the dataset:

```bash
python src/data_generator.py
```

2. Train models and create reports/plots:

```bash
python src/train.py
```

3. Run the app:

```bash
streamlit run app.py
```

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Evaluation Metrics

Accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix.

For loan default prediction, recall is emphasized because failing to detect a risky borrower can create a direct financial loss.

