import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_default_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("Model files not found. Run `python src/data_generator.py` and `python src/train.py` first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


def build_reason_codes(applicant, probability):
    reasons = []
    if applicant["credit_score"] < 620:
        reasons.append("Low credit score increases default risk.")
    if applicant["debt_to_income"] > 0.40:
        reasons.append("Debt-to-income ratio is high.")
    if applicant["previous_defaults"] > 0:
        reasons.append("Previous default history is a strong warning signal.")
    if applicant["loan_amount"] / max(applicant["annual_income"], 1) > 0.75:
        reasons.append("Loan amount is large compared with annual income.")
    if applicant["employment_years"] < 2:
        reasons.append("Short employment history reduces repayment confidence.")
    if applicant["interest_rate"] > 18:
        reasons.append("High interest rate is associated with riskier loan profiles.")
    if not reasons and probability < 0.35:
        reasons.append("Strong credit score, stable income, and manageable debt indicate lower risk.")
    return reasons[:4]


st.set_page_config(page_title="Loan Default Risk Prediction", page_icon=":credit_card:", layout="wide")

model, metadata = load_artifacts()
threshold = metadata["business_threshold"]

st.title("Loan Default Risk Prediction System")
st.caption("Explainable AI model for estimating loan applicant default risk")

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Applicant Details")
    col1, col2 = st.columns(2)
    with col1:
        applicant_age = st.slider("Applicant age", 21, 68, 36)
        annual_income = st.number_input("Annual income", min_value=12000, max_value=240000, value=52000, step=1000)
        employment_years = st.slider("Employment years", 0.0, 35.0, 5.0, 0.5)
        credit_history_years = st.slider("Credit history years", 1.0, 35.0, 8.0, 0.5)
        credit_score = st.slider("Credit score", 300, 850, 660)
        previous_defaults = st.slider("Previous defaults", 0, 4, 0)
    with col2:
        loan_amount = st.number_input("Loan amount", min_value=3000, max_value=180000, value=28000, step=1000)
        loan_term_months = st.selectbox("Loan term months", [12, 24, 36, 48, 60, 72], index=2)
        monthly_debt = st.number_input("Monthly debt", min_value=0, max_value=7500, value=850, step=50)
        debt_to_income = monthly_debt / (annual_income / 12)
        interest_rate = st.slider("Interest rate (%)", 5.5, 29.0, 13.5, 0.1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate", "Post Graduate"])
        home_ownership = st.selectbox("Home ownership", ["Rent", "Own", "Mortgage"])
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["Debt Consolidation", "Business", "Education", "Home Improvement", "Medical", "Vehicle"],
        )

applicant = {
    "applicant_age": applicant_age,
    "education": education,
    "home_ownership": home_ownership,
    "annual_income": annual_income,
    "employment_years": employment_years,
    "credit_history_years": credit_history_years,
    "credit_score": credit_score,
    "previous_defaults": previous_defaults,
    "loan_purpose": loan_purpose,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "monthly_debt": monthly_debt,
    "debt_to_income": round(debt_to_income, 3),
    "interest_rate": interest_rate,
}

input_df = pd.DataFrame([applicant])
probability = float(model.predict_proba(input_df)[:, 1][0])
prediction = int(probability >= threshold)

with right:
    st.subheader("Risk Output")
    st.metric("Default Probability", f"{probability:.1%}")
    st.metric("Decision Threshold", f"{threshold:.0%}")

    if prediction == 1:
        st.error("High Risk: Applicant is likely to default")
    else:
        st.success("Low Risk: Applicant is less likely to default")

    st.write("Key Explanation")
    for reason in build_reason_codes(applicant, probability):
        st.write(f"- {reason}")

    st.write("Most Important Model Features")
    top_features = pd.DataFrame(metadata["top_features"])
    st.dataframe(top_features, hide_index=True, use_container_width=True)

st.divider()
st.write("Applicant record used by the model")
st.dataframe(input_df, hide_index=True, use_container_width=True)
