from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_STATE = 42


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def generate_loan_data(n_rows=5000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    applicant_age = np.clip(rng.normal(38, 11, n_rows).round(), 21, 68).astype(int)
    education = rng.choice(
        ["Graduate", "Not Graduate", "Post Graduate"],
        size=n_rows,
        p=[0.48, 0.37, 0.15],
    )
    home_ownership = rng.choice(
        ["Rent", "Own", "Mortgage"],
        size=n_rows,
        p=[0.42, 0.24, 0.34],
    )
    loan_purpose = rng.choice(
        ["Debt Consolidation", "Business", "Education", "Home Improvement", "Medical", "Vehicle"],
        size=n_rows,
        p=[0.30, 0.18, 0.13, 0.16, 0.11, 0.12],
    )

    employment_years = np.clip(rng.gamma(3.0, 2.2, n_rows), 0, 35).round(1)
    annual_income = rng.lognormal(mean=10.95, sigma=0.45, size=n_rows)
    annual_income *= np.where(education == "Post Graduate", 1.28, 1.0)
    annual_income *= np.where(education == "Not Graduate", 0.82, 1.0)
    annual_income = np.clip(annual_income, 12000, 240000).round(0)

    credit_history_years = np.clip((applicant_age - 20) * rng.uniform(0.18, 0.82, n_rows), 1, 35).round(1)
    previous_defaults = rng.poisson(
        lam=np.where(credit_history_years < 4, 0.35, 0.16) + np.where(annual_income < 35000, 0.18, 0),
        size=n_rows,
    )
    previous_defaults = np.clip(previous_defaults, 0, 4)

    base_score = (
        650
        + (annual_income - annual_income.mean()) / annual_income.std() * 38
        + employment_years * 3.2
        + credit_history_years * 2.5
        - previous_defaults * 72
        + rng.normal(0, 42, n_rows)
    )
    credit_score = np.clip(base_score, 300, 850).round().astype(int)

    loan_amount = rng.lognormal(mean=10.25, sigma=0.55, size=n_rows)
    loan_amount *= np.where(loan_purpose == "Business", 1.35, 1.0)
    loan_amount *= np.where(loan_purpose == "Medical", 0.82, 1.0)
    loan_amount = np.clip(loan_amount, 3000, 180000).round(0)

    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72], size=n_rows, p=[0.07, 0.15, 0.30, 0.18, 0.22, 0.08])
    monthly_debt = np.clip(
        annual_income / 12 * rng.uniform(0.06, 0.48, n_rows) + rng.normal(0, 170, n_rows),
        0,
        7500,
    ).round(0)
    debt_to_income = monthly_debt / (annual_income / 12)

    interest_rate = (
        8.5
        + (720 - credit_score) * 0.035
        + debt_to_income * 5
        + previous_defaults * 1.8
        + rng.normal(0, 1.1, n_rows)
    )
    interest_rate = np.clip(interest_rate, 5.5, 29.0).round(2)

    logit = (
        -2.15
        + (680 - credit_score) / 85
        + debt_to_income * 2.15
        + loan_amount / np.maximum(annual_income, 1) * 0.95
        + previous_defaults * 0.85
        - employment_years * 0.045
        - credit_history_years * 0.035
        + np.where(home_ownership == "Rent", 0.28, 0)
        + np.where(loan_purpose == "Business", 0.18, 0)
        + np.where(loan_purpose == "Medical", 0.22, 0)
        + rng.normal(0, 0.45, n_rows)
    )
    default_probability = sigmoid(logit)
    defaulted = rng.binomial(1, default_probability)

    data = pd.DataFrame(
        {
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
            "debt_to_income": debt_to_income.round(3),
            "interest_rate": interest_rate,
            "defaulted": defaulted,
        }
    )

    for column in ["annual_income", "employment_years", "credit_score", "loan_amount", "home_ownership"]:
        missing_idx = rng.choice(data.index, size=int(n_rows * 0.025), replace=False)
        data.loc[missing_idx, column] = np.nan

    return data


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_loan_data()
    output_path = DATA_DIR / "loan_default_risk_dataset.csv"
    data.to_csv(output_path, index=False)
    print(f"Created dataset: {output_path}")
    print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
    print(f"Default rate: {data['defaulted'].mean():.2%}")


if __name__ == "__main__":
    main()
