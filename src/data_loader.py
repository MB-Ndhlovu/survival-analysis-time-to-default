"""
Generates synthetic loan data with survival analysis fields.
5000 observations with ~35% censored at 24 months.
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(seed=42)

N = 5000
CENSOR_TIME = 24


def generate_loan_data():
    # Credit score: 300-850 range
    credit_score = rng.integers(300, 850, size=N)

    # Income: 20k-250k, correlated with credit score
    income = (credit_score / 850) * 230000 + rng.normal(0, 10000, size=N)
    income = np.clip(income, 15000, 500000)

    # Employment years: 0-40
    employment_years = rng.exponential(5, size=N)
    employment_years = np.clip(employment_years, 0, 40)

    # Loan amount: 5k-500k
    loan_amount = rng.lognormal(10, 1.2, size=N)
    loan_amount = np.clip(loan_amount, 2000, 600000)

    # Interest rate: inversely correlated with credit score
    base_rate = 12 - (credit_score - 300) / 55 * 8
    interest_rate = base_rate + rng.normal(0, 0.5, size=N)
    interest_rate = np.clip(interest_rate, 3, 24)

    # Debt-to-income ratio
    monthly_payment = loan_amount * (interest_rate / 100 / 12)
    monthly_income = income / 12
    debt_to_income = monthly_payment / monthly_income
    debt_to_income = np.clip(debt_to_income, 0.05, 1.5)

    # LTV ratio: loan_amount / collateral_value
    collateral = income * 1.5 + 20000
    LTV_ratio = loan_amount / collateral
    LTV_ratio = np.clip(LTV_ratio, 0.1, 1.5)

    # Survival time: exponential with rate driven by covariates
    # Higher credit score -> lower hazard; higher DTI, LTV, rate -> higher hazard
    log_hazard = (
        -0.012 * (credit_score - 500)
        + 1.5 * (debt_to_income - 0.25)
        + 1.2 * (LTV_ratio - 0.5)
        + 0.08 * (interest_rate - 8)
        - 0.03 * employment_years
        + rng.normal(0, 0.4, size=N)
    )

    # Exponential survival times (months)
    survival_months = rng.exponential(scale=30.0, size=N) * np.exp(-0.3 * log_hazard)

    # ~35% censored means ~65% experience default
    event_default = (survival_months <= CENSOR_TIME).astype(int)
    time_end = np.where(
        survival_months <= CENSOR_TIME,
        np.clip(survival_months, 0.5, CENSOR_TIME),  # min 0.5 months to avoid zero
        CENSOR_TIME
    )

    df = pd.DataFrame({
        'time_start': np.zeros(N, dtype=int),
        'time_end': np.round(time_end, 2),
        'event_default': event_default,
        'income': np.round(income, 2),
        'credit_score': credit_score,
        'employment_years': np.round(employment_years, 2),
        'debt_to_income': np.round(debt_to_income, 4),
        'loan_amount': np.round(loan_amount, 2),
        'interest_rate': np.round(interest_rate, 3),
        'LTV_ratio': np.round(LTV_ratio, 4),
    })

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Default rate: {df['event_default'].mean():.1%}")
    print(f"Censored: {(df['event_default'] == 0).mean():.1%}")