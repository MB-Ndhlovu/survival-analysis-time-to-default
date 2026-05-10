"""
Generate synthetic loan data for time-to-default survival analysis.
Creates 5000 loan records with realistic covariates and censored observations.
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(42)


def generate_loan_data(n=5000, censor_at_month=24):
    """
    Generate synthetic loan dataset with time-to-default survival data.

    Parameters
    ----------
    n : int
        Number of loan records to generate
    censor_at_month : int
        Observation window end — loans not defaulted by this month are censored

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time_start: 0 (origin month)
        - time_end: observed survival time in months (or censor time if censored)
        - event_default: 1 if defaulted, 0 if censored
        - income: annual income in ZAR
        - credit_score: numeric credit score (300-850)
        - employment_years: years employed
        - debt_to_income: ratio of monthly debt payment to gross income
        - loan_amount: loan principal in ZAR
        - interest_rate: annual interest rate (decimal)
        - LTV_ratio: loan-to-value ratio at origination
    """
    # Credit score distribution (realistic, skewed toward higher scores)
    raw_scores = rng.normal(650, 120, size=n)
    credit_score = np.round(np.clip(raw_scores, 300, 850)).astype(int)

    # Income (ZAR, log-normal distribution)
    income = rng.lognormal(mean=11.5, sigma=0.5, size=n)
    income = np.round(income / 1000) * 1000

    # Employment years (exponential-like, many early-career)
    employment_years = np.clip(rng.exponential(scale=5, size=n), 0, 40)

    # Debt-to-income ratio (0.10 to 0.55)
    debt_to_income = rng.uniform(0.10, 0.55, size=n)

    # Loan amount (correlated with income)
    loan_amount = (income * rng.uniform(0.4, 2.0, size=n))
    loan_amount = np.round(loan_amount / 5000) * 5000
    loan_amount = np.clip(loan_amount, 10000, 5_000_000)

    # Interest rate (credit score driven, inverse relationship)
    base_rate = 0.16
    score_effect = (credit_score - 600) / 200
    interest_rate = base_rate - score_effect + rng.normal(0, 0.015, n)
    interest_rate = np.clip(interest_rate, 0.05, 0.30)

    # LTV ratio
    collateral_value = income * rng.uniform(1.5, 4.0, size=n)
    LTV_ratio = loan_amount / collateral_value
    LTV_ratio = np.clip(LTV_ratio, 0.15, 1.10)

    # Generate survival time using a model that produces defaults within 24 months
    # for a substantial portion of borrowers
    # Higher risk factors → shorter survival (more defaults)
    risk_score = (
        (700 - credit_score) / 150           # lower score → higher risk
        + 2.0 * (debt_to_income - 0.20)      # higher DTI → higher risk
        + 1.5 * (LTV_ratio - 0.70)           # higher LTV → higher risk
        + 1.0 * (interest_rate - 0.10) / 0.10  # higher rate → higher risk
        + rng.normal(0, 0.6, n)               # individual variation
    )

    # Use exponential distribution with risk_score modulating the rate
    # Higher risk_score → faster defaults (shorter survival)
    # Target: ~35% censored at 24 months
    base_lambda = 0.03  # slow baseline hazard
    hazard_rate = base_lambda * np.exp(risk_score * 0.4)
    survival_months = rng.exponential(scale=1 / hazard_rate)
    survival_months = np.clip(survival_months, 1, 60)

    # Apply censoring at censor_at_month
    event_default = (survival_months <= censor_at_month).astype(int)
    time_end = np.where(
        event_default == 1,
        np.round(survival_months).astype(int),
        censor_at_month
    )
    time_start = np.zeros(n, dtype=int)

    df = pd.DataFrame({
        "time_start": time_start,
        "time_end": time_end,
        "event_default": event_default,
        "income": income.astype(int),
        "credit_score": credit_score,
        "employment_years": np.round(employment_years, 2),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_amount": loan_amount.astype(int),
        "interest_rate": np.round(interest_rate, 4),
        "LTV_ratio": np.round(LTV_ratio, 4),
    })

    return df


def get_credit_score_band(score):
    """Map numeric credit score to band label."""
    if score < 580:
        return "Deep Subprime (<580)"
    elif score < 670:
        return "Subprime (580-669)"
    elif score < 740:
        return "Near Prime (670-739)"
    else:
        return "Prime (740+)"


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")
    print(f"Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(df.describe().round(2))