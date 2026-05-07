"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd

np.random.seed(42)

CENSORE_MONTHS = 24
N_SAMPLES = 5000
CENSOR_RATE = 0.35


def generate_loan_data(n_samples=N_SAMPLES, censore_months=CENSORE_MONTHS, censor_rate=CENSOR_RATE):
    """Generate loan data with survival characteristics.

    Returns DataFrame with columns:
        time_start: observation start month (0)
        time_end: event or censoring month
        event_default: 1 if defaulted, 0 if censored
        income: annual income in ZAR
        credit_score: FICO-equivalent score
        employment_years: years employed
        debt_to_income: monthly debt payment / monthly income
        loan_amount: loan principal in ZAR
        interest_rate: annual interest rate as decimal
        LTV_ratio: loan-to-value ratio
    """
    # Credit score determines baseline risk
    credit_score = np.random.normal(680, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    # Income correlates loosely with credit score
    income = credit_score * 150 + np.random.normal(30000, 20000, n_samples)
    income = np.clip(income, 15000, 500000)

    # Employment years
    employment_years = np.random.exponential(5, n_samples)
    employment_years = np.clip(employment_years, 0, 40)

    # Debt to income ratio - higher for lower credit scores
    base_dti = 0.28 + (700 - credit_score) / 700 * 0.15
    debt_to_income = base_dti + np.random.normal(0, 0.05, n_samples)
    debt_to_income = np.clip(debt_to_income, 0.05, 0.6)

    # Loan amount - correlated with income and credit
    loan_amount = income * (0.3 + (credit_score - 500) / 1000) + np.random.normal(0, 50000, n_samples)
    loan_amount = np.clip(loan_amount, 10000, 2000000)

    # Interest rate - higher for lower credit scores (risk-based pricing)
    base_rate = 0.08 + (700 - credit_score) / 700 * 0.12
    interest_rate = base_rate + np.random.normal(0, 0.015, n_samples)
    interest_rate = np.clip(interest_rate, 0.04, 0.25)

    # LTV ratio - loan amount / collateral value (assume collateral ~ income * 2)
    collateral = income * 2
    LTV_ratio = loan_amount / collateral + np.random.normal(0, 0.05, n_samples)
    LTV_ratio = np.clip(LTV_ratio, 0.1, 1.5)

    # Survival time model:
    # Base hazard is higher for lower credit scores
    # Hazard increases with DTI, interest rate, LTV
    base_hazard = (
        0.15
        - 0.0003 * credit_score
        + 0.5 * debt_to_income
        + 2.0 * (interest_rate - 0.08)
        + 0.3 * LTV_ratio
        + np.random.normal(0, 0.3, n_samples)
    )

    # Convert hazard to approximate survival months (exponential model)
    time_to_default = np.random.exponential(scale=1 / np.maximum(base_hazard, 0.01), size=n_samples)
    time_to_default = np.clip(time_to_default, 1, 60)

    # Determine censoring - some loans haven't defaulted by censore_months
    censor_mask = np.random.random(n_samples) < censor_rate
    # Among those censored, some may naturally survive past 24 months
    # Use actual time_to_default for non-censored
    time_end = np.where(censor_mask, censore_months, np.minimum(time_to_default, censore_months))

    # Event: 1 if defaulted (time_end < censore_months and not censored), 0 if censored
    event_default = (~censor_mask).astype(int)

    # Adjust: if actual default happened before censoring, mark as event
    actual_default_mask = (time_to_default < censore_months) & (~censor_mask)
    time_end = np.where(actual_default_mask, time_to_default, time_end)
    event_default = np.where(actual_default_mask, 1, event_default)

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': time_end.astype(int),
        'event_default': event_default,
        'income': income.astype(int),
        'credit_score': credit_score,
        'employment_years': np.round(employment_years, 1),
        'debt_to_income': np.round(debt_to_income, 4),
        'loan_amount': loan_amount.astype(int),
        'interest_rate': np.round(interest_rate, 4),
        'LTV_ratio': np.round(LTV_ratio, 4),
    })

    return df


def get_credit_score_band(score):
    """Assign credit score to risk band."""
    if score < 580:
        return 'Very Poor (<580)'
    elif score < 670:
        return 'Fair (580-669)'
    elif score < 740:
        return 'Good (670-739)'
    else:
        return 'Excellent (740+)'


if __name__ == '__main__':
    df = generate_loan_data()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Censored: {(df['event_default'] == 0).sum()} ({(df['event_default'] == 0).mean()*100:.1f}%)")
    print(f"Defaults: {(df['event_default'] == 1).sum()}")