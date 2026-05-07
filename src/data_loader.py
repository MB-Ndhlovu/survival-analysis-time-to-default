"""
Generate synthetic loan data with survival analysis structure.
Creates 5000 loan records with time-to-default or censoring.
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(seed=42)


def generate_loan_data(n=5000, censor_at=24):
    """
    Generate synthetic loan data suitable for survival analysis.

    Parameters
    ----------
    n : int
        Number of loan records to generate
    censor_at : int
        Right-censoring time in months (default 24)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time_start: observation start time (0)
        - time_end: event time or censoring time
        - event_default: 1 if default occurred, 0 if censored
        - income: annual income in ZAR
        - credit_score: credit score (300-850)
        - employment_years: years employed
        - debt_to_income: monthly debt payment / monthly income
        - loan_amount: loan principal in ZAR
        - interest_rate: annual interest rate as decimal
        - LTV_ratio: loan-to-value ratio
    """
    # Credit score distribution (skewed toward higher scores)
    credit_score = np.clip(rng.normal(680, 100, n), 300, 850).astype(int)

    # Employment years (exponential-like distribution)
    employment_years = rng.exponential(5, n)
    employment_years = np.clip(employment_years, 0, 40)

    # Income based on credit score (higher score = higher income)
    base_income = 25000 + credit_score * 150
    income = np.clip(base_income + rng.normal(0, 10000, n), 15000, 500000)

    # Loan amount based on income and credit
    loan_amount = income * rng.uniform(0.5, 3.0, n)
    loan_amount = np.clip(loan_amount, 50000, 2000000)

    # Interest rate inversely related to credit score
    base_rate = 0.28 - (credit_score - 300) * 0.0003
    interest_rate = np.clip(base_rate + rng.normal(0, 0.02, n), 0.07, 0.30)

    # Debt-to-income ratio
    monthly_income = income / 12
    monthly_debt = loan_amount * interest_rate / 12 / 30  # simplified payment
    debt_to_income = monthly_debt / monthly_income
    debt_to_income = np.clip(debt_to_income + rng.normal(0, 0.05, n), 0.05, 0.80)

    # LTV ratio (loan amount / collateral value, assume collateral = income * 2)
    collateral = income * 2
    LTV_ratio = loan_amount / collateral
    LTV_ratio = np.clip(LTV_ratio + rng.normal(0, 0.05, n), 0.10, 1.50)

    # Time to default modeling
    # Model: Weibull-like distribution where mean varies by credit score
    # Good credit (750+) -> mean survival ~40 months
    # Poor credit (550) -> mean survival ~10 months
    # This gives us meaningful survival curves

    # Scale factor: higher score = higher scale = longer survival
    # Base scale at credit 550 is ~10, at credit 850 is ~50
    scale = 10 + (credit_score - 500) * (40 / 350)
    scale = np.clip(scale, 8, 50)

    # Adjust scale by other risk factors
    scale = scale * (1 - 0.2 * np.minimum(debt_to_income / 0.40, 1.0))  # DTI impact
    scale = scale * (1 - 0.15 * np.minimum((LTV_ratio - 0.50) / 0.50, 0.5))  # LTV impact
    scale = scale * (1 + 0.05 * np.minimum(employment_years / 20, 1.0))  # job tenure
    scale = np.clip(scale, 5, 55)

    # Shape parameter (Weibull) - lower = more skew toward early defaults
    shape = 0.8 + (credit_score - 500) * (0.3 / 350)
    shape = np.clip(shape, 0.6, 1.2)

    # Generate Weibull survival times
    u = rng.uniform(0, 1, n)
    time_to_default = scale * (-np.log(u)) ** (1 / shape)
    time_to_default = np.clip(time_to_default, 0.5, 100)

    # Censor approximately 35% at censor_at months
    censor_mask = time_to_default > censor_at

    # Set time_end: min of time_to_default or censor_at
    time_end = np.where(censor_mask, censor_at, time_to_default)
    event_default = (~censor_mask).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        'time_start': np.zeros(n, dtype=int),
        'time_end': np.round(time_end, 2),
        'event_default': event_default,
        'income': np.round(income, 2),
        'credit_score': credit_score,
        'employment_years': np.round(employment_years, 2),
        'debt_to_income': np.round(debt_to_income, 3),
        'loan_amount': np.round(loan_amount, 2),
        'interest_rate': np.round(interest_rate, 4),
        'LTV_ratio': np.round(LTV_ratio, 3),
    })

    return df


def add_credit_band(df):
    """Add credit score band column to dataframe."""
    def band(score):
        if score < 580:
            return 'Deep Subprime (<580)'
        elif score < 670:
            return 'Subprime (580-669)'
        elif score < 740:
            return 'Near Prime (670-739)'
        else:
            return 'Prime (740+)'

    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(band)
    return df


if __name__ == '__main__':
    df = generate_loan_data(5000)
    df = add_credit_band(df)
    print(f"Generated {len(df)} loans")
    n_censored = (df['event_default'] == 0).sum()
    n_defaults = df['event_default'].sum()
    print(f"Censored: {n_censored} ({n_censored/len(df)*100:.1f}%)")
    print(f"Defaults: {n_defaults} ({n_defaults/len(df)*100:.1f}%)")
    print("\nCredit band distribution:")
    print(df['credit_band'].value_counts())
    print("\nSample data:")
    print(df.head())