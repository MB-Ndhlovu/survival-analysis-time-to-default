"""
Data loader for loan survival analysis.
Generates synthetic loan data with time-to-default information.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_loan_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan data for survival analysis.

    Args:
        n_samples: Number of loan records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with loan features and survival data
    """
    np.random.seed(seed)

    # Generate credit scores (FICO-style bands)
    credit_score = np.random.normal(680, 100, n_samples).clip(300, 850).astype(int)

    # Generate income ($30k - $250k, skewed toward middle)
    income = np.random.lognormal(10.8, 0.5, n_samples).clip(20000, 500000)

    # Generate employment years (0 - 40)
    employment_years = np.random.exponential(8, n_samples).clip(0, 40)

    # Generate loan amount ($5k - $100k)
    loan_amount = np.random.lognormal(9.5, 0.7, n_samples).clip(5000, 150000)

    # Interest rate correlated with credit score
    base_rate = 0.12 - (credit_score - 580) * 0.0002
    interest_rate = (base_rate + np.random.normal(0, 0.015, n_samples)).clip(0.03, 0.25)

    # Debt-to-income ratio
    annual_debt_payment = loan_amount * interest_rate
    debt_to_income = (annual_debt_payment / income).clip(0.05, 0.6)

    # LTV ratio (loan-to-value)
    collateral_value = loan_amount * np.random.uniform(0.8, 1.3, n_samples)
    LTV_ratio = (loan_amount / collateral_value).clip(0.3, 1.5)

    # --- Generate survival times ---
    # Hazard increases with: lower credit score, higher DTI, higher rate, lower income
    # Base hazard is a function of credit quality
    credit_factor = (850 - credit_score) / 200
    dti_factor = debt_to_income * 2
    rate_factor = interest_rate * 5
    income_factor = 1 / (income / 100000 + 0.5)

    # Combine into monthly hazard
    monthly_hazard = 0.001 + credit_factor * 0.008 + dti_factor * 0.003 + rate_factor * 0.002 + income_factor * 0.002

    # Generate time-to-default using exponential survival
    survival_time = np.random.exponential(1 / monthly_hazard)

    # Cap at 60 months max
    survival_time = survival_time.clip(0, 60)

    # --- Create censoring indicator ---
    # ~35% censored at 24 months
    censor_threshold = 24
    censored_mask = survival_time > censor_threshold

    # Time_end = min(survival_time, censor_threshold) for censored
    time_end = np.where(censored_mask, censor_threshold, np.floor(survival_time))
    event_default = np.where(censored_mask, 0, 1).astype(int)

    # Time_start is always 0 (all loans start at observation beginning)
    time_start = np.zeros(n_samples, dtype=int)

    # Build DataFrame
    df = pd.DataFrame({
        'time_start': time_start,
        'time_end': time_end.astype(int),
        'event_default': event_default,
        'income': income.astype(int),
        'credit_score': credit_score,
        'employment_years': employment_years.round(1),
        'debt_to_income': debt_to_income.round(4),
        'loan_amount': loan_amount.astype(int),
        'interest_rate': interest_rate.round(4),
        'LTV_ratio': LTV_ratio.round(4)
    })

    return df


def get_credit_score_band(score: int) -> str:
    """Map credit score to band label."""
    if score < 580:
        return '< 580 (Subprime)'
    elif score < 670:
        return '580-669 (Near-prime)'
    elif score < 740:
        return '670-739 (Prime)'
    else:
        return '740+ (Super-prime)'


def add_credit_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit score band column to DataFrame."""
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(get_credit_score_band)
    return df


if __name__ == '__main__':
    df = generate_loan_data()
    print(f"Generated {len(df)} loan records")
    print(f"Censored: {df['event_default'].eq(0).sum()} ({df['event_default'].eq(0).mean()*100:.1f}%)")
    print(f"Defaults: {df['event_default'].eq(1).sum()} ({df['event_default'].eq(1).mean()*100:.1f}%)")
    print("\nSample data:")
    print(df.head())