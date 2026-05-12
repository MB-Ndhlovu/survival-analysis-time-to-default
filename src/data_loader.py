"""Generate synthetic loan data with time-to-default survival structure."""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_loan_data(n=5000, censor_at_month=24):
    """
    Generate synthetic loan data with survival structure.

    Parameters
    ----------
    n : int
        Number of loan records to generate
    censor_at_month : int
        Right-censoring time (observation end)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time_start: 0 (origin)
        - time_end: time of default or censoring
        - event_default: 1 if defaulted, 0 if censored
        - income, credit_score, employment_years, debt_to_income
        - loan_amount, interest_rate, LTV_ratio
    """
    # Covariate distributions
    income = np.random.lognormal(mean=10.8, sigma=0.45, size=n)  # ~45k median
    credit_score = np.random.normal(680, 110, size=n).clip(300, 850).astype(int)
    employment_years = np.random.exponential(scale=4, size=n).clip(0, 40)
    debt_to_income = np.random.beta(2, 8, size=n) * 0.55  # skewed low
    loan_amount = np.random.lognormal(mean=10.2, sigma=0.65, size=n)  # ~30k median
    interest_rate = np.random.beta(2, 5, size=n) * 0.18 + 0.04  # 4-22%
    LTV_ratio = np.random.beta(3, 7, size=n) * 1.2 + 0.5  # 50-170%

    # Baseline hazard varies by credit score band
    # Lower scores -> higher hazard (faster default)
    score_band = np.where(credit_score < 580, 0,
                  np.where(credit_score < 670, 1,
                  np.where(credit_score < 740, 2, 3)))

    # Scale parameters per band: higher = faster default
    scale = np.array([8, 14, 20, 28])[score_band]

    # Time-to-default drawn from Exponential distribution
    time_to_default = np.random.exponential(scale=scale)

    # 35% censored at censor_at_month
    censor_mask = np.random.random(n) < 0.35

    time_end = np.where(censor_mask,
                        censor_at_month,
                        np.minimum(time_to_default, censor_at_month))

    event_default = np.where(censor_mask, 0, 1)

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': np.round(time_end, 2),
        'event_default': event_default,
        'income': np.round(income, 2),
        'credit_score': credit_score,
        'employment_years': np.round(employment_years, 2),
        'debt_to_income': np.round(debt_to_income, 4),
        'loan_amount': np.round(loan_amount, 2),
        'interest_rate': np.round(interest_rate, 4),
        'LTV_ratio': np.round(LTV_ratio, 4),
    })

    return df

if __name__ == '__main__':
    df = generate_loan_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Default rate: {df['event_default'].mean():.3f}")
    print(f"Censored rate: {(df['event_default'] == 0).mean():.3f}")