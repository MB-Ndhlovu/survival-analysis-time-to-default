"""Generate synthetic loan data with survival times for time-to-default analysis."""

import numpy as np
import pandas as pd
from numpy.random import default_rng

RNG = default_rng(seed=42)

def generate_loan_data(n=5000):
    """
    Generate n rows of synthetic loan data with survival characteristics.

    Returns DataFrame with columns:
    - time_start: observation start time (0)
    - time_end: event or censoring time in months
    - event_default: 1 if defaulted, 0 if censored
    - income: annual income in ZAR
    - credit_score: FICO-equivalent score (300-850)
    - employment_years: years at current employer
    - debt_to_income: monthly debt / monthly income ratio
    - loan_amount: loan principal in ZAR
    - interest_rate: annual rate as decimal
    - LTV_ratio: loan-to-value ratio (loan / collateral value)

    The risk model creates three tiers:
    - HIGH RISK (score < 620): defaults in months 2-18 (median ~9m)
    - MEDIUM RISK (620-720): defaults in months 6-30 (median ~16m)
    - LOW RISK (720+): defaults in months 12-40 or censored at 24
    """
    # Credit score distribution
    credit_score = np.clip(RNG.normal(670, 110, n).astype(int), 300, 850)

    # Income
    income = (credit_score - 300) * 80 + RNG.normal(80_000, 35_000, n)
    income = np.clip(income, 12_000, 2_000_000)

    # Employment years
    employment_years = RNG.exponential(5, n)
    employment_years = np.clip(employment_years, 0, 40)

    # DTI ratio
    debt_to_income = RNG.uniform(0.1, 0.5, n) + (850 - credit_score) / 2500
    debt_to_income = np.clip(debt_to_income, 0.05, 0.95)

    # Loan amount
    loan_amount = income * RNG.uniform(0.3, 2.5, n)
    loan_amount = np.clip(loan_amount, 10_000, 5_000_000)

    # Interest rate: strong function of credit score
    interest_rate = 0.06 + (700 - credit_score) / 1000 + RNG.uniform(0, 0.03, n)
    interest_rate = np.clip(interest_rate, 0.045, 0.22)

    # LTV ratio
    LTV_ratio = RNG.uniform(0.5, 0.95, n) + (850 - credit_score) / 3000
    LTV_ratio = np.clip(LTV_ratio, 0.3, 1.5)

    # Risk tier: 0=low, 1=medium, 2=high
    risk_tier = np.zeros(n, dtype=int)
    risk_tier[credit_score < 620] = 2   # high risk
    risk_tier[(credit_score >= 620) & (credit_score < 720)] = 1  # medium

    # Time to default by tier (in months)
    # Tier 2 (HIGH): ~90% default by month 18, median ~8m
    # Tier 1 (MED): ~80% default by month 24, median ~15m
    # Tier 0 (LOW): ~40% default by month 30, censored at 24
    time_to_default = np.zeros(n)
    rng_uniform = RNG.uniform(0, 1, n)

    # Tier 2 - high risk
    mask_h = risk_tier == 2
    # Weibull-like: most default early, few survive long
    # Use power transformation to get right skew
    ttd_h = 2 + (rng_uniform[mask_h] ** 0.5) * 20  # 2-22 months
    time_to_default[mask_h] = ttd_h

    # Tier 1 - medium risk
    mask_m = risk_tier == 1
    ttd_m = 6 + (rng_uniform[mask_m] ** 0.6) * 22  # 6-28 months
    time_to_default[mask_m] = ttd_m

    # Tier 0 - low risk
    mask_l = risk_tier == 0
    ttd_l = 14 + (rng_uniform[mask_l] ** 0.7) * 40  # 14-54 months
    time_to_default[mask_l] = ttd_l

    # Add noise
    time_to_default += RNG.exponential(1.5, n)
    time_to_default = np.clip(time_to_default, 1, 60)

    # Censor ~35% at 24 months (weighted: more low-risk loans censored)
    censor_prob = np.where(risk_tier == 0, 0.50, np.where(risk_tier == 1, 0.30, 0.10))
    censor_mask = RNG.random(n) < censor_prob
    time_end = np.where(censor_mask, 24.0, time_to_default)
    event_default = np.where(censor_mask, 0, 1)

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': time_end,
        'event_default': event_default,
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': LTV_ratio,
    })

    return df

def load_data():
    return generate_loan_data(5000)

if __name__ == '__main__':
    df = load_data()
    print(f"Generated {len(df)} rows")
    print(f"Default rate: {df['event_default'].mean():.1%}")
    print(f"Censored rate: {(df['event_default'] == 0).mean():.1%}")
    print(df.describe())