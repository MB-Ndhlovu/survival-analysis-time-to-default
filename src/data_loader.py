"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(42)

CENSORE_MONTH = 24
N_RECORDS = 5000
CENSOR_RATE = 0.35


def generate_loan_data(n_records: int = N_RECORDS, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan data with time-to-default survival structure.

    Creates 5000 loan records with realistic credit risk features. Default
    probability increases with lower credit scores, higher DTI, higher LTV,
    and shorter employment history.

    Each record has:
    - time_start: 0 (month of origination)
    - time_end: months until default or censoring
    - event_default: 1 if defaulted, 0 if censored
    - income, credit_score, employment_years, debt_to_income, loan_amount,
      interest_rate, LTV_ratio

    Roughly 35% of accounts are censored at 24 months.
    """
    rng = default_rng(seed)

    n = n_records

    # Credit score: uniform distribution 500-820
    credit_score = rng.integers(500, 821, size=n)

    # Income: lognormal distribution $30k-$250k
    income = rng.lognormal(mean=10.8, sigma=0.45, size=n)

    # Employment years: exponential distribution clipped
    employment_years = rng.exponential(scale=5, size=n).clip(0, 40)

    # Debt-to-income: gamma distribution
    debt_to_income = rng.gamma(shape=2, scale=8, size=n).clip(5, 60)

    # Loan amount: function of income and credit
    loan_amount = (income * rng.uniform(0.2, 0.8, size=n) * (800 - credit_score) / 300).clip(5000, 500000)

    # Interest rate: function of credit score
    base_rate = 0.04
    interest_rate = base_rate + (800 - credit_score) / 800 * 0.15 + rng.uniform(0, 0.02, size=n)
    interest_rate = interest_rate.clip(0.035, 0.25)

    # LTV ratio: loan / collateral value
    LTV_ratio = rng.uniform(0.5, 1.1, size=n)

    # Determine default time using hazard model
    # Base hazard scales with risk factors
    risk_score = (
        (800 - credit_score) / 300 * 2.0 +          # lower credit = higher risk
        debt_to_income / 60 * 1.5 +                   # higher DTI = higher risk
        LTV_ratio - 0.7 +                             # higher LTV = higher risk
        (40 - employment_years.clip(0, 40)) / 40 +    # less employment = higher risk
        interest_rate * 5                             # higher rate = higher risk
    )

    # Monthly default probability
    monthly_hazard = 0.002 + 0.015 * risk_score.clip(0, 3) ** 1.5

    # Generate time-to-default
    time_to_default = rng.exponential(scale=1 / monthly_hazard.clip(0.001, 0.5))

    # ~35% censored at 24 months (still performing)
    censor_mask = rng.random(size=n) < CENSOR_RATE

    # Also censor if time_to_default > 24
    time_to_default_adj = time_to_default.copy()
    time_to_default_adj[censor_mask] = CENSORE_MONTH

    time_end = time_to_default_adj.clip(1, CENSORE_MONTH)
    event_default = (~censor_mask).astype(int)

    # Adjust: if defaulted after 24 months, treat as censored
    actual_default = time_to_default > CENSORE_MONTH
    event_default[actual_default] = 0
    time_end[actual_default] = CENSORE_MONTH
    censor_mask[actual_default] = True

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': time_end.astype(int),
        'event_default': event_default,
        'income': income.round(2),
        'credit_score': credit_score,
        'employment_years': employment_years.round(2),
        'debt_to_income': debt_to_income.round(2),
        'loan_amount': loan_amount.round(2),
        'interest_rate': interest_rate.round(4),
        'LTV_ratio': LTV_ratio.round(3),
    })

    return df


def get_credit_band(score: int) -> str:
    """Map credit score to risk band."""
    if score < 580:
        return 'Deep Subprime (<580)'
    elif score < 670:
        return 'Subprime (580-669)'
    elif score < 740:
        return 'Near Prime (670-739)'
    else:
        return 'Prime (740+)'


if __name__ == '__main__':
    df = generate_loan_data()
    print(f"Generated {len(df)} records")
    print(f"Censored: {(df['event_default'] == 0).sum()} ({((df['event_default'] == 0).mean() * 100):.1f}%)")
    print(f"Defaults: {(df['event_default'] == 1).sum()}")
    print(df.describe().round(2))