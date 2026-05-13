"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate loan data with survival characteristics.

    Args:
        n: Number of observations to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: time_start, time_end, event_default,
        income, credit_score, employment_years, debt_to_income,
        loan_amount, interest_rate, LTV_ratio
    """
    rng = np.random.default_rng(seed)

    # Base characteristics
    income = rng.lognormal(mean=10.8, sigma=0.45, size=n)  # ~$50k median
    credit_score = rng.normal(680, 120, size=n)
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    employment_years = rng.exponential(scale=5, size=n)
    employment_years = np.clip(employment_years, 0, 40)

    debt_to_income = rng.beta(2, 8, size=n) * 0.6  # mostly low DTI
    loan_amount = rng.lognormal(mean=11.5, sigma=0.6, size=n)  # ~$100k median

    # Interest rate based on credit score (risk-based pricing)
    base_rate = 0.05
    rate_premium = (700 - credit_score) / 400 * 0.08  # worse credit = higher rate
    rate_premium = np.maximum(rate_premium, 0)
    interest_rate = base_rate + rate_premium + rng.normal(0, 0.01, size=n)
    interest_rate = np.maximum(interest_rate, 0.02)

    # LTV ratio
    LTV_ratio = rng.beta(3, 7, size=n) * 1.2  # mostly below 80%
    LTV_ratio = np.minimum(LTV_ratio, 1.5)

    # Hazard model: higher score = lower hazard; higher DTI = higher hazard
    # Baseline hazard scales with loan amount and LTV
    log_hazard = (
        -4.0
        - 0.008 * (credit_score - 700)  # higher score = lower hazard
        - 0.25 * employment_years  # more experience = lower hazard
        + 5.0 * debt_to_income  # higher DTI = higher hazard
        + 0.6 * (loan_amount / 100000)  # larger loans = slightly higher hazard
        + 1.2 * LTV_ratio  # higher LTV = higher hazard
        + rng.normal(0, 0.5, size=n)
    )
    hazard = np.exp(log_hazard)

    # Convert hazard to time-to-default (exponential distribution)
    time_to_default = rng.exponential(scale=1 / hazard)

    # Censor at 24 months for ~35% of observations
    max_follow_up = 24
    event_default = (time_to_default <= max_follow_up).astype(int)
    time_end = np.where(event_default == 1, time_to_default, max_follow_up)

    df = pd.DataFrame({
        "time_start": 0,
        "time_end": np.clip(time_end, 0.1, max_follow_up),
        "event_default": event_default,
        "income": np.round(income, 2),
        "credit_score": credit_score,
        "employment_years": np.round(employment_years, 2),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_amount": np.round(loan_amount, 2),
        "interest_rate": np.round(interest_rate, 4),
        "LTV_ratio": np.round(LTV_ratio, 4),
    })

    return df


def get_credit_score_band(score: int) -> str:
    """Assign credit score to risk band."""
    if score < 580:
        return "Deep Subprime (< 580)"
    elif score < 670:
        return "Subprime (580-669)"
    elif score < 740:
        return "Near Prime (670-739)"
    else:
        return "Prime (740+)"


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loan records")
    print(f"Censored: {(df['event_default'] == 0).sum()} ({(df['event_default'] == 0).mean()*100:.1f}%)")
    print(f"Defaults: {(df['event_default'] == 1).sum()} ({(df['event_default'] == 1).mean()*100:.1f}%)")
    print(df.describe())