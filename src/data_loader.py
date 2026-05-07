"""
Generate synthetic loan data for survival analysis.
5000 rows with time-to-default (censored at 24 months for ~35% of observations).
"""

import numpy as np
import pandas as pd


def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    credit_score = rng.integers(480, 840, size=n)
    income = rng.lognormal(10.8, 0.45, size=n)
    employment_years = rng.exponential(5.5, size=n).clip(0, 40)
    debt_to_income = rng.beta(2, 5, size=n) * 0.6
    loan_amount = rng.lognormal(10.2, 0.55, size=n)
    interest_rate = rng.beta(2, 8, size=n) * 0.15 + 0.04
    LTV_ratio = rng.beta(2, 6, size=n) * 0.95

    score_norm = (credit_score - 480) / (840 - 480)

    log_hazard = (
        -2.8 * score_norm
        + 0.8 * (interest_rate - 0.08)
        + 1.5 * debt_to_income
        + 0.4 * LTV_ratio
        - 0.05 * employment_years
        + 0.00002 * (loan_amount - 20000)
        + rng.normal(0, 0.6, size=n)
    )
    base_hazard = np.exp(log_hazard) * 0.08

    time_to_event = rng.exponential(1 / (base_hazard + 0.001), size=n)
    time_to_event = np.clip(time_to_event, 0.5, 120)

    censor_time = np.full(n, 24.0)
    random_censor = rng.uniform(20, 36, size=n)
    censor_mask = rng.choice([True, False], size=n, p=[0.35, 0.65])
    censor_time = np.where(censor_mask, random_censor, 24.0)

    observed_time = np.minimum(time_to_event, censor_time)
    event_default = (time_to_event <= censor_time).astype(int)

    df = pd.DataFrame({
        "time_start": np.zeros(n, dtype=int),
        "time_end": observed_time.round(2),
        "event_default": event_default,
        "income": income.round(2),
        "credit_score": credit_score,
        "employment_years": employment_years.round(2),
        "debt_to_income": debt_to_income.round(4),
        "loan_amount": loan_amount.round(2),
        "interest_rate": interest_rate.round(4),
        "LTV_ratio": LTV_ratio.round(4),
    })

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Censored: {(df['event_default']==0).mean():.1%}")
    print(df.describe())