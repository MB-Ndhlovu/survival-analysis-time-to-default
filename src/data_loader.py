"""Generate synthetic loan data for survival analysis.

Generates 5000 rows with:
  time_start, time_end (or censored), event_default (0/1),
  income, credit_score, employment_years, debt_to_income,
  loan_amount, interest_rate, LTV_ratio

~35% of observations are right-censored at 24 months.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

CENSOR_MONTHS = 24


def generate_loan_data(n=5000):
    """Generate n synthetic loan records."""
    # Credit score drives baseline hazard
    credit_score = np.random.normal(680, 100, n).clip(400, 850).astype(int)

    # Income correlates loosely with credit score
    income = (
        credit_score * 60
        + np.random.normal(0, 15000, n)
        + 20000
    ).clip(15000, 300000).astype(int)

    employment_years = np.random.exponential(5, n).clip(0, 40)

    # Debt-to-income: higher DTI → higher hazard
    debt_to_income = np.random.beta(2, 8, n) * 0.6  # 0–0.6
    loan_amount = (debt_to_income * income * 36).clip(5000, 500000).astype(int)

    # Loan-to-value: >80% is high risk
    LTV_ratio = np.random.beta(2, 5, n) * 1.2  # 0–1.2
    LTV_ratio = LTV_ratio.clip(0.1, 1.2)

    interest_rate = (
        5
        + (850 - credit_score) / 100 * 8
        + LTV_ratio * 3
        + np.random.normal(0, 0.5, n)
    ).clip(3, 22)

    # ---- Build time-to-default via parametric hazard ----
    # log-hazard increases with credit risk factors
    log_hazard = (
        -5.5
        + (580 - credit_score) / 100 * 0.8   # lower score → higher hazard
        + debt_to_income * 4
        + LTV_ratio * 1.5
        - employment_years * 0.03
        + np.random.exponential(0.5, n)
    )
    hazard_rate = np.exp(log_hazard)

    # Time-to-default ~ Exponential(hazard_rate)
    time_to_default = np.random.exponential(1 / hazard_rate)

    # Censor at CENSOR_MONTHS with probability 0.35
    censor_mask = np.random.random(n) < 0.35
    observed_time = np.where(censor_mask, CENSOR_MONTHS, time_to_default)
    event = (~censor_mask).astype(int)  # 1 if default happened, 0 if censored

    df = pd.DataFrame(
        {
            "time_start": 0,
            "time_end": observed_time.round(2),
            "event_default": event,
            "income": income,
            "credit_score": credit_score,
            "employment_years": employment_years.round(2),
            "debt_to_income": debt_to_income.round(4),
            "loan_amount": loan_amount,
            "interest_rate": interest_rate.round(3),
            "LTV_ratio": LTV_ratio.round(4),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Defaults: {df['event_default'].sum()} ({df['event_default'].mean():.1%})")
    print(f"Censored: {(df['event_default'] == 0).sum()} ({(df['event_default'] == 0).mean():.1%})")
    print(df.describe().round(2))