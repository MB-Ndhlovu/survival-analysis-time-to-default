"""Generate synthetic loan data with time-to-default observations."""

import numpy as np
import pandas as pd


def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan data for survival analysis.

    Each row represents a loan with:
    - time_start: start date (month 0)
    - time_end: months until default or censoring
    - event_default: 1 if default occurred, 0 if censored
    - Various risk factors
    """
    np.random.seed(seed)

    # Generate risk factors
    credit_score = np.random.normal(680, 80, n).clip(500, 850).astype(int)
    income = np.random.lognormal(10.8, 0.45, n).clip(2000, 500000)  # annual income
    employment_years = np.random.exponential(5, n).clip(0, 40)
    debt_to_income = np.random.beta(2, 8, n) * 0.5  # 0 to 50%
    loan_amount = np.random.lognormal(10.2, 0.55, n).clip(5000, 500000)
    interest_rate = np.random.normal(7.5, 2.5, n).clip(2, 20)
    LTV_ratio = np.random.beta(2, 5, n) * 1.2  # 0 to 120%

    # Base hazard varies by credit score
    # Lower scores = higher base hazard = shorter time to default
    credit_score_norm = (credit_score - 500) / 350  # 0 to 1
    base_hazard = np.exp(-3 - 2 * credit_score_norm)  # higher score = lower hazard

    # Other risk factor effects on hazard
    dti_effect = 1 + 3 * debt_to_income
    LTV_effect = 1 + 1.5 * (LTV_ratio - 0.5).clip(0, 1)
    rate_effect = 1 + 0.05 * (interest_rate - 7)
    employment_effect = np.exp(-0.03 * employment_years)

    total_hazard = base_hazard * dti_effect * LTV_effect * rate_effect * employment_effect

    # Generate time to default (exponential with varying hazard)
    time_to_default = np.random.exponential(1 / total_hazard.clip(0.001, None), n)

    # ~35% censored at 24 months
    censor_time = 24
    time_end = np.minimum(time_to_default, censor_time)
    event_default = (time_to_default <= censor_time).astype(int)

    df = pd.DataFrame({
        "time_start": 0,
        "time_end": np.round(time_end, 1),
        "event_default": event_default,
        "income": np.round(income, 2),
        "credit_score": credit_score,
        "employment_years": np.round(employment_years, 2),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_amount": np.round(loan_amount, 2),
        "interest_rate": np.round(interest_rate, 2),
        "LTV_ratio": np.round(LTV_ratio, 4),
    })

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Censored: {(df['event_default'] == 0).mean():.1%}")
    print(df.describe())