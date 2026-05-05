"""Generate synthetic loan data with survival times and covariates."""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_loan_data(n=5000, censor_at=24, default_rate=0.65):
    """
    Generate n loan records with survival data.
    - ~35% are right-censored at censor_at months
    - ~65% experience default (event=1)
    """
    # Credit score distribution
    credit_score = np.random.normal(680, 80, n).clip(500, 850).astype(int)

    # Income in ZAR
    income = np.random.lognormal(10.5, 0.5, n)

    # Employment years
    employment_years = np.random.exponential(4, n).clip(0, 30)

    # Debt-to-income ratio
    debt_to_income = np.random.beta(2, 8, n) * 0.6

    # Loan amount (R50k - R2M)
    loan_amount = np.random.lognormal(12, 0.8, n).clip(50_000, 2_000_000)

    # Interest rate (prime-linked)
    base_rate = 0.07 + (850 - credit_score) / 3500
    interest_rate = base_rate + np.random.normal(0, 0.01, n).clip(-0.02, 0.02)

    # LTV ratio
    LTV_ratio = np.random.beta(3, 7, n) * 1.2

    # Assign event (default) status first to match desired rate
    events = np.random.binomial(1, default_rate, n)

    # Time to default: generate all times, then override censored to censor_at
    # Base hazard varies by credit score
    base_hazard = 0.05 + np.exp(-(credit_score - 500) / 200) * 0.3
    base_hazard *= (1 + debt_to_income * 1.5)
    base_hazard *= (1 + LTV_ratio * 0.3)
    base_hazard = base_hazard.clip(0.01, 2.0)

    # Time to default (months)
    time_to_default = np.random.exponential(1 / base_hazard, n).clip(0.5, censor_at - 0.1)

    # For censored, time_end = censor_at
    time_end = np.where(events == 1, time_to_default.round(1), censor_at)

    df = pd.DataFrame({
        "time_start": np.zeros(n, dtype=int),
        "time_end": time_end,
        "event_default": events,
        "income": income.round(0).astype(int),
        "credit_score": credit_score,
        "employment_years": employment_years.round(1),
        "debt_to_income": (debt_to_income * 100).round(2),
        "loan_amount": loan_amount.round(0).astype(int),
        "interest_rate": (interest_rate * 100).round(2),
        "LTV_ratio": LTV_ratio.round(3),
    })

    return df

def get_credit_band(score):
    if score < 580:
        return "Very Poor (<580)"
    elif score < 670:
        return "Fair (580-669)"
    elif score < 740:
        return "Good (670-739)"
    else:
        return "Excellent (740+)"

if __name__ == "__main__":
    df = generate_loan_data()
    df.to_csv("/home/workspace/Projects/survival-analysis-time-to-default/loan_data.csv", index=False)
    print(f"Generated {len(df)} records")
    print(f"Defaulted: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"Censored: {(df['event_default']==0).sum()} ({(1-df['event_default'].mean())*100:.1f}%)")
    print(df.head())