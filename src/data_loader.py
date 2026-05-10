"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd


def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate n rows of synthetic loan data for survival analysis.

    Each row represents a loan with time-to-default or censoring information.
    ~35% of observations are censored at 24 months.
    """
    np.random.seed(seed)

    # --- Feature generation ---
    credit_score = np.random.normal(680, 80, n).clip(300, 850).astype(int)

    income = np.random.lognormal(10.5, 0.6, n).clip(30_000, 250_000)

    employment_years = np.random.exponential(5, n).clip(0, 40)

    debt_to_income = np.random.uniform(0.05, 0.45, n)

    loan_amount = np.random.lognormal(12, 0.8, n).clip(50_000, 2_000_000)

    base_rate = 0.12
    credit_effect = (1 - (credit_score - 300) / 550) * 0.10
    interest_rate = base_rate + credit_effect + np.random.normal(0, 0.015, n)
    interest_rate = interest_rate.clip(0.045, 0.28)

    LTV_ratio = np.random.uniform(0.50, 0.95, n)

    # --- Build composite risk score (higher = faster default) ---
    # Each component contributes to the monthly hazard
    credit_norm = (credit_score - 300) / 550  # 0 = worst, 1 = best
    credit_risk = (1 - credit_norm) * 2.5
    dti_risk = (debt_to_income - 0.05) / 0.40 * 2.0
    ltv_risk = (LTV_ratio - 0.50) / 0.45 * 1.5
    rate_risk = (interest_rate - 0.045) / 0.235 * 1.0
    income_norm = (income - 30_000) / 220_000  # normalized 0-1
    emp_risk = np.exp(-employment_years / 10) * 0.5  # shorter emp = higher risk
    loan_norm = loan_amount / 2_000_000
    loan_risk = loan_norm * 0.8

    composite = (
        credit_risk * 2.0
        + dti_risk * 1.5
        + ltv_risk * 1.2
        + rate_risk * 1.0
        + emp_risk
        + loan_risk
    )

    # Map composite risk to average survival time
    # Lowest risk ~45 months, highest risk ~8 months
    expected_survival = np.clip(45 - composite * 8, 8, 45)

    # --- Simulate survival times using inverse transform ---
    # Using Exponential distribution per individual (constant hazard)
    # S(t) = exp(-lambda * t), so T = -ln(U) / lambda
    lambdas = 1.0 / expected_survival  # monthly hazard rate

    time_to_default = np.zeros(n)
    event_default = np.zeros(n)

    for i in range(n):
        if lambdas[i] <= 0:
            lam = 0.001
        else:
            lam = lambdas[i]

        if np.random.random() < 0.35:
            # Censored: survive at least 24 months
            time_to_default[i] = 24
            event_default[i] = 0
        else:
            # Draw from Exp(lam) via inverse transform
            U = np.random.random()
            t = -np.log(U) / lam
            t = min(t, 60)  # cap at 60 months
            time_to_default[i] = int(round(t))
            event_default[i] = 1

    df = pd.DataFrame({
        "time_start": 0,
        "time_end": time_to_default.astype(int),
        "event_default": event_default.astype(int),
        "income": income.astype(int),
        "credit_score": credit_score,
        "employment_years": np.round(employment_years, 2),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_amount": loan_amount.astype(int),
        "interest_rate": np.round(interest_rate, 4),
        "LTV_ratio": np.round(LTV_ratio, 4),
    })

    return df


def load_data() -> pd.DataFrame:
    """Load or generate loan survival data."""
    return generate_loan_data(n=5000, seed=42)


if __name__ == "__main__":
    df = load_data()
    print(f"Generated {len(df)} loans")
    print(df[["time_end", "event_default", "credit_score"]].describe())
    print(f"\nCensored: {(df['event_default']==0).mean():.1%}")
    print(f"Defaults: {(df['event_default']==1).mean():.1%}")
    default_times = df.loc[df['event_default']==1, 'time_end']
    print(f"Avg survival (defaults only): {default_times.mean():.1f} months")
    print(f"Median survival (defaults only): {default_times.median():.1f} months")