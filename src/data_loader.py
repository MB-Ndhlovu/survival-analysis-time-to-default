import numpy as np
import pandas as pd

def generate_loan_data(n=5000, seed=42):
    """Generate synthetic loan data with survival structure.

    Args:
        n: Number of loan records
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            time_start, time_end, event_default,
            income, credit_score, employment_years,
            debt_to_income, loan_amount, interest_rate, LTV_ratio
    """
    np.random.seed(seed)

    # --- Covariate distributions ---
    income = np.random.lognormal(mean=10.8, sigma=0.45, size=n)  # ~$50k median
    credit_score = np.random.normal(680, 120, size=n).clip(300, 900).astype(int)
    employment_years = np.random.exponential(scale=4, size=n).clip(0, 40)
    debt_to_income = np.random.beta(2, 8, size=n) * 0.6  # mostly 0.1-0.4
    loan_amount = np.random.lognormal(mean=10.2, sigma=0.7, size=n)  # ~$30k median
    interest_rate = np.random.beta(2, 5, size=n) * 0.15 + 0.03  # 3%-18%
    LTV_ratio = np.random.beta(2, 6, size=n) * 1.2  # mostly 0.1-0.7

    # --- Time-to-default hazard (piecewise exponential) ---
    # Higher risk → shorter survival
    base_hazard = 0.005
    hazard_multiplier = (
        0.5 * (1 - credit_score / 900) +
        0.3 * (debt_to_income / 0.6) +
        0.2 * (LTV_ratio / 1.2) +
        0.1 * (interest_rate / 0.18)
    )

    # Survival time in months (draw from exponential distribution)
    survival_months = np.random.exponential(scale=1 / (base_hazard * np.exp(hazard_multiplier * 2)), size=n)
    survival_months = np.clip(survival_months, 1, 120).astype(int)

    # --- Censoring: ~35% censored at 24 months ---
    censor_cutoff = 24
    event_default = np.ones(n, dtype=int)
    time_end = survival_months.copy()

    censored_mask = np.random.rand(n) < 0.35
    event_default[censored_mask] = 0
    time_end[censored_mask] = np.minimum(survival_months[censored_mask], censor_cutoff)
    # Re-censor: if actual survival < cutoff, keep actual time
    time_end[censored_mask] = np.minimum(
        survival_months[censored_mask],
        censor_cutoff
    )

    time_start = np.zeros(n, dtype=int)

    return pd.DataFrame({
        "time_start": time_start,
        "time_end": time_end,
        "event_default": event_default,
        "income": income.round(2),
        "credit_score": credit_score,
        "employment_years": employment_years.round(2),
        "debt_to_income": debt_to_income.round(4),
        "loan_amount": loan_amount.round(2),
        "interest_rate": interest_rate.round(4),
        "LTV_ratio": LTV_ratio.round(4),
    })

if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} records")
    print(f"Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(df.describe())