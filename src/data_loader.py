"""
Data loader: generate synthetic loan data for survival analysis.
5000 rows with time_start, time_end (or censored), event_default, and risk covariates.
Roughly 35% of observations are censored (mostly at 24 months).
"""

import numpy as np
import pandas as pd


def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    credit_score = np.random.normal(680, 80, n).clip(500, 850).astype(int)
    income = np.random.lognormal(10.5, 0.5, n)
    employment_years = np.random.exponential(5, n).clip(0, 30)
    debt_to_income = np.random.beta(2, 8, n) * 0.6 + 0.05
    loan_amount = np.random.lognormal(12.0, 0.8, n).clip(20000, 1_200_000)
    interest_rate = np.random.normal(13.5, 3, n).clip(7, 22)
    LTV_ratio = np.random.beta(5, 5, n) * 0.6 + 0.5

    credit_z = (credit_score - 680) / 80
    income_z = (np.log(income) - 10.5) / 0.5
    dti_z = (debt_to_income - 0.25) / 0.1
    ltv_z = (LTV_ratio - 0.8) / 0.15
    emp_z = (employment_years - 5) / 5

    health = (
        0.35 * credit_z
        + 0.15 * income_z
        - 0.25 * dti_z
        - 0.20 * ltv_z
        + 0.05 * emp_z
    )
    health = (health - health.mean()) / health.std()

    monthly_hazard = 0.003 + 0.018 * np.exp(-health * 1.2)

    max_time = 36
    censor_cutoff = 24

    time_end = np.zeros(n, dtype=int)
    event_default = np.zeros(n, dtype=int)

    for i in range(n):
        t = 0
        defaulted = False
        while t < max_time:
            if np.random.random() < monthly_hazard[i]:
                time_end[i] = t + 1
                event_default[i] = 1
                defaulted = True
                break
            t += 1

        if not defaulted:
            # Censored at observation cutoff (24m) with ~35% probability overall
            time_end[i] = censor_cutoff

    n_censored = (event_default == 0).sum()
    # Re-censor a subset of those not yet defaulted at 24m to hit ~35%
    # Flip some censored to earlier times (loan repaid / lost to follow-up)
    censor_mask = event_default == 0
    censored_indices = np.where(censor_mask)[0]
    # Keep ~35% of total as censored — randomly mark some as "repaid at < 24m"
    target_censored = int(0.35 * n)
    n_repay = len(censored_indices) - target_censored
    if n_repay > 0:
        repay_idx = np.random.choice(censored_indices, size=n_repay, replace=False)
        time_end[repay_idx] = np.random.randint(1, censor_cutoff, size=n_repay)

    df = pd.DataFrame({
        "time_start":      np.zeros(n, dtype=int),
        "time_end":        time_end,
        "event_default":  event_default,
        "income":          np.round(income, 2),
        "credit_score":    credit_score,
        "employment_years": np.round(employment_years, 2),
        "debt_to_income":  np.round(debt_to_income, 4),
        "loan_amount":     np.round(loan_amount, 2),
        "interest_rate":  np.round(interest_rate, 2),
        "LTV_ratio":       np.round(LTV_ratio, 4),
    })

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} rows")
    print(f"Censored: {(df['event_default'] == 0).sum()} ({(df['event_default'] == 0).mean()*100:.1f}%)")
    print(f"Defaults: {(df['event_default'] == 1).sum()}")
    print(df.head())