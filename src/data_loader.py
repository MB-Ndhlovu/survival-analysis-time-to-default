"""
Data loader: generate synthetic loan data for survival analysis.
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(seed=42)


def generate_loan_data(n: int = 5000, censor_at: int = 24) -> pd.DataFrame:
    """
    Generate n rows of synthetic loan data with survival characteristics.

    Fields:
    - time_start: observation start (0)
    - time_end: time of default or censoring
    - event_default: 1 if default occurred, 0 if censored
    - income: annual income in R (South African Rand)
    - credit_score: FICO-equivalent score (300-850)
    - employment_years: years employed
    - debt_to_income: monthly debt / monthly income ratio
    - loan_amount: loan principal in R
    - interest_rate: annual interest rate as decimal
    - LTV_ratio: loan-to-value ratio at origination
    """
    records = []

    for i in range(n):
        # Credit score determines baseline risk
        cs = rng.integers(300, 851)

        # Income: R120k - R1.2M, skewed toward lower incomes
        income = rng.exponential(scale=300_000) + 120_000
        income = min(income, 1_200_000)

        # Employment years: 0-30, more people at lower end
        emp_yrs = min(max(rng.exponential(scale=5), 0), 30)

        # Loan amount: R50k - R500k, correlated with income
        loan_amount = min(max(income * rng.uniform(0.2, 0.8), 50_000), 500_000)

        # Interest rate: risk-based pricing
        if cs < 580:
            base_rate = rng.uniform(0.15, 0.22)
        elif cs < 670:
            base_rate = rng.uniform(0.10, 0.16)
        elif cs < 740:
            base_rate = rng.uniform(0.07, 0.11)
        else:
            base_rate = rng.uniform(0.04, 0.08)
        interest_rate = base_rate

        # DTI: higher for riskier borrowers
        if cs < 580:
            dti = rng.uniform(0.30, 0.55)
        elif cs < 670:
            dti = rng.uniform(0.20, 0.40)
        elif cs < 740:
            dti = rng.uniform(0.15, 0.32)
        else:
            dti = rng.uniform(0.08, 0.25)
        debt_to_income = dti

        # LTV: loan_amount / collateral_value
        if cs < 620:
            ltv = rng.uniform(0.70, 0.95)
        elif cs < 720:
            ltv = rng.uniform(0.55, 0.85)
        else:
            ltv = rng.uniform(0.40, 0.75)
        LTV_ratio = ltv

        # Simulate time to default using Weibull distribution
        # Shape and scale vary by credit score
        if cs < 580:
            scale = rng.uniform(8, 14)
            shape = rng.uniform(0.7, 0.9)  # increasing hazard
        elif cs < 670:
            scale = rng.uniform(14, 22)
            shape = rng.uniform(0.8, 1.0)
        elif cs < 740:
            scale = rng.uniform(20, 30)
            shape = rng.uniform(0.9, 1.1)
        else:
            scale = rng.uniform(28, 48)
            shape = rng.uniform(1.0, 1.2)

        ttf = rng.weibull(shape) * scale

        # Determine if censored or defaulted
        # ~35% censored at censor_at months
        censor_prob = 0.35
        if rng.random() < censor_prob:
            time_end = censor_at
            event_default = 0
        else:
            if ttf > censor_at:
                time_end = censor_at
                event_default = 0
            else:
                time_end = min(int(ttf), censor_at)
                event_default = 1

        records.append({
            "time_start": 0,
            "time_end": time_end,
            "event_default": event_default,
            "income": round(income, 2),
            "credit_score": cs,
            "employment_years": round(emp_yrs, 1),
            "debt_to_income": round(debt_to_income, 4),
            "loan_amount": round(loan_amount, 2),
            "interest_rate": round(interest_rate, 4),
            "LTV_ratio": round(LTV_ratio, 4),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_loan_data(5000)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Censor rate: {(df['event_default'] == 0).mean():.1%}")
    print(df.describe())