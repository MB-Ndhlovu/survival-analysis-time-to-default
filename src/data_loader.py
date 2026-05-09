"""
Generate synthetic loan data with time-to-default characteristics.
Roughly 35% of observations are censored at 24 months.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan portfolio data for survival analysis.

    Parameters
    ----------
    n : int
        Number of loan observations (default 5000)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time_start: Start date index (0)
        - time_end: Time of default or censoring
        - event_default: 1 if default occurred, 0 if censored
        - income: Annual income in ZAR
        - credit_score: FICO-equivalent credit score
        - employment_years: Years employed
        - debt_to_income: DTI ratio
        - loan_amount: Loan principal in ZAR
        - interest_rate: Annual interest rate (decimal)
        - LTV_ratio: Loan-to-value ratio
    """
    np.random.seed(seed)

    # Credit score distribution roughly matching real-world portfolios
    credit_scores = np.random.normal(680, 100, n).clip(300, 850).astype(int)

    # Income correlated with credit score (higher score → higher income)
    income_base = np.random.normal(450000, 150000, n)
    income = (income_base + (credit_scores - 680) * 500).clip(50000, 3000000).astype(int)

    # Employment years
    employment_years = np.random.exponential(5, n).clip(0, 40).astype(int)

    # Loan amount correlated with income
    loan_amount = (income * np.random.uniform(0.5, 2.5, n)).clip(50000, 2000000).astype(int)

    # Interest rate inversely correlated with credit score
    base_rate = 0.15
    interest_rate = (base_rate + (700 - credit_scores) * 0.0002 +
                     np.random.normal(0, 0.02, n)).clip(0.06, 0.28)

    # LTV ratio
    LTV_ratio = np.random.uniform(0.5, 1.0, n).round(2)

    # DTI ratio
    debt_to_income = np.random.uniform(0.1, 0.6, n).round(3)

    # Base hazard depends on credit score and DTI
    # Lower credit score → higher hazard; Higher DTI → higher hazard
    base_hazard = (
        0.5 - (credit_scores - 500) * 0.001 +
        debt_to_income * 0.5 +
        LTV_ratio * 0.2
    ).clip(0.01, 0.15)

    # Time to default (exponential distribution)
    time_to_default = np.random.exponential(1 / base_hazard, n)

    # ~35% censored at 24 months
    censoring_threshold = 24
    censor_prob = 0.35
    is_censored = np.random.random(n) < censor_prob

    time_end = np.where(is_censored,
                        np.minimum(time_to_default, censoring_threshold),
                        time_to_default)

    # Event: 1 if default occurred (not censored and time < threshold)
    # 0 if censored OR time >= threshold (observed but didn't default yet)
    event_default = (~is_censored & (time_to_default < censoring_threshold)).astype(int)

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': time_end.round(2),
        'event_default': event_default,
        'income': income,
        'credit_score': credit_scores,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate.round(4),
        'LTV_ratio': LTV_ratio
    })

    return df


def get_credit_score_band(score: int) -> str:
    """Map credit score to band label."""
    if score < 580:
        return 'Subprime (<580)'
    elif score < 670:
        return 'Near-Prime (580-669)'
    elif score < 740:
        return 'Prime (670-739)'
    else:
        return 'Super-Prime (740+)'


def main():
    df = generate_loan_data(5000)
    print(f"Generated {len(df)} loans")
    print(f"Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"Censored: {(df['event_default'] == 0).sum()} ({(df['event_default'] == 0).mean()*100:.1f}%)")
    print(f"\nCredit score distribution:")
    print(df['credit_score'].describe())
    return df


if __name__ == '__main__':
    main()