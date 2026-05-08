"""
Generate synthetic loan data for survival analysis.
5000 observations with time-to-default (censored at 24 months).
"""

import numpy as np
import pandas as pd
from scipy import stats


def generate_loan_data(n=5000, seed=42):
    """
    Generate synthetic loan data with survival characteristics.

    Returns DataFrame with columns:
    - time_start: observation start time (0)
    - time_end: event or censoring time
    - event_default: 1 if default, 0 if censored
    - income: annual income in thousands
    - credit_score: FICO score (300-850)
    - employment_years: years employed
    - debt_to_income: monthly debt / monthly income ratio
    - loan_amount: loan principal in thousands
    - interest_rate: annual rate as decimal
    - LTV_ratio: loan-to-value ratio
    """
    np.random.seed(seed)

    # Credit score distribution (approximating US population)
    credit_score = stats.truncnorm(a=-1.5, b=1.5, loc=700, scale=100).rvs(n)
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    # Income: skewed distribution $30k-$250k
    income = stats.lognorm(s=0.5, loc=20, scale=50).rvs(n)
    income = np.clip(income, 25, 500).astype(int)

    # Employment years: exponential decay
    employment_years = np.random.exponential(scale=5, size=n)
    employment_years = np.clip(employment_years, 0, 40).round(1)

    # Debt-to-income: strongly right-skewed
    debt_to_income = stats.lognorm(s=0.4, loc=0.05, scale=0.25).rvs(n)
    debt_to_income = np.clip(debt_to_income, 0.1, 0.8).round(3)

    # Loan amount: correlated with income and credit score
    base_amount = income * 2 + (credit_score - 600) * 0.5
    loan_amount = base_amount * np.random.uniform(0.7, 1.3, n)
    loan_amount = np.clip(loan_amount, 10, 500).astype(int)

    # Interest rate: inversely related to credit score
    base_rate = 0.12 - (credit_score - 500) * 0.0002
    interest_rate = base_rate + np.random.normal(0, 0.02, n)
    interest_rate = np.clip(interest_rate, 0.03, 0.25).round(4)

    # LTV ratio: home loans vs personal
    LTV_ratio = np.random.beta(2, 5, n) * 0.95 + 0.05
    LTV_ratio = LTV_ratio.round(3)

    # Time to default simulation
    # Base hazard increases with: lower credit score, higher DTI, higher rate, higher LTV
    # Default probability is shaped to give ~35% censoring at 24 months

    # Compute underlying hazard scores
    hazard_score = (
        -0.8 * (credit_score - 700) / 200  # lower score = higher hazard
        + 2.0 * (debt_to_income - 0.3)     # higher DTI = higher hazard
        + 3.0 * (interest_rate - 0.10)     # higher rate = higher hazard
        + 1.5 * (LTV_ratio - 0.7)          # higher LTV = higher hazard
        + np.random.normal(0, 0.5, n)      # random variation
    )

    # Convert hazard to time-to-default (exponential with rate = exp(hazard_score))
    default_times = np.random.exponential(scale=1 / np.exp(hazard_score * 0.1), size=n)
    default_times = np.clip(default_times, 1, 60).round(1)  # max 60 months

    # Apply censoring at 24 months (~35% censored)
    censor_prob = 0.35 + 0.1 * (credit_score > 700).astype(float) - 0.1 * (debt_to_income > 0.4).astype(float)
    is_censored = np.random.binomial(1, np.clip(censor_prob, 0.2, 0.5))

    time_end = np.where(is_censored, 24.0, default_times)
    event_default = 1 - is_censored

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': time_end,
        'event_default': event_default,
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': LTV_ratio
    })

    return df


def get_credit_score_band(score):
    """Return credit score band label."""
    if score < 580:
        return 'Deep Subprime (<580)'
    elif score < 670:
        return 'Subprime (580-669)'
    elif score < 740:
        return 'Near Prime (670-739)'
    else:
        return 'Prime (740+)'


if __name__ == '__main__':
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Censored: {df['event_default'].eq(0).sum()} ({df['event_default'].eq(0).mean()*100:.1f}%)")
    print(f"Defaults: {df['event_default'].eq(1).sum()} ({df['event_default'].eq(1).mean()*100:.1f}%)")
    print(df.describe().round(2))