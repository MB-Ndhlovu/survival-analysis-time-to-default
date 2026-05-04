"""Generate synthetic loan data with survival outcomes for time-to-default analysis."""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_loan_data(n=5000, censor_at=24):
    """Generate synthetic loan data with time-to-default survival characteristics.

    Args:
        n: Number of loan records
        censor_at: Time in months at which uncensored loans are observed

    Returns:
        DataFrame with columns:
            time_start: Start time (0)
            time_end: Event time or censored time
            event_default: 1 if default occurred, 0 if censored
            income: Annual income in thousands
            credit_score: Credit score (300-850)
            employment_years: Years of employment
            debt_to_income: Monthly debt payment / monthly income
            loan_amount: Loan amount in thousands
            interest_rate: Annual interest rate as decimal
            LTV_ratio: Loan-to-value ratio
    """
    # Credit score distribution (skewed toward higher scores)
    credit_score = np.random.normal(680, 100, n).clip(300, 850)

    # Employment years (0-30)
    employment_years = np.random.exponential(5, n).clip(0, 30)

    # Income (20k-200k, correlated with credit score)
    income = (credit_score / 850 * 150 + np.random.normal(0, 20, n) + 20).clip(20, 200)

    # Loan amount (correlated with income)
    loan_amount = (income * 2 + np.random.normal(0, 20, n) * 1000).clip(5, 500).astype(int) / 1000

    # Interest rate (inversely related to credit score)
    base_rate = 0.05 + (850 - credit_score) / 850 * 0.15
    interest_rate = base_rate + np.random.normal(0, 0.01, n)

    # Debt-to-income ratio
    monthly_income = income * 1000 / 12
    monthly_payment = loan_amount * 1000 * (interest_rate / 12)
    debt_to_income = (monthly_payment / monthly_income + np.random.normal(0, 0.05, n)).clip(0.05, 0.8)

    # LTV ratio
    LTV_ratio = (loan_amount * 1000 / (income * 1000 * 0.8) + np.random.normal(0, 0.05, n)).clip(0.1, 1.2)

    # Base hazard (higher for lower credit scores)
    base_hazard = np.exp(-(credit_score - 300) / 200)

    # Time to default simulation
    time_to_default = -np.log(np.random.uniform(0, 1, n)) / (base_hazard * 0.05)
    time_to_default = np.clip(time_to_default, 1, 60).astype(int)

    # Determine censoring (~35% censored at 24 months)
    censor_prob = 0.35
    event_default = np.where(time_to_default <= censor_at, 1, 0)
    time_end = np.where(time_to_default <= censor_at, time_to_default, censor_at)

    # Adjust some events to censor at exactly 24 months
    censor_mask = event_default == 0
    actual_censored = int(n * censor_prob)
    censor_indices = np.random.choice(n, actual_censored, replace=False)
    event_default[censor_indices] = 0
    time_end[censor_indices] = censor_at

    df = pd.DataFrame({
        'time_start': 0,
        'time_end': time_end,
        'event_default': event_default,
        'income': income.round(2),
        'credit_score': credit_score.round().astype(int),
        'employment_years': employment_years.round(2),
        'debt_to_income': debt_to_income.round(4),
        'loan_amount': loan_amount.round(2),
        'interest_rate': interest_rate.round(4),
        'LTV_ratio': LTV_ratio.round(4)
    })

    return df

if __name__ == "__main__":
    df = generate_loan_data()
    df.to_csv('loan_data.csv', index=False)
    print(f"Generated {len(df)} loan records")
    print(f"Default rate: {df['event_default'].mean():.1%}")
    print(f"Censored at 24 months: {(df['event_default'] == 0).mean():.1%}")