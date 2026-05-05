"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd


def generate_loan_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate loan data with survival outcomes.
    
    Args:
        n_samples: Number of loan records to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with columns: time_start, time_end, event_default, income,
        credit_score, employment_years, debt_to_income, loan_amount,
        interest_rate, LTV_ratio.
    """
    np.random.seed(seed)
    
    # Credit score distribution (realistic US range)
    credit_score = np.random.normal(680, 100, n_samples).clip(300, 850).astype(int)
    
    # Income (annual, in $)
    income = np.random.lognormal(10.8, 0.45, n_samples).clip(15000, 500000)
    
    # Employment years
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)
    
    # Debt-to-income ratio
    debt_to_income = np.random.lognormal(2.8, 0.6, n_samples).clip(0.1, 0.8)
    
    # Loan amount (correlated with income)
    loan_amount = income * (0.5 + np.random.uniform(0.1, 0.8, n_samples))
    loan_amount = loan_amount.clip(5000, 500000)
    
    # Interest rate (higher for lower credit scores)
    base_rate = 0.04
    credit_adjustment = (700 - credit_score) / 700 * 0.08
    interest_rate = base_rate + credit_adjustment.clip(0, 0.15) + np.random.uniform(0, 0.02, n_samples)
    
    # LTV ratio
    LTV_ratio = np.random.beta(2, 8, n_samples).clip(0.2, 0.95)
    
    # Base hazard varies by credit score and DTI
    base_hazard = np.exp(-7 + (700 - credit_score) / 100 * 2 + (debt_to_income - 0.3) * 3)
    
    # Time to default (exponential with credit-dependent hazard)
    time_to_default = np.random.exponential(1 / base_hazard.clip(0.01, None))
    time_to_default = time_to_default.clip(1, 120)
    
    # Censoring at 24 months (~35% censored)
    time_end = np.where(time_to_default > 24, 24, time_to_default)
    event_default = np.where(time_to_default <= 24, 1, 0)
    
    # Mark censored
    censored_mask = time_to_default > 24
    event_default[censored_mask] = 0
    
    # Recalculate for censored
    time_end = np.where(censored_mask, 24, time_to_default)
    
    df = pd.DataFrame({
        'time_start': np.zeros(n_samples, dtype=float),
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


def load_data() -> pd.DataFrame:
    """Load or generate loan data."""
    return generate_loan_data()


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loan records")
    print(f"Censored: {df['event_default'].eq(0).sum()} ({df['event_default'].eq(0).mean()*100:.1f}%)")
    print(f"Defaults: {df['event_default'].eq(1).sum()} ({df['event_default'].eq(1).mean()*100:.1f}%)")
    print(df.head())