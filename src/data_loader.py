"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd


def generate_loan_data(n_observations: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan data with time-to-default survival characteristics.
    
    Parameters
    ----------
    n_observations : int
        Number of loan records to generate
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        DataFrame with loan features and survival data
    """
    np.random.seed(seed)
    
    # Credit score distribution (weighted toward prime/-super-prime)
    credit_scores = np.concatenate([
        np.random.normal(520, 40, int(n_observations * 0.15)),   # Subprime
        np.random.normal(625, 35, int(n_observations * 0.20)),   # Near-prime
        np.random.normal(705, 30, int(n_observations * 0.35)),   # Prime
        np.random.normal(780, 35, int(n_observations * 0.30)),   # Super-prime
    ])
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    # Income correlates somewhat with credit score
    base_income = 30000 + (credit_scores - 500) * 150 + np.random.normal(0, 15000, n_observations)
    incomes = np.clip(base_income, 15000, 500000).astype(int)
    
    # Employment years
    employment_years = np.random.exponential(5, n_observations) + np.random.uniform(0, 2, n_observations)
    employment_years = np.clip(employment_years, 0, 40).astype(float)
    
    # Debt-to-income ratio
    base_dti = 0.25 + (1 - credit_scores / 850) * 0.35 + np.random.normal(0, 0.05, n_observations)
    debt_to_income = np.clip(base_dti, 0.05, 0.65)
    
    # Loan amount (correlated with income)
    loan_amounts = incomes * np.random.uniform(1.5, 4, n_observations) * (1 + (1 - credit_scores / 850) * 0.5)
    loan_amounts = np.clip(loan_amounts, 5000, 2000000).astype(int)
    
    # Interest rate (inverse of credit score)
    base_rate = 3.5 + (1 - credit_scores / 850) * 12 + np.random.normal(0, 0.5, n_observations)
    interest_rates = np.clip(base_rate, 2.5, 22.0)
    
    # LTV ratio
    base_ltv = 0.75 + (1 - credit_scores / 850) * 0.15 + np.random.normal(0, 0.05, n_observations)
    ltv_ratios = np.clip(base_ltv, 0.3, 1.2)
    
    # --- Survival time generation ---
    # Base hazard depends on credit score (lower score = higher hazard)
    base_hazard = np.exp(-(credit_scores - 300) / 150) * 0.8
    
    # Adjust hazard by other risk factors
    hazard_multiplier = (
        1 + (debt_to_income - 0.3) * 1.5 +
        (ltv_ratios - 0.7) * 0.5 +
        (interest_rates - 6) * 0.03
    )
    
    hazard = base_hazard * hazard_multiplier
    
    # Generate time-to-event (default) from exponential distribution
    time_to_default = np.random.exponential(1 / hazard)
    
    # Censoring: ~35% censored at 24 months
    censor_prob = 0.35
    censors = np.random.random(n_observations) < censor_prob
    
    time_end = np.where(censors, np.minimum(time_to_default, 24.0), time_to_default)
    event_default = np.where(censors, 0, 1).astype(int)
    
    # Ensure time_start is 0 for all
    time_start = np.zeros(n_observations)
    
    df = pd.DataFrame({
        'time_start': time_start,
        'time_end': np.round(time_end, 2),
        'event_default': event_default,
        'income': incomes,
        'credit_score': credit_scores,
        'employment_years': np.round(employment_years, 1),
        'debt_to_income': np.round(debt_to_income, 4),
        'loan_amount': loan_amounts,
        'interest_rate': np.round(interest_rates, 3),
        'LTV_ratio': np.round(ltv_ratios, 4),
    })
    
    return df


def get_credit_band(score: int) -> str:
    """Assign credit score to risk band."""
    if score < 580:
        return 'Subprime (<580)'
    elif score < 670:
        return 'Near-Prime (580-669)'
    elif score < 740:
        return 'Prime (670-739)'
    else:
        return 'Super-Prime (740+)'


if __name__ == '__main__':
    df = generate_loan_data()
    print(f"Generated {len(df)} loan records")
    print(f"Censored: {df['event_default'].eq(0).sum()} ({df['event_default'].eq(0).mean()*100:.1f}%)")
    print(f"Defaulted: {df['event_default'].eq(1).sum()} ({df['event_default'].eq(1).mean()*100:.1f}%)")
    print(df.describe())