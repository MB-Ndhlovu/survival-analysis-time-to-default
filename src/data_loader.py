"""
Data loader for survival analysis of loan defaults.
Generates 5000 synthetic loan records with realistic distributions.
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_loan_data(n=5000):
    """
    Generate synthetic loan dataset for survival analysis.
    
    Features:
    - income: Annual income in ZAR
    - credit_score: FICO-equivalent score (300-850)
    - employment_years: Years employed
    - debt_to_income: DTI ratio
    - loan_amount: Loan principal in ZAR
    - interest_rate: Annual interest rate as decimal
    - LTV_ratio: Loan-to-value ratio
    
    Output:
    - time_start: Observation start (0 for all)
    - time_end: Time of default or censoring
    - event_default: 1 if defaulted, 0 if censored
    """
    
    # Credit score distribution (skewed toward higher scores)
    credit_score = np.random.normal(680, 120, n)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Income correlated with credit score
    base_income = 400000 + (credit_score - 600) * 2000
    income = base_income + np.random.normal(0, 150000, n)
    income = np.clip(income, 120000, 5000000)
    
    # Employment years
    employment_years = np.random.exponential(5, n) + 0.5
    employment_years = np.clip(employment_years, 0.5, 40)
    
    # Debt-to-income: higher for lower credit scores
    dti_base = 0.25 + (700 - credit_score) / 1500
    debt_to_income = dti_base + np.random.normal(0, 0.08, n)
    debt_to_income = np.clip(debt_to_income, 0.05, 0.65)
    
    # Loan amount proportional to income
    loan_amount = income * np.random.uniform(0.8, 2.5, n)
    loan_amount = np.clip(loan_amount, 50000, 5000000)
    
    # Interest rate inversely related to credit score
    base_rate = 0.12 + (650 - credit_score) / 800
    interest_rate = base_rate + np.random.normal(0, 0.02, n)
    interest_rate = np.clip(interest_rate, 0.045, 0.28)
    
    # LTV ratio (loan / collateral value) — collateral value generated
    collateral = loan_amount * np.random.uniform(1.0, 1.5, n)
    LTV_ratio = loan_amount / collateral
    LTV_ratio = np.clip(LTV_ratio, 0.3, 1.2)
    
    # --- Time to default modeling ---
    # Base hazard depends on credit score band
    # Deep subprime (<580): high hazard, fast defaults
    # Prime (740+): low hazard, slow defaults
    
    # Convert credit score to hazard multiplier
    hazard_base = np.exp(-(credit_score - 300) / 200) * 0.3
    
    # Add risk from other factors
    hazard_dti = debt_to_income * 0.5
    hazard_rate = interest_rate * 2.0
    hazard_ltv = LTV_ratio * 0.3
    
    total_hazard = hazard_base + hazard_dti + hazard_rate + hazard_ltv
    
    # Time to default (exponential with hazard)
    time_to_default = np.random.exponential(1 / (total_hazard + 0.01), n)
    time_to_default = np.clip(time_to_default, 0.5, 84)  # max 7 years
    
    # --- Censoring ---
    # ~35% censored at 24 months
    censoring_mask = np.random.random(n) < 0.35
    
    time_end = np.where(censoring_mask, 24.0, time_to_default)
    event_default = np.where(censoring_mask, 0, 1)
    
    # Ensure non-censored defaults at 24 months or beyond
    time_end = np.clip(time_end, 1.0, 84.0)
    
    df = pd.DataFrame({
        'time_start': 0.0,
        'time_end': time_end,
        'event_default': event_default,
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': LTV_ratio,
    })
    
    return df


def get_credit_score_band(score):
    """Map credit score to risk band."""
    if score < 580:
        return 'Deep Subprime (<580)'
    elif score < 670:
        return 'Subprime (580-669)'
    elif score < 740:
        return 'Near Prime (670-739)'
    else:
        return 'Prime (740+)'


def add_credit_bands(df):
    """Add credit score band column to dataframe."""
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(get_credit_score_band)
    return df


if __name__ == '__main__':
    df = generate_loan_data()
    df = add_credit_bands(df)
    print(f"Generated {len(df)} loans")
    print(f"Censored: {df['event_default'].eq(0).sum()} ({df['event_default'].eq(0).mean()*100:.1f}%)")
    print(f"Defaults: {df['event_default'].eq(1).sum()} ({df['event_default'].eq(1).mean()*100:.1f}%)")
    print(f"\nCredit band distribution:\n{df['credit_band'].value_counts()}")
    print(f"\nSample data:\n{df.head()}")