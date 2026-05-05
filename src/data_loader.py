import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_loan_data(n=5000, seed=42):
    """Generate synthetic loan data with survival outcomes."""
    np.random.seed(seed)

    # Credit score bands: <580 (Deep Subprime), 580-669 (Subprime),
    # 670-739 (Near Prime), 740+ (Prime)
    credit_scores = np.random.normal(680, 100, n).clip(300, 850).astype(int)

    # Income: skewed distribution, correlated with credit score
    base_income = 40000 + (credit_scores - 500) * 150
    income = np.random.lognormal(np.log(base_income), 0.35).clip(15000, 500000)

    # Employment years: older borrowers have more tenure
    age = (credit_scores - 300) / 15 + np.random.normal(30, 10, n)
    employment_years = np.clip((age - 18) * 0.7 + np.random.exponential(2, n), 0, 50)

    # Loan amount: correlated with income and credit score
    loan_amount = income * (0.3 + (850 - credit_scores) / 2000) + np.random.normal(0, 10000, n)
    loan_amount = loan_amount.clip(5000, 500000)

    # Interest rate: inverse relation with credit score
    base_rate = 12 - (credit_scores - 400) / 60
    interest_rate = base_rate + np.random.normal(0, 1.5, n)
    interest_rate = interest_rate.clip(3, 25)

    # LTV ratio: loan-to-value
    ltv_ratio = np.random.beta(2, 8, n) * 0.95 + 0.3
    ltv_ratio = ltv_ratio.clip(0.2, 1.0)

    # Debt-to-income ratio
    debt_to_income = np.random.lognormal(-1.5, 0.6, n) + 0.1
    debt_to_income = debt_to_income.clip(0.05, 0.8)

    # Time to default/hazard rate depends on credit score and other factors
    # Lower credit score = higher hazard = shorter time to default
    hazard_base = np.exp(-7 + (850 - credit_scores) / 150)
    hazard_base *= (1 + debt_to_income * 0.5)
    hazard_base *= (1 + ltv_ratio * 0.3)
    hazard_base *= (1 + interest_rate / 20)

    # Time to default (in months) for those who default
    time_to_default = np.random.exponential(1 / hazard_base)

    # Censor at 24 months for ~35% of observations
    censor_time = np.full(n, 24.0)
    event_default = (time_to_default <= 24).astype(int)
    time_end = np.where(time_to_default <= 24, time_to_default, 24.0)

    df = pd.DataFrame({
        'time_start': np.zeros(n),
        'time_end': time_end,
        'event_default': event_default,
        'income': income.round(2),
        'credit_score': credit_scores,
        'employment_years': employment_years.round(1),
        'debt_to_income': debt_to_income.round(4),
        'loan_amount': loan_amount.round(2),
        'interest_rate': interest_rate.round(3),
        'ltv_ratio': ltv_ratio.round(4)
    })

    return df

def get_credit_score_band(score):
    """Assign credit score to band."""
    if score < 580:
        return 'Deep Subprime (<580)'
    elif score < 670:
        return 'Subprime (580-669)'
    elif score < 740:
        return 'Near Prime (670-739)'
    else:
        return 'Prime (740+)'

if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Default rate: {df['event_default'].mean():.2%}")
    print(df.head())