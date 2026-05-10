"""
Generate synthetic loan data for survival analysis.
5000 observations with time-to-default (or censoring at 24 months).
"""

import numpy as np
import pandas as pd


def generate_loan_data(n=5000, seed=42):
    np.random.seed(seed)

    # Credit score distribution (realistic US FICO-like)
    credit_score = np.random.normal(680, 80, n).clip(300, 850).astype(int)

    # Income: skewed distribution $30k–$200k
    income = np.random.lognormal(11.0, 0.5, n).clip(25000, 300000)

    # Employment years: exponential decay, most <10 years
    employment_years = np.random.exponential(5, n).clip(0, 40)

    # Debt-to-income: 0.1 to 0.6
    debt_to_income = np.random.beta(2, 8, n) * 0.5 + 0.1

    # Loan amount: correlated with income, $50k–$500k
    loan_amount = income * (np.random.uniform(0.5, 2.5, n) * 0.3).clip(50000, 800000)

    # Interest rate: higher for lower credit scores
    base_rate = 0.04
    credit_penalty = (700 - credit_score) / 100 * 0.015
    interest_rate = base_rate + credit_penalty.clip(0, 0.20) + np.random.normal(0, 0.01, n)
    interest_rate = interest_rate.clip(0.03, 0.25)

    # LTV ratio: loan / collateral value
    # Add random variation to make it actually vary
    collateral = income * 2
    loan_to_income = np.random.uniform(0.3, 2.5, n)  # loan as multiple of income
    loan_amount = income * loan_to_income.clip(0.3, 2.5)
    LTV_ratio = (loan_amount / collateral).clip(0.2, 2.0)

    # Combine into base features
    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': LTV_ratio
    })

    # Generate time-to-default using a hazard model
    # Base hazard varies by credit score band
    credit_band = pd.cut(credit_score, bins=[0, 580, 670, 740, 850],
                         labels=['<580', '580-669', '670-739', '740+'])

    # Monthly hazard multipliers by credit band
    hazard_multiplier = {'<580': 3.0, '580-669': 1.8, '670-739': 1.0, '740+': 0.5}

    # Monthly default probability (base)
    monthly_base_default = 0.002

    # Feature-based hazard adjustment
    hr_income = np.exp(-df['income'] / 200000)  # higher income = lower hazard
    hr_dti = np.exp(df['debt_to_income'] * 3)     # higher DTI = higher hazard
    hr_ltv = np.exp(df['LTV_ratio'] * 1.5)       # higher LTV = higher hazard
    hr_rate = np.exp((df['interest_rate'] - 0.08) * 10)  # higher rate = higher hazard

    # Composite log-hazard
    log_hazard = (
        np.log(monthly_base_default)
        + np.log(pd.Series(credit_band.astype(str)).map(hazard_multiplier).values)
        + hr_income * 0.5
        + hr_dti * 0.3
        + hr_ltv * 0.3
        + hr_rate * 0.2
        + np.random.normal(0, 0.3, n)
    )

    monthly_default_prob = np.exp(log_hazard).clip(0.0005, 0.05)

    # Simulate time-to-default
    time_start = np.zeros(n, dtype=int)
    time_end = np.zeros(n, dtype=int)
    event_default = np.zeros(n, dtype=int)

    censoring_time = 24  # censor at 24 months

    for i in range(n):
        p = monthly_default_prob[i]
        # Simulate survival months using geometric distribution
        survival_months = 0
        while survival_months < censoring_time:
            if np.random.random() < p:
                break
            survival_months += 1

        if survival_months < censoring_time:
            time_end[i] = survival_months + np.random.randint(0, 3)
            event_default[i] = 1
        else:
            time_end[i] = censoring_time + np.random.randint(0, 2)
            event_default[i] = 0  # censored

    df['time_start'] = time_start
    df['time_end'] = time_end.astype(int)
    df['event_default'] = event_default.astype(int)
    
    # Drop any NaN rows
    df = df.dropna()

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


if __name__ == '__main__':
    df = generate_loan_data(5000)
    save_data(df, 'data/loan_data.csv')
    print(f"Generated {len(df)} rows")
    print(f"Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")
    print(f"Defaults: {(df['event_default']==1).sum()} ({(df['event_default']==1).mean()*100:.1f}%)")
    print(df.describe())