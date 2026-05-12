"""Cox Proportional Hazards model for time-to-default."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def fit_cox_ph(df):
    """
    Fit Cox PH model on loan features.
    Returns fitted model and a results DataFrame.
    """
    # Prepare features
    features = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'LTV_ratio']

    X = df[features + ['time_end', 'event_default']].copy()

    # Standardize for numerical stability
    for col in features:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # Add log-transformed loan_amount
    X['log_loan_amount'] = np.log(df['loan_amount'] + 1)

    duration_col = 'time_end'
    event_col = 'event_default'

    cph = CoxPHFitter()
    cph.fit(X, duration_col=duration_col, event_col=event_col)

    return cph

def print_cox_summary(cph):
    """Print formatted Cox PH results."""
    print("\n=== Cox Proportional Hazards Model ===")
    print(cph.print_summary(decimals=4))

    # Extract key metrics
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['hr_lower'] = np.exp(summary['coef lower 95%'])
    summary['hr_upper'] = np.exp(summary['coef upper 95%'])

    print("\n=== Hazard Ratios (per 1-SD increase) ===")
    for idx, row in summary.iterrows():
        var = idx
        hr = row['hazard_ratio']
        ci = f"[{row['hr_lower']:.3f}, {row['hr_upper']:.3f}]"
        p = row['p']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {var:25s} HR = {hr:.4f}  95% CI: {ci:20s}  p = {p:.4f} {sig}")

    return summary

def top_risk_factors(summary, n=5):
    """Return top n risk factors by hazard ratio magnitude."""
    summary['abs_hr_minus_1'] = np.abs(summary['hazard_ratio'] - 1)
    top = summary.sort_values('abs_hr_minus_1', ascending=False).head(n)
    return top[['coef', 'hazard_ratio', 'p']]