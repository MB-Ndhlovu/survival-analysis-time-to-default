"""
Cox Proportional Hazards model for time-to-default.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats


def fit_cox_ph(df, features=None):
    """
    Fit Cox Proportional Hazards model.

    Features:
    - credit_score: FICO score
    - income: annual income
    - employment_years: years employed
    - debt_to_income: DTI ratio
    - loan_amount: principal
    - interest_rate: annual rate
    - LTV_ratio: loan-to-value
    """
    if features is None:
        features = ['credit_score', 'income', 'employment_years',
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Prepare data
    X = df[features].copy().astype(float)

    # Standardize for better convergence
    for col in features:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    cox_df = pd.DataFrame({
        'duration': df['time_end'],
        'event': df['event_default'],
        **X
    })

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration', event_col='event')

    return cph


def print_cox_summary(cph):
    """Print formatted Cox PH results."""
    print("\n" + "="*70)
    print("COX PROPORTIONAL HAZARDS MODEL")
    print("="*70)

    # Use print_summary method
    print(cph.print_summary(decimals=4))

    # Interpretation
    print("\n--- HAZARD RATIO INTERPRETATION ---")
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])

    for _, row in summary.iterrows():
        var = row.name
        hr = row['hazard_ratio']
        p = row['p']

        if hr > 1:
            direction = "increases"
            pct = (hr - 1) * 100
        else:
            direction = "decreases"
            pct = (1 - hr) * 100

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {var:20s}: HR={hr:.4f} ({pct:.1f}% {direction} risk) {sig}")


def get_hazard_ratios(cph):
    """Extract hazard ratios as dict."""
    summary = cph.summary.copy()
    hazard_ratios = {
        var: {
            'coef': row['coef'],
            'hazard_ratio': row['exp(coef)'],
            'se': row['se(coef)'],
            'z': row['z'],
            'p': row['p'],
        }
        for var, row in summary.iterrows()
    }
    return hazard_ratios


def top_default_drivers(hazard_ratios, n=5):
    """Return top N variables by hazard ratio magnitude."""
    sorted_vars = sorted(
        hazard_ratios.items(),
        key=lambda x: abs(np.log(x[1]['hazard_ratio'])),
        reverse=True
    )
    return sorted_vars[:n]


if __name__ == '__main__':
    from data_loader import generate_loan_data

    df = generate_loan_data()
    cph = fit_cox_ph(df)
    print_cox_summary(cph)