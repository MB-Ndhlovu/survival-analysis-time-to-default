"""
Cox Proportional Hazards model for identifying default risk factors.
Semi-parametric approach estimates hazard ratios for each covariate.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df, duration_col='time_end', event_col='event_default'):
    """
    Fit Cox PH model to identify factors affecting default hazard.

    Returns fitted model and coefficient summary.
    """
    # Prepare features
    feature_cols = ['income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Normalize continuous variables for better interpretation
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(pd.concat([X, df[[duration_col, event_col]]], axis=1),
            duration_col=duration_col,
            event_col=event_col)

    return cph


def print_cox_summary(cph):
    """Print formatted Cox PH results."""
    print("\n" + "="*70)
    print("COX PROPORTIONAL HAZARDS MODEL")
    print("="*70)
    print("\nCoefficients and Hazard Ratios:")

    # Get summary DataFrame
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['hr_lower'] = np.exp(summary['coef lower 95%'])
    summary['hr_upper'] = np.exp(summary['coef upper 95%'])

    print(f"\n{'Variable':<20} {'Coef':>8} {'HR':>8} {'95% CI':>15} {'p-value':>10}")
    print("-" * 65)

    for idx, row in summary.iterrows():
        ci = f"[{row['hr_lower']:.2f}, {row['hr_upper']:.2f}]"
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        print(f"{idx:<20} {row['coef']:>8.3f} {row['hazard_ratio']:>8.3f} {ci:>15} {row['p']:>10.4f} {sig}")

    print("\nInterpretation:")
    print("  HR > 1: Increases hazard (faster default)")
    print("  HR < 1: Decreases hazard (slower default)")
    print("  HR = 1: No effect")

    # Concordance index
    print(f"\nModel Concordance Index: {cph.concordance_index_:.4f}")
    print("(0.5 = random, 0.7+ = good, 0.8+ = excellent fit)")

    return summary


def get_top_risk_factors(summary, n=3):
    """Return the top n risk factors by hazard ratio."""
    sorted_hr = summary.sort_values('hazard_ratio', ascending=False)
    top_risks = []

    for idx, row in sorted_hr.head(n).iterrows():
        top_risks.append({
            'variable': idx,
            'hazard_ratio': row['hazard_ratio'],
            'interpretation': f"1 SD increase → {row['hazard_ratio']:.2f}x default risk"
        })

    return top_risks


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    cph = fit_cox_ph(df)
    summary = print_cox_summary(cph)
    top_risks = get_top_risk_factors(summary)

    print("\nTop Risk Factors:")
    for i, risk in enumerate(top_risks, 1):
        print(f"  {i}. {risk['variable']}: {risk['interpretation']}")