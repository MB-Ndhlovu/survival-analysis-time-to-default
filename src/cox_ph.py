"""Cox Proportional Hazards model for time-to-default."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def fit_cox_ph(df):
    """
    Fit Cox PH model and return coefficients with hazard ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Data from data_loader.py

    Returns
    -------
    dict
        Model coefficients, hazard ratios, and interpretation
    """
    # Prepare features — drop non-features
    feature_cols = ['income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    df_model = df[feature_cols + ['time_end', 'event_default']].copy()

    # Standardize for model stability
    for col in feature_cols:
        df_model[col] = (df_model[col] - df_model[col].mean()) / df_model[col].std()

    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='time_end', event_col='event_default')

    # Extract coefficients
    coef_df = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].copy()
    coef_df.columns = ['coefficient', 'hazard_ratio', 'std_error', 'p_value']

    # Interpret coefficients
    interpretations = {}
    for var in feature_cols:
        hr = coef_df.loc[var, 'hazard_ratio']
        p = coef_df.loc[var, 'p_value']
        direction = "increases" if hr > 1 else "decreases"
        interpretations[var] = {
            'hazard_ratio': round(hr, 4),
            'p_value': round(p, 4),
            'significant': p < 0.05,
            'interpretation': (
                f"A 1 SD increase in {var} {direction} default risk by "
                f"{abs(hr - 1) * 100:.1f}% (HR={hr:.3f})"
            )
        }

    results = {
        'concordance_index': round(cph.concordance_index_, 4),
        'coefficients': coef_df.to_dict(),
        'interpretations': interpretations,
        'log_likelihood': round(cph.log_likelihood_, 4),
    }

    return results

if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = fit_cox_ph(df)
    print(f"C-index: {results['concordance_index']}")
    print("\nHazard Ratios (sorted by significance):")
    sorted_vars = sorted(results['interpretations'].items(),
                        key=lambda x: x[1]['p_value'])
    for var, info in sorted_vars:
        sig = "***" if info['p_value'] < 0.001 else "**" if info['p_value'] < 0.01 else "*" if info['p_value'] < 0.05 else ""
        print(f"  {var}: HR={info['hazard_ratio']:.4f} p={info['p_value']:.4f} {sig}")