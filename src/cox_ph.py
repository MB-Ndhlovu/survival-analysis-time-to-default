"""Cox Proportional Hazards model for credit default risk."""

import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame) -> tuple:
    """Fit Cox Proportional Hazards model on loan features.

    Features: credit_score, employment_years, debt_to_income, loan_amount,
              interest_rate, LTV_ratio, income

    Returns dict with coefficients, hazard ratios, and model summary.
    """
    cph = CoxPHFitter()
    
    features = [
        'credit_score', 'employment_years', 'debt_to_income',
        'loan_amount', 'interest_rate', 'LTV_ratio', 'income'
    ]

    # Prepare data - Cox requires positive duration and binary event
    cox_df = df[['time_end', 'event_default'] + features].copy()
    
    # Standardize income and loan_amount for better interpretation
    cox_df['log_income'] = np.log(cox_df['income'])
    cox_df['log_loan_amount'] = np.log(cox_df['loan_amount'])
    
    fit_features = [
        'credit_score', 'employment_years', 'debt_to_income',
        'log_loan_amount', 'interest_rate', 'LTV_ratio', 'log_income'
    ]

    cph.fit(cox_df, duration_col='time_end', event_col='event_default')

    # Extract results
    summary = cph.summary.copy()
    summary.columns = [c.replace(' ', '_').replace('(', '_').replace(')', '') for c in summary.columns]
    summary['hazard_ratio'] = np.exp(summary['coef'])
    
    results = {
        'concordance_index': round(float(cph.concordance_index_), 4),
        'log_likelihood': round(float(cph.log_likelihood_), 4),
        'AIC': round(float(cph.AIC_partial_), 4),
        'coefficients': {},
        'hazard_ratios': {},
    }

    print("\n=== Cox Proportional Hazards Results ===")
    print(f"Concordance Index: {results['concordance_index']:.4f}")
    print(f"Log-Likelihood: {results['log_likelihood']:.4f}")
    print(f"AIC (partial): {results['AIC']:.4f}")
    print("\n--- Coefficients & Hazard Ratios ---")
    print(f"{'Variable':<22} {'Coef':>10} {'Exp(Coef)':>10} {'SE':>8} {'z':>8} {'p-value':>10}")
    print("-" * 72)

    col_map = {'coef': 'coef', 'exp_coef': 'hazard_ratio', 'se_coef': 'se', 'z': 'z', 'p': 'p_value'}

    for var in fit_features:
        row = summary.loc[var]
        hr = row['hazard_ratio']
        coef = row['coef']
        se = row['se_coef']
        z = row['z']
        p = row['p']
        
        results['coefficients'][var] = round(float(coef), 6)
        results['hazard_ratios'][var] = round(float(hr), 4)

        sig = '*' if p < 0.05 else ''
        print(f"{var:<22} {coef:>10.4f} {hr:>10.4f} {se:>8.4f} {z:>8.3f} {p:>10.4f} {sig}")

    print("\n--- Interpretation ---")

    for var in fit_features:
        hr = results['hazard_ratios'][var]
        print(f"\n{var}:")
        if var == 'credit_score':
            print(f"  HR={hr:.4f}: Each 1-point increase in credit score reduces hazard by {(1-hr)*100:.2f}%")
            print(f"  Each 100-point increase reduces hazard by {(1 - hr**100)*100:.1f}%")
        elif var == 'employment_years':
            print(f"  HR={hr:.4f}: Each additional year of employment reduces hazard by {(1-hr)*100:.2f}%")
        elif var == 'debt_to_income':
            print(f"  HR={hr:.4f}: Each 1-unit increase in DTI raises hazard by {(hr-1)*100:.2f}%")
        elif var == 'interest_rate':
            print(f"  HR={hr:.4f}: Each 1% (0.01) increase in rate raises hazard by {(hr-1)*100:.1f}%")
        elif var == 'LTV_ratio':
            print(f"  HR={hr:.4f}: Each 0.1 increase in LTV raises hazard by {(hr-1)*10*100:.1f}%")

    cph.print_summary()
    
    return results, cph


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results, cph = fit_cox_ph(df)
    print("\nFull results:", json.dumps(results, indent=2))