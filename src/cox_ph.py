"""
Cox Proportional Hazards model for loan default.
Identifies which factors accelerate or delay default.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df):
    """
    Fit Cox Proportional Hazards model.
    
    Features: income, credit_score, employment_years, debt_to_income,
              loan_amount, interest_rate, LTV_ratio
    
    Returns fitted CoxPHFitter and summary DataFrame.
    """
    # Prepare features — standardize for interpretability
    feature_cols = [
        'income', 'credit_score', 'employment_years',
        'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio'
    ]
    
    cox_df = df[feature_cols + ['time_end', 'event_default']].copy()
    
    # Log-transform skewed features
    cox_df['log_income'] = np.log(cox_df['income'])
    cox_df['log_loan_amount'] = np.log(cox_df['loan_amount'])
    
    # Drop original skewed columns
    cox_df = cox_df.drop(columns=['income', 'loan_amount'])
    
    # Rename for cleaner output
    cox_df = cox_df.rename(columns={
        'log_income': 'income',
        'log_loan_amount': 'loan_amount'
    })
    
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='time_end', event_col='event_default')
    
    return cph


def print_cox_results(cph):
    """Print formatted Cox PH results."""
    print("\n" + "="*70)
    print("COX PROPORTIONAL HAZARDS MODEL — HAZARD RATIOS")
    print("="*70)
    print("\nHazard Ratio interpretation:")
    print("  HR > 1 : factor increases default hazard (risk)")
    print("  HR < 1 : factor decreases default hazard (protective)")
    print("  HR = 1 : no effect")
    print("\n  e.g. HR = 1.5 → 50% higher instantaneous default rate")
    print("  e.g. HR = 0.7 → 30% lower instantaneous default rate")
    print()
    
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['hr_interpretation'] = summary.apply(
        lambda r: 'increases risk' if r['hazard_ratio'] > 1 else 'decreases risk',
        axis=1
    )
    
    print(f"{'Variable':<25} {'Coef':>8} {'Exp(Coef)':>10} {'p-value':>10} {'Effect':>15}")
    print("-"*70)
    
    for idx, row in summary.iterrows():
        var = idx.replace('_', ' ').title()
        coef = row['coef']
        hr = row['hazard_ratio']
        pval = row['p']
        effect = row['hr_interpretation']
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        print(f"{var:<25} {coef:>8.3f} {hr:>10.3f} {pval:>9.3f}{sig}   {effect:>15}")
    
    print("-"*70)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FROM COX PH MODEL")
    print("="*70)
    
    # Sort by absolute coefficient magnitude
    summary_abs = summary.copy()
    summary_abs['abs_coef'] = summary_abs['coef'].abs()
    summary_abs = summary_abs.sort_values('abs_coef', ascending=False)
    
    print("\nTop drivers of default risk (by coefficient magnitude):")
    for i, (idx, row) in enumerate(summary_abs.iterrows(), 1):
        direction = "↑ increases" if row['coef'] > 0 else "↓ decreases"
        var = idx.replace('_', ' ').title()
        print(f"  {i}. {var}: {direction} hazard by {abs(row['hazard_ratio']-1):.1%} per unit")
    
    print("\nBusiness implications:")
    
    # Credit score effect
    cs_hr = summary.loc['credit_score', 'hazard_ratio']
    if cs_hr < 1:
        print(f"  - Credit score is PROTECTIVE: HR={cs_hr:.3f}")
        print(f"    Each 100-point increase reduces hazard by {(1-cs_hr)*100:.1f}%")
    
    # Interest rate effect
    ir_hr = summary.loc['interest_rate', 'hazard_ratio']
    if ir_hr > 1:
        print(f"  - Higher interest rate INCREASES default hazard: HR={ir_hr:.3f}")
        print(f"    Each 1% rate increase raises hazard by {(ir_hr-1)*100:.0f}%")
    
    # DTI effect
    dti_hr = summary.loc['debt_to_income', 'hazard_ratio']
    if dti_hr > 1:
        print(f"  - Debt-to-income ratio increases hazard: HR={dti_hr:.3f}")
        print(f"    High DTI borrowers are riskier")
    
    # LTV effect
    ltv_hr = summary.loc['LTV_ratio', 'hazard_ratio']
    if ltv_hr > 1:
        print(f"  - LTV ratio increases hazard: HR={ltv_hr:.3f}")
        print(f"    High LTV loans have elevated default risk")
    
    print("="*70)


def get_hazard_ratio_table(cph):
    """Return clean hazard ratio DataFrame."""
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['lower_ci'] = np.exp(summary['coef lower 95%'])
    summary['upper_ci'] = np.exp(summary['coef upper 95%'])
    summary = summary[['coef', 'hazard_ratio', 'lower_ci', 'upper_ci', 'p']]
    summary.columns = ['coefficient', 'hazard_ratio', 'ci_lower', 'ci_upper', 'p_value']
    return summary.reset_index().rename(columns={'index': 'variable'})


def concordance_index(cph):
    """Return the concordance index — model discrimination ability."""
    return cph.concordance_index_


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands
    
    df = generate_loan_data()
    df = add_credit_bands(df)
    
    cph = fit_cox_ph(df)
    print_cox_results(cph)
    
    hr_table = get_hazard_ratio_table(cph)
    print("\n\nHazard Ratio Table:")
    print(hr_table.to_string(index=False))
    
    c_index = concordance_index(cph)
    print(f"\nConcordance Index: {c_index:.4f}")
    print("(0.5 = random, 1.0 = perfect discrimination)")