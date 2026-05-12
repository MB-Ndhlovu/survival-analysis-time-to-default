"""Cox Proportional Hazards model for time-to-default prediction."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


def fit_cox_ph(df: pd.DataFrame) -> dict:
    """
    Fit Cox Proportional Hazards model on loan data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Loan data with survival columns
    
    Returns
    -------
    dict
        Model results including coefficients, hazard ratios, and summary
    """
    # Prepare features
    feature_cols = ['credit_score', 'income', 'employment_years', 
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']
    
    # Normalize income and loan_amount for better convergence
    model_df = df.copy()
    model_df['income_k'] = model_df['income'] / 1000  # Income in thousands
    model_df['loan_amount_k'] = model_df['loan_amount'] / 1000  # Loan in thousands
    
    feature_cols_model = ['credit_score', 'income_k', 'employment_years',
                          'debt_to_income', 'loan_amount_k', 'interest_rate', 'LTV_ratio']
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(model_df[['time_end', 'event_default'] + feature_cols_model],
           duration_col='time_end',
           event_col='event_default')
    
    # Extract results
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    
    # Confidence intervals for hazard ratio
    summary['hr_lower'] = np.exp(summary['coef lower 95%'])
    summary['hr_upper'] = np.exp(summary['coef upper 95%'])
    
    # Sort by hazard ratio magnitude
    summary = summary.sort_values('hazard_ratio', ascending=False)
    
    return {
        'model': cph,
        'summary': summary,
        'feature_cols': feature_cols_model,
    }


def print_cox_results(cox_results: dict) -> None:
    """Print Cox PH model results in readable format."""
    cph = cox_results['model']
    summary = cox_results['summary']
    
    print("\n" + "="*70)
    print("COX PROPORTIONAL HAZARDS MODEL RESULTS")
    print("="*70)
    print(f"\nConcordance Index: {cph.concordance_index_:.4f}")
    print(f"Log-Likelihood: {cph.log_likelihood_:.2f}")
    print(f"Number of events: {int(cph.event_observed.sum())}")
    
    print("\n" + "-"*70)
    print(f"{'Variable':<20} {'Coef':>8} {'HR':>8} {'95% CI':>15} {'p-value':>10}")
    print("-"*70)
    
    for idx, row in summary.iterrows():
        ci_str = f"[{row['hr_lower']:.2f}, {row['hr_upper']:.2f}]"
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        print(f"{idx:<20} {row['coef']:>8.4f} {row['hazard_ratio']:>8.3f} {ci_str:>15} {row['p']:>10.4f} {sig}")
    
    print("\n" + "-"*70)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    # Interpretation
    print("\n" + "="*70)
    print("KEY RISK FACTORS (sorted by hazard ratio)")
    print("="*70)
    
    for idx, row in summary.head(5).iterrows():
        hr = row['hazard_ratio']
        if hr > 1:
            direction = "INCREASES"
            pct = (hr - 1) * 100
            print(f"  - {idx}: HR={hr:.3f} → {pct:.1f}% higher risk per unit increase")
        else:
            direction = "DECREASES"
            pct = (1 - hr) * 100
            print(f"  - {idx}: HR={hr:.3f} → {pct:.1f}% lower risk per unit increase")


def get_top_risk_factors(cox_results: dict, n: int = 3) -> list:
    """Return top n risk factors with highest hazard ratios."""
    summary = cox_results['summary']
    top_factors = []
    
    for idx, row in summary.head(n).iterrows():
        top_factors.append({
            'variable': idx,
            'hazard_ratio': round(row['hazard_ratio'], 3),
            'p_value': round(row['p'], 4),
        })
    
    return top_factors


if __name__ == '__main__':
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    results = fit_cox_ph(df)
    print_cox_results(results)