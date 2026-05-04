"""Cox Proportional Hazards model for identifying default risk factors."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def fit_cox_ph_model(df):
    """Fit Cox PH model to identify risk factors for default.

    Args:
        df: DataFrame with survival data and covariates

    Returns:
        CoxPHFitter fitted model and summary DataFrame
    """
    # Prepare covariates
    X = df[['income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']].copy()

    # Standardize for better coefficient interpretation
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    duration = df['time_end']
    event = df['event_default']

    cph = CoxPHFitter()
    cph.fit(pd.concat([X, duration, event], axis=1).rename(columns={'time_end': 'duration', 'event_default': 'event'}),
            duration_col='duration', event_col='event')

    return cph

def get_hazard_ratios(cph):
    """Extract hazard ratios and confidence intervals from fitted model."""
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['hr_lower'] = np.exp(summary['coef lower 95%'])
    summary['hr_upper'] = np.exp(summary['coef upper 95%'])

    return summary[['coef', 'hazard_ratio', 'hr_lower', 'hr_upper', 'p']]

def interpret_coefficients(summary):
    """Interpret Cox PH coefficients in business terms."""
    interpretations = {
        'credit_score': 'Each 1-SD decrease in credit score increases hazard by {hr:.1%}',
        'interest_rate': 'Each 1-SD increase in interest rate increases hazard by {hr:.1%}',
        'debt_to_income': 'Each 1-SD increase in DTI ratio increases hazard by {hr:.1%}',
        'LTV_ratio': 'Each 1-SD increase in LTV ratio increases hazard by {hr:.1%}',
        'loan_amount': 'Each 1-SD increase in loan amount increases hazard by {hr:.1%}',
        'income': 'Each 1-SD increase in income decreases hazard by {hr:.1%}',
        'employment_years': 'Each 1-SD increase in employment years decreases hazard by {hr:.1%}'
    }

    results = []
    for var, interp_template in interpretations.items():
        if var in summary.index:
            hr = summary.loc[var, 'hazard_ratio']
            p_val = summary.loc[var, 'p']
            direction = 'decreases' if hr < 1 else 'increases'
            interp = interp_template.format(hr=abs(1 - hr))
            results.append({
                'variable': var,
                'hazard_ratio': round(hr, 3),
                'p_value': round(p_val, 4),
                'significant': p_val < 0.05,
                'interpretation': interp
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    cph = fit_cox_ph_model(df)

    print("Cox PH Model Summary:")
    summary = get_hazard_ratios(cph)
    print(summary.to_string())

    print("\nInterpretation:")
    interp = interpret_coefficients(summary)
    for _, row in interp.iterrows():
        sig = "***" if row['significant'] else ""
        print(f"  {row['variable']}: HR={row['hazard_ratio']} {sig} — {row['interpretation']}")