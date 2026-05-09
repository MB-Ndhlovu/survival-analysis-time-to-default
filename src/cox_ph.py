"""
Cox Proportional Hazards model for time-to-default.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame, time_col: str = 'time_end',
               event_col: str = 'event_default') -> CoxPHFitter:
    """
    Fit Cox Proportional Hazards model to loan data.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data
    time_col : str
        Column containing time to event or censoring
    event_col : str
        Column containing event indicator

    Returns
    -------
    CoxPHFitter
        Fitted Cox PH model
    """
    features = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'LTV_ratio']

    model_df = df[[time_col, event_col] + features].copy()

    # Log-transform wide-range variables
    model_df['log_income'] = np.log(model_df['income'])
    model_df['log_loan_amount'] = np.log(model_df['loan_amount'])
    model_df = model_df.drop(columns=['income', 'loan_amount'])
    model_df = model_df.rename(columns={'log_income': 'income', 'log_loan_amount': 'loan_amount'})

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col=time_col, event_col=event_col)

    return cph


def print_cox_summary(cph: CoxPHFitter):
    """Print formatted Cox PH summary."""
    print("\n=== Cox Proportional Hazards Model ===\n")
    cph.print_summary()

    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])

    print("\n=== Hazard Ratios (HR > 1 means higher default risk) ===\n")
    for idx, row in summary.iterrows():
        var = idx
        hr = row['hazard_ratio']
        ci_lower = np.exp(row['coef lower 95%'])
        ci_upper = np.exp(row['coef upper 95%'])
        pval = row['p']

        sig = ''
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'

        print(f"{var:20s}: HR = {hr:.4f}  [{ci_lower:.4f}, {ci_upper:.4f}]  p = {pval:.4f} {sig}")

    print("\nInterpretation:")
    print("- HR > 1: Increases default hazard (higher risk)")
    print("- HR < 1: Decreases default hazard (lower risk)")
    print("- HR = 1: No effect")
    print("- Significance: *** p<0.001, ** p<0.01, * p<0.05")


def get_hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    """Extract hazard ratios with confidence intervals."""
    summary = cph.summary.copy()
    hr_df = pd.DataFrame({
        'coefficient': summary['coef'],
        'hazard_ratio': np.exp(summary['coef']),
        'ci_lower': np.exp(summary['coef lower 95%']),
        'ci_upper': np.exp(summary['coef upper 95%']),
        'p_value': summary['p']
    })
    return hr_df


def main(df: pd.DataFrame):
    cph = fit_cox_ph(df)
    print_cox_summary(cph)
    return cph


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    main(df)