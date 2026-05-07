"""
Cox Proportional Hazards model for time-to-default.
Fits model, interprets coefficients, computes hazard ratios.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df, time_col='time_end', event_col='event_default'):
    """
    Fit Cox Proportional Hazards model on loan features.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time, event, and feature columns
    time_col : str
        Name of time column
    event_col : str
        Name of event indicator (1=default, 0=censored)

    Returns
    -------
    CoxPHFitter
        Fitted Cox model
    """
    # Select features for the model
    feature_cols = [
        'credit_score',
        'employment_years',
        'debt_to_income',
        'loan_amount',
        'interest_rate',
        'LTV_ratio',
    ]

    # Prepare data - add log-transformed features to the dataframe
    df_model = df.copy()
    df_model['log_loan_amount'] = np.log(df_model['loan_amount'] + 1)

    features_for_model = [
        'credit_score',
        'employment_years',
        'debt_to_income',
        'interest_rate',
        'LTV_ratio',
        'log_loan_amount',
    ]

    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(
        df_model[features_for_model + [time_col, event_col]],
        duration_col=time_col,
        event_col=event_col
    )

    return cph


def hazard_ratios_summary(cph):
    """
    Extract hazard ratios and confidence intervals from fitted Cox model.

    Returns
    -------
    pd.DataFrame
        Summary with variable, coef, hazard_ratio, se, ci_lower, ci_upper, p_value
    """
    summary = cph.summary.copy()

    hr_df = pd.DataFrame({
        'variable': summary.index,
        'coefficient': summary['coef'].values,
        'hazard_ratio': summary['exp(coef)'].values,
        'std_error': summary['se(coef)'].values,
        'p_value': summary['p'].values,
    })

    # Calculate 95% CI for hazard ratios
    z = 1.96
    hr_df['ci_lower'] = np.exp(hr_df['coefficient'] - z * hr_df['std_error'])
    hr_df['ci_upper'] = np.exp(hr_df['coefficient'] + z * hr_df['std_error'])

    hr_df = hr_df.round(4)

    return hr_df.sort_values('hazard_ratio', ascending=False)


def top_risk_factors(hr_df, n=5):
    """Return top N risk factors by hazard ratio."""
    return hr_df.head(n)[['variable', 'hazard_ratio', 'ci_lower', 'ci_upper', 'p_value']]


def interpret_coefficient(var_name, coef, hr):
    """Generate human-readable interpretation of a coefficient."""
    if hr > 1:
        direction = 'increases'
        factor = hr
    else:
        direction = 'decreases'
        factor = 1 / hr

    return (
        f"{var_name}: each unit increase {direction} default hazard by "
        f"{factor:.2f}x (HR={hr:.3f})"
    )


def concordance_index(cph):
    """Get model concordance index (C-statistic)."""
    return round(cph.concordance_index_, 4)


if __name__ == '__main__':
    from src.data_loader import generate_loan_data, add_credit_band

    df = generate_loan_data(5000)
    df = add_credit_band(df)

    cph = fit_cox_ph(df)
    print("Cox PH Model Summary")
    print("=" * 50)
    cph.print_summary()

    print("\n\nHazard Ratios:")
    print("-" * 50)
    hr_df = hazard_ratios_summary(cph)
    print(hr_df.to_string(index=False))

    print(f"\nConcordance Index: {concordance_index(cph)}")