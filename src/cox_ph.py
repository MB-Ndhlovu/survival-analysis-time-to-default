"""
Cox Proportional Hazards model for time-to-default.

Fits a Cox PH regression to identify which factors increase or decrease
the hazard (instantaneous risk) of default.

Outputs:
- Coefficients and standard errors
- Hazard ratios (exp(coef)) with 95% CI
- Concordance index (model discrimination)
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame) -> dict:
    """
    Fit Cox Proportional Hazards model on loan covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with time_end, event_default, and covariate columns.

    Returns
    -------
    dict
        Results dictionary with cph model, summary DataFrame, and interpretation.
    """
    # Prepare features
    feature_cols = ['income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Normalize features to improve Cox PH convergence
    df = df.copy()
    df['income_100k'] = df['income'] / 100000
    df['loan_amount_m'] = df['loan_amount'] / 1000000

    # Standardize credit score: 0 = mean, 1 = 1 std (100 pts)
    cs_mean, cs_std = df['credit_score'].mean(), df['credit_score'].std()
    df['credit_score_z'] = (df['credit_score'] - cs_mean) / cs_std

    feature_cols_model = ['income_100k', 'credit_score_z', 'employment_years',
                          'debt_to_income', 'loan_amount_m', 'interest_rate', 'LTV_ratio']

    # Drop rows with missing values
    df_model = df[feature_cols_model + ['time_end', 'event_default']].dropna()

    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='time_end', event_col='event_default')

    # Extract summary
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])

    # For interpretability: convert credit_score_z back to per-point HR
    # HR per 100 points = exp(coef * 100)
    cs_coef = summary.loc['credit_score_z', 'coef']
    cs_hr_per_100 = np.exp(cs_coef * 100)
    summary.loc['credit_score_z', 'hazard_ratio'] = cs_hr_per_100

    summary['hr_lower'] = np.exp(summary['coef lower 95%'])
    summary['hr_upper'] = np.exp(summary['coef upper 95%'])

    # Concordance index (model discrimination)
    concordance = cph.concordance_index_

    results = {
        'cph': cph,
        'summary': summary,
        'concordance': concordance,
        'feature_cols': feature_cols_model
    }

    return results


def print_cox_summary(results: dict) -> None:
    """Print Cox PH results with hazard ratio interpretation."""
    summary = results['summary']

    print("\n" + "=" * 80)
    print("COX PROPORTIONAL HAZARDS MODEL - HAZARD RATIOS")
    print("=" * 80)
    print(f"Concordance Index: {results['concordance']:.4f}  (1.0 = perfect, 0.5 = random)")
    print("-" * 80)
    print(f"{'Variable':<22} {'Coef':>8} {'HR':>8} {'95% CI':>20} {'Interpretation'}")
    print("-" * 80)

    interpretations = {
        'income_100k': "Per R100k increase",
        'credit_score': "Per point increase",
        'employment_years': "Per year additional",
        'debt_to_income': "Per 1% DTI increase",
        'loan_amount_m': "Per R1M increase",
        'interest_rate': "Per 1% rate increase",
        'LTV_ratio': "Per 1% LTV increase"
    }

    for var in summary.index:
        row = summary.loc[var]
        hr = row['hazard_ratio']
        ci = f"[{row['hr_lower']:.3f}, {row['hr_upper']:.3f}]"
        p = row['p']

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""

        # Interpret HR
        if hr > 1:
            direction = "↑ risk"
        else:
            direction = "↓ risk"

        label = interpretations.get(var, var)
        print(f"{var:<22} {row['coef']:>8.4f} {hr:>8.3f} {ci:>20} {label} {direction} {sig}")

    print("-" * 80)
    print("Signif. codes: *** p<0.001, ** p<0.01, * p<0.05")
    print("HR > 1 means higher hazard (faster default)")
    print("HR < 1 means lower hazard (slower default)")
    print("=" * 80)

    # Top risk factors
    print("\nKEY INSIGHTS:")
    hr_series = summary['hazard_ratio'].sort_values(ascending=False)
    print("  Largest default accelerators (HR > 1):")
    for var in hr_series.index[:3]:
        if hr_series[var] > 1:
            print(f"    - {var}: HR = {hr_series[var]:.3f}")

    print("\n  Largest default decelerators (HR < 1):")
    for var in hr_series.index[-3:]:
        if hr_series[var] < 1:
            print(f"    - {var}: HR = {hr_series[var]:.3f}")


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_cox_ph(df)
    print_cox_summary(results)