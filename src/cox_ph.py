"""Cox Proportional Hazards model for credit risk factors."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from src.data_loader import generate_loan_data


def fit_cox_ph_model(df, time_col='time_end', event_col='event_default'):
    """Fit Cox PH model and return results."""
    df_model = df.copy()

    # Select features for Cox model
    features = ['credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'LTV_ratio']

    # Drop rows with missing values
    df_model = df_model.dropna(subset=features + [time_col, event_col])

    # Standardize for better interpretation
    for col in features:
        df_model[f'{col}_scaled'] = (df_model[col] - df_model[col].mean()) / df_model[col].std()

    scaled_features = [f'{col}_scaled' for col in features]

    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(df_model[scaled_features + [time_col, event_col]],
            duration_col=time_col, event_col=event_col)

    # Extract coefficients
    coef_df = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']].copy()
    coef_df.columns = ['coefficient', 'hazard_ratio', 'std_error', 'z_stat', 'p_value']

    # Add feature names
    coef_df['feature'] = features
    coef_df = coef_df[['feature', 'coefficient', 'hazard_ratio', 'std_error', 'z_stat', 'p_value']]

    # Significance stars
    def significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        elif p < 0.1:
            return '.'
        return ''

    coef_df['significance'] = coef_df['p_value'].apply(significance_stars)

    return {
        'model': cph,
        'coefficients': coef_df,
        'concordance_index': cph.concordance_index_,
        'log_likelihood': cph.log_likelihood_,
        'AIC': cph.AIC_partial_,
    }


def interpret_coefficients(results):
    """Interpret Cox PH results."""
    print("\nCox Proportional Hazards Model Results")
    print("=" * 70)
    print(f"Concordance Index: {results['concordance_index']:.4f}")
    print(f"Log-Likelihood: {results['log_likelihood']:.2f}")
    print(f"AIC: {results['AIC']:.2f}")
    print("\nCoefficients:")
    print("-" * 70)
    print(f"{'Feature':<20} {'Coef':>8} {'HR':>8} {'StdErr':>8} {'z':>8} {'P>|z|':>8} {'Sig':>4}")
    print("-" * 70)

    for _, row in results['coefficients'].iterrows():
        print(f"{row['feature']:<20} {row['coefficient']:>8.4f} {row['hazard_ratio']:>8.4f} "
              f"{row['std_error']:>8.4f} {row['z_stat']:>8.3f} {row['p_value']:>8.4f} {row['significance']:>4}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("-" * 70)
    for _, row in results['coefficients'].iterrows():
        hr = row['hazard_ratio']
        feature = row['feature']
        if hr > 1:
            effect = f"increases"
            magnitude = f"{((hr - 1) * 100):.1f}% higher"
        else:
            effect = f"decreases"
            magnitude = f"{((1 - hr) * 100):.1f}% lower"

        print(f"- {feature}: HR={hr:.4f} → {effect} default risk by {magnitude} per 1 SD increase")
    print("=" * 70)

    return results


if __name__ == '__main__':
    df = generate_loan_data()
    results = fit_cox_ph_model(df)
    interpret_coefficients(results)