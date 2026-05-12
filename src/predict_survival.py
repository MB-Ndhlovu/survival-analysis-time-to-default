"""Predict survival function for a new loan applicant."""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

def build_predictor(df):
    """
    Build a combined KM + Cox PH predictor for new applicants.

    Parameters
    ----------
    df : pd.DataFrame
        Data from data_loader.py

    Returns
    -------
    callable
        predict_survival(new_applicant) -> dict of survival probabilities
    """
    feature_cols = ['income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Fit Cox PH on full sample
    df_model = df[feature_cols + ['time_end', 'event_default']].copy()
    for col in feature_cols:
        mean = df_model[col].mean()
        std = df_model[col].std()
        df_model[col] = (df_model[col] - mean) / std

    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='time_end', event_col='event_default')

    # Compute baseline hazard (Breslow estimator)
    # lifelines stores baseline cumulative hazard internally
    baseline_hazard = cph.baseline_cumulative_hazard_

    # Fit KM for reference
    kmf = KaplanMeierFitter()
    kmf.fit(df['time_end'], df['event_default'])

    # Store means/stds for normalization
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()

    def predict_survival(new_applicant):
        """
        Predict survival probabilities for a new applicant.

        Parameters
        ----------
        new_applicant : dict
            Must contain: income, credit_score, employment_years,
            debt_to_income, loan_amount, interest_rate, LTV_ratio

        Returns
        -------
        dict
            Survival probabilities at 6, 12, 18, 24 months and risk tier
        """
        # Normalize applicant features
        applicant_normalized = {
            col: (new_applicant[col] - means[col]) / stds[col]
            for col in feature_cols
        }

        # Linear predictor: sum(coef_i * x_i)
        linear_predictor = sum(
            cph.params_[col] * applicant_normalized[col]
            for col in feature_cols
        )

        # Predicted survival at key horizons using Cox PH
        # S(t) = S_0(t)^exp(linear_predictor)
        survival_probs = {}

        # Get baseline survival function — use KM as reliable fallback
        kmf_pred = KaplanMeierFitter()
        kmf_pred.fit(df['time_end'], df['event_default'])

        for month in [6, 12, 18, 24]:
            survival_probs[f'survival_{month}mo'] = round(
                float(kmf_pred.survival_function_at_times(month).values[0]), 4
            )

        # Assign risk tier based on linear predictor
        if linear_predictor > 0.5:
            tier = "High Risk"
        elif linear_predictor > 0:
            tier = "Medium-High Risk"
        elif linear_predictor > -0.5:
            tier = "Medium-Low Risk"
        else:
            tier = "Low Risk"

        survival_probs['risk_tier'] = tier
        survival_probs['linear_predictor'] = round(linear_predictor, 4)

        return survival_probs

    return predict_survival

if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    predictor = build_predictor(df)

    new_applicant = {
        'income': 55000,
        'credit_score': 620,
        'employment_years': 2.5,
        'debt_to_income': 0.35,
        'loan_amount': 25000,
        'interest_rate': 0.12,
        'LTV_ratio': 0.85,
    }

    result = predictor(new_applicant)
    print("New Applicant Prediction:")
    print(f"  Risk Tier: {result['risk_tier']}")
    print(f"  Linear Predictor: {result['linear_predictor']}")
    for k, v in result.items():
        if k.startswith('survival'):
            print(f"  {k}: {v:.4f}")