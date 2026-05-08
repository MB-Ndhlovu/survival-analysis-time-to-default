"""
Predict survival function for a new loan applicant.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter


def train_survival_models(df):
    """
    Train Kaplan-Meier and Cox PH models for prediction.
    Returns (kmf_global, cph, feature_means, feature_stds)
    """
    features = ['credit_score', 'income', 'employment_years',
                'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Fit Kaplan-Meier on all data
    kmf = KaplanMeierFitter()
    kmf.fit(df['time_end'], df['event_default'], label='Overall')

    # Fit Cox PH
    X = df[features].copy()
    feature_means = X.mean()
    feature_stds = X.std()

    for col in features:
        X[col] = (X[col] - feature_means[col]) / feature_stds[col]

    cox_df = pd.DataFrame({
        'duration': df['time_end'],
        'event': df['event_default'],
        **X
    })

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration', event_col='event')

    return kmf, cph, feature_means, feature_stds


def predict_new_applicant(applicant, kmf, cph, feature_means, feature_stds, features=None):
    """
    Predict survival curve for a new applicant.

    Parameters:
    - applicant: dict with keys {credit_score, income, employment_years,
                                 debt_to_income, loan_amount, interest_rate, LTV_ratio}
    - kmf: trained KaplanMeierFitter
    - cph: trained CoxPHFitter
    - feature_means, feature_stds: normalization params
    """
    if features is None:
        features = ['credit_score', 'income', 'employment_years',
                    'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Normalize applicant features
    normalized = {}
    for f in features:
        val = applicant.get(f, 0)
        normalized[f] = (val - feature_means[f]) / feature_stds[f]

    # Build DataFrame for prediction
    pred_df = pd.DataFrame([normalized])

    # Predict conditional survival using Cox PH
    try:
        surv_func = cph.predict_survival_function(pred_df)
        times = surv_func.index
        survival_probs = surv_func.values.flatten()
    except Exception as e:
        print(f"Cox prediction failed: {e}, using baseline KM")
        times = kmf.survival_function_.index[:36]
        survival_probs = kmf.survival_function_.iloc[:36, 0].values

    return {
        'times': times.tolist(),
        'survival_probability': [round(p, 4) for p in survival_probs],
        'applicant': applicant,
    }


def print_prediction(applicant_pred):
    """Print prediction results for an applicant."""
    print("\n" + "="*70)
    print("NEW APPLICANT SURVIVAL PREDICTION")
    print("="*70)

    applicant = applicant_pred['applicant']
    print(f"\nApplicant Profile:")
    print(f"  Credit Score:    {applicant.get('credit_score', 'N/A')}")
    print(f"  Income:          ${applicant.get('income', 0):,}k")
    print(f"  Employment:      {applicant.get('employment_years', 0):.1f} years")
    print(f"  DTI:             {applicant.get('debt_to_income', 0):.1%}")
    print(f"  Loan Amount:     ${applicant.get('loan_amount', 0):,}k")
    print(f"  Interest Rate:   {applicant.get('interest_rate', 0):.2%}")
    print(f"  LTV:             {applicant.get('LTV_ratio', 0):.3f}")

    times = applicant_pred['times']
    probs = applicant_pred['survival_probability']

    print(f"\nPredicted Survival Probabilities:")
    for t in [6, 12, 18, 24, 36]:
        if t < len(times):
            idx = min(max(0, t-1), len(probs)-1)
            print(f"  {t:2d} months: {probs[idx]:.4f} ({probs[idx]*100:.2f}% no default)")

    return applicant_pred


if __name__ == '__main__':
    from data_loader import generate_loan_data

    df = generate_loan_data()
    kmf, cph, means, stds = train_survival_models(df)

    # Example applicant
    new_applicant = {
        'credit_score': 720,
        'income': 85,
        'employment_years': 6.5,
        'debt_to_income': 0.28,
        'loan_amount': 150,
        'interest_rate': 0.08,
        'LTV_ratio': 0.75,
    }

    pred = predict_new_applicant(new_applicant, kmf, cph, means, stds)
    print_prediction(pred)