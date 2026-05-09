"""
Predict survival function for a new loan applicant using Cox PH model.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def predict_survival_for_applicant(cph: CoxPHFitter, applicant: dict,
                                    months: list = None) -> pd.DataFrame:
    """
    Predict survival curve for a new loan applicant.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox PH model
    applicant : dict
        Dictionary with applicant features:
        {
            'income': float,
            'credit_score': int,
            'employment_years': float,
            'debt_to_income': float,
            'loan_amount': float,
            'interest_rate': float,
            'LTV_ratio': float
        }
    months : list, optional
        List of months at which to predict survival

    Returns
    -------
    pd.DataFrame
        Predicted survival probabilities at each month
    """
    if months is None:
        months = list(range(0, 37))

    # Prepare applicant data in same format as training data
    row = {
        'income': np.log(applicant['income']),
        'credit_score': applicant['credit_score'],
        'employment_years': applicant['employment_years'],
        'debt_to_income': applicant['debt_to_income'],
        'loan_amount': np.log(applicant['loan_amount']),
        'interest_rate': applicant['interest_rate'],
        'LTV_ratio': applicant['LTV_ratio']
    }

    applicant_df = pd.DataFrame([row])

    # Use predict_survival_function for the applicant
    survival_probs = []
    for m in months:
        try:
            S_m = cph.predict_survival_function(applicant_df, times=[m]).iloc[0, 0]
        except Exception:
            S_m = 1.0
        survival_probs.append({
            'month': m,
            'survival_probability': round(float(S_m), 4)
        })

    return pd.DataFrame(survival_probs)


def print_applicant_summary(applicant: dict, survival_df: pd.DataFrame):
    """Print prediction summary for a loan applicant."""
    print("\n=== New Applicant Prediction ===\n")
    print("Applicant Profile:")
    for k, v in applicant.items():
        if k in ['income', 'loan_amount']:
            print(f"  {k}: ZAR {v:,.0f}")
        elif k == 'interest_rate':
            print(f"  {k}: {v*100:.2f}%")
        elif k == 'credit_score':
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print("\nSurvival Probabilities:")
    for _, row in survival_df.iterrows():
        if row['month'] in [6, 12, 18, 24, 36]:
            print(f"  S({row['month']}m): {row['survival_probability']:.2%}")

    # 12-month and 24-month default probabilities
    s12 = survival_df[survival_df['month'] == 12]['survival_probability'].values[0]
    s24 = survival_df[survival_df['month'] == 24]['survival_probability'].values[0]

    print(f"\n  Implied 12-month default probability: {1-s12:.2%}")
    print(f"  Implied 24-month default probability: {1-s24:.2%}")


def main(df: pd.DataFrame):
    from src.cox_ph import fit_cox_ph

    # Fit model on existing data
    cph = fit_cox_ph(df)

    # New applicant example
    applicant = {
        'income': 550000,
        'credit_score': 660,
        'employment_years': 3,
        'debt_to_income': 0.35,
        'loan_amount': 800000,
        'interest_rate': 0.14,
        'LTV_ratio': 0.75
    }

    survival_df = predict_survival_for_applicant(cph, applicant)
    print_applicant_summary(applicant, survival_df)

    return survival_df


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    main(df)