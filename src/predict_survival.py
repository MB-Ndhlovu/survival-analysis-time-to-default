import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def predict_survival(new_applicant, cph, timeline=None):
    """Predict survival function for a new loan applicant.

    Args:
        new_applicant: dict with keys:
            income, credit_score, employment_years,
            debt_to_income, loan_amount, interest_rate, LTV_ratio
        cph: fitted CoxPHFitter
        timeline: months to predict (default 0..36)

    Returns:
        dict with timeline and survival probabilities
    """
    if timeline is None:
        timeline = np.arange(0, 37)

    features = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_amount", "interest_rate", "LTV_ratio"
    ]
    row = pd.DataFrame([new_applicant])[features]
    row["income"] = np.log1p(row["income"])
    row["loan_amount"] = np.log1p(row["loan_amount"])

    sf = cph.predict_survival_function(row, times=timeline)
    probs = sf.values.flatten()

    return {
        "timeline": timeline.tolist(),
        "survival_prob": [float(p) for p in probs],
        "applicant": new_applicant,
    }

def default_applicant():
    """Return a sample new applicant for demonstration."""
    return {
        "income": 55000,
        "credit_score": 710,
        "employment_years": 4.5,
        "debt_to_income": 0.28,
        "loan_amount": 25000,
        "interest_rate": 0.085,
        "LTV_ratio": 0.65,
    }

if __name__ == "__main__":
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph
    df = generate_loan_data()
    result = fit_cox_ph(df)
    applicant = default_applicant()
    pred = predict_survival(applicant, result["cph"])
    print(f"12-month survival: {pred['survival_prob'][12]:.3f}")
    print(f"24-month survival: {pred['survival_prob'][24]:.3f}")