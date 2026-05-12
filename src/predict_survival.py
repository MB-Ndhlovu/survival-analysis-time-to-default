"""Predict survival function for a new loan applicant using the fitted Cox PH model."""

import numpy as np

from lifelines import KaplanMeierFitter


def predict_survival(cph, new_applicant, kmf_reference=None):
    """Predict survival curve for a new applicant using Cox PH model.

    Args:
        cph: fitted CoxPHFitter
        new_applicant: dict with keys:
            credit_score, debt_to_income, LTV_ratio,
            interest_rate, employment_years, income, loan_amount
    Returns:
        dict with predicted survival probabilities at key time horizons
    """
    df = np.array([[
        new_applicant["credit_score"],
        new_applicant["debt_to_income"],
        new_applicant["LTV_ratio"],
        new_applicant["interest_rate"],
        new_applicant["employment_years"],
    ]])
    col_names = ["credit_score", "debt_to_income", "LTV_ratio",
                 "interest_rate", "employment_years"]

    # median baseline_hazard is derived from the model's baseline hazard
    # Use conditional_with_covariates for prediction
    survival_probs = cph.predict_survival_function(df)

    # Interpolate at key time points
    times = [6, 12, 18, 24, 36]
    result = {"time_horizons_months": times}
    for t in times:
        idx = (survival_probs.index.to_numpy() - t).astype(float)
        idx = np.abs(idx).argmin()
        result[f"survival_at_{t}m"] = round(float(survival_probs.iloc[idx].values[0]), 4)

    return result


def demo_prediction(cph, data):
    """Run a demo prediction for a new high-risk and low-risk applicant."""
    print("\n=== New Applicant Predictions ===")

    # High-risk applicant: low credit score, high DTI, high LTV
    high_risk = {
        "credit_score": 520,
        "debt_to_income": 0.48,
        "LTV_ratio": 0.95,
        "interest_rate": 18.5,
        "employment_years": 1.0,
        "income": 35000,
        "loan_amount": 120000,
    }

    # Low-risk applicant: high credit score, low DTI, low LTV
    low_risk = {
        "credit_score": 780,
        "debt_to_income": 0.15,
        "LTV_ratio": 0.55,
        "interest_rate": 6.5,
        "employment_years": 10.0,
        "income": 90000,
        "loan_amount": 200000,
    }

    for label, app in [("High-Risk Applicant", high_risk), ("Low-Risk Applicant", low_risk)]:
        pred = predict_survival(cph, app)
        print(f"\n{label}:")
        for k, v in pred.items():
            if k != "time_horizons_months":
                print(f"  {k}: {v}")

    return {"high_risk": predict_survival(cph, high_risk),
            "low_risk": predict_survival(cph, low_risk)}