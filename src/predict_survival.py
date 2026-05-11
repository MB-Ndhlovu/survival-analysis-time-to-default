"""
Predict survival: use a fitted Cox PH model to predict
the survival function for a new loan applicant.
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def predict_survival(
    cph: CoxPHFitter,
    new_applicant: dict,
    times: list = None,
) -> pd.Series:
    """
    Predict survival probabilities at each time step for a new applicant.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox PH model.
    new_applicant : dict
        Dictionary of covariate values for the new applicant.
        Keys must match the covariates used in the fitted model.
    times : list of int, optional
        Time horizons to evaluate. Defaults to [1..36].

    Returns
    -------
    pd.Series
        Survival probabilities indexed by time.
    """
    if times is None:
        times = list(range(1, 37))

    df = pd.DataFrame([new_applicant])

    # lifelines predict_survival_function expects DataFrame with correct columns
    surv_func = cph.predict_survival_function(df, times=times)

    return surv_func.squeeze()


def sample_applicant() -> dict:
    """
    Returns a sample applicant with realistic (medium-risk) values.
    """
    return {
        "credit_score":     670,
        "income":           75_000,
        "employment_years": 4.0,
        "debt_to_income":   0.30,
        "loan_amount":      350_000,
        "interest_rate":    13.5,
        "LTV_ratio":        0.80,
    }


if __name__ == "__main__":
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph

    df = generate_loan_data()
    cph = fit_cox_ph(df)

    applicant = sample_applicant()
    surv = predict_survival(cph, applicant, times=list(range(1, 37)))

    print("=== New Applicant Survival Probabilities ===")
    print(f"Applicant: {applicant}")
    print()
    print(f"{'Month':>6} {'S(t)':>8}")
    print("-" * 16)
    for t, s in surv.items():
        print(f"{t:>6} {s:>8.4f}")

    # Key horizons
    print()
    print(f"S(12m) = {surv[12]:.4f}  |  S(24m) = {surv[24]:.4f}  |  S(36m) = {surv[36]:.4f}")