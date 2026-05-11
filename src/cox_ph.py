"""
Cox Proportional Hazards model: fit Cox PH regression,
interpret coefficients, and compute hazard ratios.
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def fit_cox_ph(
    df: pd.DataFrame,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> CoxPHFitter:
    """
    Fit a Cox PH model on loan data.
    Uses covariates: credit_score, income, employment_years,
    debt_to_income, loan_amount, interest_rate, LTV_ratio.
    Returns the fitted CoxPHFitter.
    """
    cph = CoxPHFitter()

    cov_cols = [
        "credit_score",
        "income",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    fit_df = df[[duration_col, event_col] + cov_cols].copy()
    cph.fit(fit_df, duration_col=duration_col, event_col=event_col)
    return cph


def summary_table(cph: CoxPHFitter) -> pd.DataFrame:
    """
    Return a nicely formatted summary table of Cox PH coefficients
    with hazard ratios, confidence intervals, and p-values.
    """
    s = cph.summary.copy()
    # lifelines uses these column names
    s = s.rename(columns={
        "coef":                      "coefficient",
        "exp(coef)":                 "hazard_ratio",
        "se(coef)":                  "std_error",
        "exp(coef) lower 95%":       "hr_lower",
        "exp(coef) upper 95%":       "hr_upper",
        "p":                         "p_value",
    })
    s.index.name = "covariate"
    s = s.sort_values("hazard_ratio", ascending=False)
    s["significant"] = s["p_value"] < 0.05

    return s[[
        "coefficient", "hazard_ratio", "std_error",
        "hr_lower", "hr_upper", "p_value", "significant"
    ]]


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    cph = fit_cox_ph(df)
    print(summary_table(cph).to_string())
    print()
    cph.print_summary()