"""
Cox Proportional Hazards model for time-to-default.
Fit Cox PH, interpret coefficients, compute hazard ratios,
and assess which factors most increase default risk.
"""

import json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .data_loader import get_credit_score_band


def fit_cox_ph(df):
    """
    Fit Cox Proportional Hazards model on loan covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with time_end, event_default, and covariate columns.

    Returns
    -------
    CoxPHFitter, pd.DataFrame
        Fitted model and summary DataFrame.
    """
    covariates = [
        "credit_score", "income", "employment_years",
        "debt_to_income", "loan_amount", "interest_rate", "LTV_ratio"
    ]

    # Log-transform income and loan_amount (highly skewed)
    X = df[covariates].copy()
    X["log_income"] = np.log1p(X["income"])
    X["log_loan_amount"] = np.log1p(X["loan_amount"])
    X = X.drop(columns=["income", "loan_amount"])

    # Add duration and event columns
    X["time_end"] = df["time_end"].values
    X["event_default"] = df["event_default"].values

    # Standardize for better convergence
    for col in ["credit_score", "employment_years", "debt_to_income",
                "interest_rate", "LTV_ratio", "log_income", "log_loan_amount"]:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    duration = X["time_end"]
    event = X["event_default"]

    cph = CoxPHFitter()
    cph.fit(X, duration_col="time_end", event_col="event_default")

    X_train = X.drop(columns=["time_end", "event_default"])
    return cph, X_train


def print_cox_summary(cph):
    """Print formatted Cox PH summary."""
    print("\n" + "=" * 70)
    print("COX PROPORTIONAL HAZARDS MODEL".center(70))
    print("=" * 70)
    print(cph.print_summary(decimals=4))
    print("\nHazard Ratio Interpretation:")
    print("  HR > 1  → increases default hazard")
    print("  HR < 1  → decreases default hazard")
    print("  HR = 1  → no effect")


def get_hazard_ratios(cph):
    """
    Extract hazard ratios and confidence intervals from fitted model.

    Returns
    -------
    pd.DataFrame
        DataFrame with covariate, coef, exp_coef (HR), se, ci_lower, ci_upper
    """
    summary = cph.summary.copy()
    summary = summary.rename(columns={
        "coef": "coefficient",
        "exp(coef)": "hazard_ratio",
        "se(coef)": "std_error",
        "coef lower 95%": "ci_lower",
        "coef upper 95%": "ci_upper",
        "exp(coef) lower 95%": "hr_ci_lower",
        "exp(coef) upper 95%": "hr_ci_upper",
        "p": "p_value"
    })
    return summary


def top_hazard_factors(cph, n=5):
    """Return top N factors that most increase default hazard (by HR)."""
    hr = get_hazard_ratios(cph)
    hr = hr.sort_values("hazard_ratio", ascending=False)
    return hr.head(n)


def export_cox_json(cph, path):
    """Export Cox PH results to JSON."""
    hr = get_hazard_ratios(cph)
    records = hr.reset_index().rename(columns={"index": "covariate"}).to_dict("records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, (np.floating, float, np.integer)):
                r[k] = round(float(v), 6)
    with open(path, "w") as f:
        json.dump({"hazard_ratios": records, "concordance_index": round(float(cph.concordance_index_), 4)}, f, indent=2)
    print(f"Exported Cox PH results to {path}")


if __name__ == "__main__":
    from .data_loader import generate_loan_data

    df = generate_loan_data()
    cph, X = fit_cox_ph(df)
    print_cox_summary(cph)
    print("\nTop hazard factors:")
    print(top_hazard_factors(cph))