import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def fit_cox_ph(df):
    """Fit Cox Proportional Hazards model.

    Returns dict with:
        - cph: fitted CoxPHFitter
        - summary: DataFrame of coefficients, hazard ratios, p-values
        - concordance_index: model discrimination
    """
    cph = CoxPHFitter()

    # Prepare features with duration and event
    features = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_amount", "interest_rate", "LTV_ratio"
    ]
    X = df[features + ["time_end", "event_default"]].copy()
    X["income"] = np.log1p(X["income"])
    X["loan_amount"] = np.log1p(X["loan_amount"])

    # Fit
    cph.fit(X, duration_col="time_end", event_col="event_default")

    # Hazard ratios
    summary = cph.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])

    print("=== Cox PH Summary ===")
    print(summary[["coef", "hazard_ratio", "p"]].to_string())

    # Concordance index (discrimination)
    ci = cph.concordance_index_
    print(f"\nConcordance Index: {ci:.4f}")

    return {
        "cph": cph,
        "summary": summary,
        "concordance_index": ci,
    }

if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    result = fit_cox_ph(df)
    print(result["summary"][["hazard_ratio", "p"]].head())