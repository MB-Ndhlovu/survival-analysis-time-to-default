"""Cox Proportional Hazards model for time-to-default."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame) -> dict:
    """
    Fit a Cox Proportional Hazards model on loan features.
    Returns coefficients, hazard ratios, and model concordance.
    """
    features = [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    # Prepare DataFrame with correct column structure
    cox_df = pd.DataFrame({
        "duration": df["time_end"],
        "event": df["event_default"],
    })
    for f in features:
        cox_df[f] = df[f]

    # Scale features for numerical stability
    for f in features:
        mean = cox_df[f].mean()
        std = cox_df[f].std()
        cox_df[f] = (cox_df[f] - mean) / std

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event")

    # Extract summary
    summary_df = cph.summary
    if summary_df.empty:
        print("Warning: Cox PH summary is empty — check feature scaling")
        return {"concordance_index": 0.5, "hazard_ratios": {}, "coefficients": {}}

    # Build results dict
    hazard_ratios = summary_df["exp(coef)"].to_dict()
    coefficients = summary_df["coef"].to_dict()
    concordance = cph.concordance_index_

    result = {
        "concordance_index": round(concordance, 4),
        "hazard_ratios": {k: round(v, 4) for k, v in hazard_ratios.items()},
        "coefficients": {k: round(v, 4) for k, v in coefficients.items()},
        "log_likelihood": round(cph.log_likelihood_, 4),
    }

    # Print interpretation
    print("\n=== Cox PH Results ===")
    print(f"Concordance Index: {concordance:.4f}")
    print("\nHazard Ratios (HR > 1 = higher default risk):")
    for cov, hr in sorted(hazard_ratios.items(), key=lambda x: x[1], reverse=True):
        direction = "UP risk" if hr > 1 else "DOWN risk"
        print(f"  {cov}: HR={hr:.4f} {direction}")

    return result


if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    result = fit_cox_ph(df)
    print(result)