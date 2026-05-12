"""Cox Proportional Hazards model for time-to-default."""

import pandas as pd

from lifelines import CoxPHFitter


def fit_cox(data, duration_col="time_end", event_col="event_default"):
    """Fit Cox PH model on loan features, return coefficients and hazard ratios."""
    features = [
        "credit_score",
        "debt_to_income",
        "LTV_ratio",
        "interest_rate",
        "employment_years",
    ]
    df = data[features + [duration_col, event_col]].copy()

    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    # Summary table
    summary = cph.summary.copy()
    summary["hazard_ratio"] = summary["exp(coef)"]
    cols = ["coef", "exp(coef)", "se(coef)", "z", "p"]
    available = [c for c in cols if c in summary.columns]
    summary = summary[available]
    summary = summary.rename(columns={"exp(coef)": "hazard_ratio", "se(coef)": "se"})
    summary = summary.round(4)

    print("\n=== Cox PH Coefficients ===")
    print(summary.to_string())

    # Concordance index (model discrimination)
    concord = cph.concordance_index_
    print(f"\nConcordance index: {concord:.4f}")

    # Aggregate results
    results = {
        "concordance_index": round(concord, 4),
        "coefficients": {},
    }
    for idx, row in summary.iterrows():
        results["coefficients"][idx] = {
            "coef": float(row["coef"]),
            "hazard_ratio": float(row["hazard_ratio"]),
            "p_value": float(row["p"]),
        }

    return cph, results