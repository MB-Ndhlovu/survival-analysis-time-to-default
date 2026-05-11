"""Cox Proportional Hazards model for default risk."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame) -> dict:
    """
    Fit Cox Proportional Hazards model to identify factors increasing default risk.

    Returns coefficients, hazard ratios, and model summary.
    """
    cph = CoxPHFitter()
    cph.fit(
        df[["time_end", "event_default", "income", "credit_score", "employment_years",
            "debt_to_income", "loan_amount", "interest_rate", "LTV_ratio"]],
        duration_col="time_end",
        event_col="event_default"
    )

    # Extract coefficients and hazard ratios
    summary = cph.summary.copy()
    hazard_ratios = np.exp(summary["coef"])

    results = {
        "log_likelihood": cph.log_likelihood_,
        "concordance_index": cph.concordance_index_,
        "partial_log_likelihood": cph.log_likelihood_,
        "coefficients": {},
        "hazard_ratios": {},
        "p_values": {},
    }

    for var in summary.index:
        results["coefficients"][var] = round(summary.loc[var, "coef"], 4)
        results["hazard_ratios"][var] = round(hazard_ratios.loc[var], 4)
        results["p_values"][var] = round(summary.loc[var, "p"], 4)

    # Print formatted results
    print("\n" + "=" * 60)
    print("COX PROPORTIONAL HAZARDS MODEL RESULTS")
    print("=" * 60)
    print(f"Concordance Index: {results['concordance_index']:.4f}")
    print(f"Log-Likelihood: {results['log_likelihood']:.2f}")
    print("\nHazard Ratios (effect on instantaneous default rate):")
    print("-" * 50)
    print(f"{'Variable':<20} {'HR':>8} {'Coef':>8} {'p-value':>8}")
    print("-" * 50)

    for var in sorted(results["hazard_ratios"].keys()):
        hr = results["hazard_ratios"][var]
        coef = results["coefficients"][var]
        p = results["p_values"][var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{var:<20} {hr:>8.3f} {coef:>8.4f} {p:>8.4f} {sig}")

    print("\nInterpretation:")
    print("  HR > 1: Factor increases default risk")
    print("  HR < 1: Factor decreases default risk")
    print("  HR = 1: No effect")
    print("  Significance: *** p<0.001, ** p<0.01, * p<0.05")

    return results


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_cox_ph(df)