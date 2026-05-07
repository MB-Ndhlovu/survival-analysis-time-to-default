"""Cox Proportional Hazards model for time-to-default analysis."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def fit_cox_ph_model(
    df: pd.DataFrame,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> dict:
    """
    Fit Cox Proportional Hazards model.

    Returns dict with:
    - cph: fitted CoxPHFitter
    - summary: DataFrame with coefficients, hazard ratios, p-values
    - concordance_index: model discrimination metric
    """
    features = [
        "credit_score",
        "income",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    # Prepare data
    X = df[features].copy()
    X["credit_score"] = X["credit_score"] / 100  # Scale for numerical stability

    cph = CoxPHFitter()
    cph.fit(
        df[[duration_col, event_col] + features],
        duration_col=duration_col,
        event_col=event_col,
    )

    # Extract summary
    summary = cph.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])

    # Concordance index
    concordance = cph.concordance_index_

    return {
        "cph": cph,
        "summary": summary,
        "concordance_index": concordance,
    }


def print_cox_results(results: dict) -> str:
    """Format Cox PH results for display."""
    cph = results["cph"]
    summary = results["summary"]

    lines = ["\n=== Cox Proportional Hazards Model ===\n"]
    lines.append(f"Concordance Index: {results['concordance_index']:.4f}\n")
    lines.append("Coefficients and Hazard Ratios:\n")
    lines.append(f"{'Variable':<20} {'Coef':>10} {'HR':>10} {'p-value':>12} {'Significant':>12}\n")
    lines.append("-" * 66)

    for var in summary.index:
        coef = summary.loc[var, "coef"]
        hr = summary.loc[var, "hazard_ratio"]
        p = summary.loc[var, "p"]
        sig = "Yes" if p < 0.05 else "No"
        lines.append(f"{var:<20} {coef:>10.4f} {hr:>10.4f} {p:>12.4f} {sig:>12}")

    lines.append("\nInterpretation:")
    lines.append("- HR > 1: Increases default hazard (shorter time to default)")
    lines.append("- HR < 1: Decreases default hazard (longer time to default)")
    lines.append("- HR = 1: No effect on default hazard")

    return "\n".join(lines)


def get_top_risk_factors(summary: pd.DataFrame, n: int = 3) -> list:
    """Return top n risk factors by hazard ratio magnitude."""
    risk_factors = []
    for var in summary.index:
        hr = summary.loc[var, "hazard_ratio"]
        coef = summary.loc[var, "coef"]
        p = summary.loc[var, "p"]
        if p < 0.05:
            if hr > 1:
                risk_factors.append((var, hr, "increases", p))
            else:
                risk_factors.append((var, hr, "decreases", p))

    # Sort by hazard ratio deviation from 1
    risk_factors.sort(key=lambda x: abs(x[1] - 1), reverse=True)
    return risk_factors[:n]


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_cox_ph_model(df)
    print(print_cox_results(results))