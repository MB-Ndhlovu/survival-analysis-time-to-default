"""
Cox Proportional Hazards model — interpret coefficients and hazard ratios.
"""

import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame) -> dict:
    """
    Fit Cox PH model on loan features.
    Returns dict with coefficients, hazard ratios, and confidence intervals.
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

    df_model = df[features + ["time_end", "event_default"]].copy()

    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(df_model, duration_col="time_end", event_col="event_default")

    summary = cph.summary.copy()

    results = {
        "concordance_index": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "coefficients": {},
    }

    for var in features:
        row = summary.loc[var]
        results["coefficients"][var] = {
            "coef": float(row["coef"]),
            "hazard_ratio": float(row["exp(coef)"]),
            "se": float(row["se(coef)"]),
            "z": float(row["z"]),
            "p_value": float(row["p"]),
            "ci_lower": float(row["coef lower 95%"]),
            "ci_upper": float(row["coef upper 95%"]),
        }

    return results


def print_cox_summary(results: dict) -> str:
    lines = ["\n" + "=" * 60]
    lines.append("COX PROPORTIONAL HAZARDS MODEL RESULTS")
    lines.append("=" * 60)
    lines.append(f"Concordance Index: {results['concordance_index']:.4f}")
    lines.append(f"Log-Likelihood:    {results['log_likelihood']:.2f}")

    lines.append("\nHazard Ratios (HR > 1 = increases default risk):")
    lines.append("-" * 60)
    lines.append(f"{'Variable':<22} {'HR':>8}  {'99% CI':>20}  {'P-value':>10}")
    lines.append("-" * 60)

    sorted_vars = sorted(
        results["coefficients"].items(),
        key=lambda x: x[1]["hazard_ratio"],
        reverse=True,
    )

    for var, vals in sorted_vars:
        hr = vals["hazard_ratio"]
        ci = f"({vals['ci_lower']:.3f}, {vals['ci_upper']:.3f})"
        p = vals["p_value"]
        flag = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        lines.append(f"{var:<22} {hr:>8.4f}  {ci:>20}  {p:>10.4f}{flag}")

    lines.append("-" * 60)
    lines.append("Signif. codes:  '***' p<0.001  '**' p<0.01  '*' p<0.05")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_cox_ph(df)
    print(print_cox_summary(results))