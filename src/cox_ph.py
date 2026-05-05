"""Cox Proportional Hazards model for default risk."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

def fit_cox_ph(df):
    """
    Fit Cox PH model on loan covariates.
    Returns fitted model and hazard ratio summary.
    """
    # Prepare features
    X = df[[
        "credit_score", "employment_years", "debt_to_income",
        "loan_amount", "interest_rate", "LTV_ratio"
    ]].copy()

    # Standardize
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # Log loan amount
    X["log_loan_amount"] = np.log(df["loan_amount"])

    # Duration and event
    X["duration"] = df["time_end"]
    X["event_default"] = df["event_default"]

    cph = CoxPHFitter()
    cph.fit(X, duration_col="duration", event_col="event_default")

    # Print summary
    print("\n=== Cox PH Model Summary ===")
    cph.print_summary()

    # Hazard ratios
    hr_summary = cph.hazard_ratios_.rename("hazard_ratio").to_frame()
    hr_summary["coef"] = cph.params_.values
    hr_summary["se"] = cph.standard_errors_.values
    hr_summary["p"] = cph.summary["p"].values

    # Plot forest
    fig, ax = plt.subplots(figsize=(8, 5))
    hazards = hr_summary["hazard_ratio"].sort_values()
    colors = ["red" if h > 1 else "steelblue" for h in hazards]
    ax.barh(list(hazards.index), hazards.values, color=colors, alpha=0.7)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Hazard Ratio")
    ax.set_title("Cox PH Hazard Ratios (per 1-SD increase)")
    ax.grid(alpha=0.3, axis="x")
    for i, (name, row) in enumerate(hazards.items()):
        ax.text(row + 0.02, i, f"{row:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    out_path = "/home/workspace/Projects/survival-analysis-time-to-default/reports/cox_hazard_ratios.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved hazard ratio plot: {out_path}")

    # Interpretation
    print("\n=== Top Risk Factors (by hazard ratio) ===")
    sorted_hr = hr_summary.sort_values("hazard_ratio", ascending=False)
    for name, row in sorted_hr.iterrows():
        direction = "increases" if row["hazard_ratio"] > 1 else "decreases"
        change = abs(row["hazard_ratio"] - 1) * 100
        print(f"  {name}: HR={row['hazard_ratio']:.3f} → {change:.1f}% {direction} default risk")

    return cph, hr_summary

if __name__ == "__main__":
    from .data_loader import generate_loan_data
    df = generate_loan_data()
    cph, hr = fit_cox_ph(df)
    print(hr[["hazard_ratio"]].sort_values("hazard_ratio", ascending=False).to_string())