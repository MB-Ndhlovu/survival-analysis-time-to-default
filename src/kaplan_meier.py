"""Kaplan-Meier survival curves by credit score band."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def assign_credit_band(score: int) -> str:
    if score < 580:
        return "Poor (<580)"
    elif score < 670:
        return "Fair (580-669)"
    elif score < 740:
        return "Good (670-739)"
    else:
        return "Excellent (740+)"


def fit_kaplan_meier(df: pd.DataFrame, output_path: str = "reports/km_curves.png") -> dict:
    """
    Fit Kaplan-Meier curves for each credit score band.

    Returns dict with median survival times and survival probabilities.
    """
    df = df.copy()
    df["credit_band"] = df["credit_score"].apply(assign_credit_band)

    band_order = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]
    colors = {"Poor (<580)": "#d62728", "Fair (580-669)": "#ff7f0e",
             "Good (670-739)": "#2ca02c", "Excellent (740+)": "#1f77b4"}

    kmf = KaplanMeierFitter()
    results = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    for band in band_order:
        subset = df[df["credit_band"] == band]
        if len(subset) == 0:
            continue

        kmf.fit(subset["time_end"], subset["event_default"], label=band)
        kmf.plot_survival_function(ax=ax, color=colors.get(band, None))

        # Median survival time (time at which S(t) < 0.5)
        survival_times = kmf.survival_function_
        median_survival = None
        for t in survival_times.index:
            val = survival_times.loc[t]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if val < 0.5:
                median_survival = t
                break

        # 12 and 24 month survival
        s12 = kmf.survival_function_at_times(12).values[0]
        s24 = kmf.survival_function_at_times(24).values[0]

        results[band] = {
            "n_loans": len(subset),
            "n_defaults": subset["event_default"].sum(),
            "median_survival_time": median_survival,
            "survival_12m": s12,
            "survival_24m": s24,
        }

    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return results


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_kaplan_meier(df)
    for band, metrics in results.items():
        print(f"\n{band}:")
        print(f"  Loans: {metrics['n_loans']}, Defaults: {metrics['n_defaults']}")
        print(f"  Median survival: {metrics['median_survival_time']} months")
        print(f"  12m survival: {metrics['survival_12m']:.1%}")
        print(f"  24m survival: {metrics['survival_24m']:.1%}")