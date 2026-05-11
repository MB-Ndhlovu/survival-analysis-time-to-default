"""
Kaplan-Meier survival analysis: fit curves for credit score bands,
compute median survival times, and plot survival functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# Credit score band definitions
BANDS = {
    "< 580":      (0,   579),
    "580–669":    (580, 669),
    "670–739":    (670, 739),
    "740+":       (740, 9999),
}


def assign_band(score: int) -> str:
    for label, (lo, hi) in BANDS.items():
        if lo <= score <= hi:
            return label
    return "< 580"


def fit_by_credit_band(
    df: pd.DataFrame,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> dict:
    """
    Fit separate Kaplan-Meier models for each credit score band.
    Returns a dict of {band_label: {"kmf": KaplanMeierFitter, "median": float|None}}.
    """
    results = {}

    for label, (lo, hi) in BANDS.items():
        mask = (df["credit_score"] >= lo) & (df["credit_score"] <= hi)
        band_df = df[mask].copy()

        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df[duration_col],
            band_df[event_col],
            label=label,
        )

        median = kmf.median_survival_time_
        results[label] = {"kmf": kmf, "median": median, "n": len(band_df)}

    return results


def survival_probabilities_at(
    kmf_results: dict, times: list
) -> pd.DataFrame:
    """
    Extract survival probabilities at specific time horizons.
    Returns a DataFrame with columns [band, 12m, 24m, 36m, median].
    """
    rows = []
    for label, data in kmf_results.items():
        kmf = data["kmf"]
        median = data["median"]
        row = {"band": label, "n": data["n"], "median_time": median}
        for t in times:
            survival_prob = kmf.survival_function_at_times(t).values[0]
            row[f"S({t}m)"] = round(survival_prob, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_km_curves(
    kmf_results: dict,
    output_path: str = "km_curves.png",
    duration_col: str = "time_end",
    event_col: str = "event_default",
    df: pd.DataFrame = None,
):
    """
    Plot Kaplan-Meier survival curves for all credit score bands.
    Overlays individual band curves with censoring marks.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in kmf_results.items():
        kmf = data["kmf"]
        kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Months Since Loan Origination")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", title="Credit Score Band")
    ax.grid(True, alpha=0.3)

    # Annotate median survival time for each band
    for label, data in kmf_results.items():
        median = data["median"]
        if median is not np.inf and not np.isnan(median):
            ax.axvline(x=median, color="gray", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Kaplan-Meier plot to {output_path}")


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_by_credit_band(df)
    plot_km_curves(results, df=df)
    probs = survival_probabilities_at(results, times=[12, 24, 36])
    print(probs.to_string(index=False))