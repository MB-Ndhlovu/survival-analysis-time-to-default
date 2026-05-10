"""
Kaplan-Meier survival analysis for credit score bands.
Fits KM curves per credit score segment, plots survival functions,
computes median survival times, and 12/24-month survival probabilities.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from .data_loader import get_credit_score_band


def fit_kaplan_meier_by_band(df):
    """
    Fit Kaplan-Meier curves for each credit score band.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with time_end, event_default, credit_score columns.

    Returns
    -------
    dict
        Dictionary mapping band label -> (kmf, median_time, survival_probs)
    """
    df = df.copy()
    df["band"] = df["credit_score"].apply(get_credit_score_band)
    bands_ordered = [
        "Deep Subprime (<580)",
        "Subprime (580-669)",
        "Near Prime (670-739)",
        "Prime (740+)"
    ]

    results = {}
    for band in bands_ordered:
        band_df = df[df["band"] == band]
        if len(band_df) == 0:
            continue

        duration = band_df["time_end"]
        event = band_df["event_default"]

        kmf = KaplanMeierFitter()
        kmf.fit(duration, event, label=band)

        # Median survival time (None if >50% never default)
        median_survival = kmf.median_survival_time_
        if median_survival == np.inf:
            median_survival = None

        # Survival probabilities at 12 and 24 months
        survival_12 = kmf.survival_function_at_times(12).values[0]
        survival_24 = kmf.survival_function_at_times(24).values[0]

        results[band] = {
            "kmf": kmf,
            "n": int(len(band_df)),
            "events": int(event.sum()),
            "median_survival_months": float(median_survival) if median_survival is not None else None,
            "survival_12_month": round(float(survival_12), 4),
            "survival_24_month": round(float(survival_24), 4),
        }

    return results


def plot_kaplan_meier(results, save_path=None):
    """
    Plot Kaplan-Meier curves for all bands on a single figure.

    Parameters
    ----------
    results : dict
        Output from fit_kaplan_meier_by_band()
    save_path : str, optional
        File path to save the plot (PNG).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Deep Subprime (<580)": "#e74c3c",
              "Subprime (580-669)": "#e67e22",
              "Near Prime (670-739)": "#f1c40f",
              "Prime (740+)": "#2ecc71"}

    for band, data in results.items():
        kmf = data["kmf"]
        color = colors.get(band, "#3498db")
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    # Add median markers
    for band, data in results.items():
        median = data["median_survival_months"]
        if median is not None:
            ax.axvline(x=median, color=colors.get(band, "#3498db"),
                       linestyle="--", alpha=0.5, linewidth=1)
            ax.text(median + 0.5, 0.95, f"{band.split()[0]}: {median:.0f}m",
                    fontsize=7, color=colors.get(band, "#3498db"), rotation=90, va="top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Kaplan-Meier plot to {save_path}")
    plt.close()


def summarize_km_results(results):
    """Print a formatted summary of KM results."""
    print("\n" + "=" * 70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS BY CREDIT SCORE BAND".center(70))
    print("=" * 70)
    print(f"{'Band':<30} {'N':>6} {'Events':>7} {'Median(m)':>10} {'12m S':>8} {'24m S':>8}")
    print("-" * 70)
    for band, data in results.items():
        med = f"{data['median_survival_months']:.0f}" if data['median_survival_months'] else "∞"
        print(f"{band:<30} {data['n']:>6} {data['events']:>7} {med:>10} "
              f"{data['survival_12_month']:>8.3f} {data['survival_24_month']:>8.3f}")
    print("-" * 70)


def export_km_json(results, path):
    """Export KM results to JSON (excluding KM objects)."""
    export = {}
    for band, data in results.items():
        clean = {}
        for k, v in data.items():
            if k == "kmf":
                continue
            if isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean[k] = float(round(v, 6)) if v is not None else None
            else:
                clean[k] = v
        export[band] = clean
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported KM results to {path}")


if __name__ == "__main__":
    from .data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_kaplan_meier_by_band(df)
    summarize_km_results(results)
    plot_kaplan_meier(results)