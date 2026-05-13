"""Kaplan-Meier survival curves for credit score bands."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from .data_loader import get_credit_score_band


def fit_by_credit_band(
    df: pd.DataFrame,
    time_col: str = "time_end",
    event_col: str = "event_default",
) -> pd.DataFrame:
    """Fit Kaplan-Meier curves for each credit score band.

    Returns DataFrame with summary statistics per band.
    """
    df = df.copy()
    df["credit_band"] = df["credit_score"].apply(get_credit_score_band)

    bands = [
        "Deep Subprime (< 580)",
        "Subprime (580-669)",
        "Near Prime (670-739)",
        "Prime (740+)",
    ]

    results = []
    kmfs = {}

    for band in bands:
        subset = df[df["credit_band"] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], subset[event_col], label=band)

        # Median survival time
        median_time = kmf.median_survival_time_
        if pd.isna(median_time):
            median_time = ">24 months"

        # 12-month and 24-month survival
        survival_12 = kmf.survival_function_at_times(12).values[0]
        survival_24 = kmf.survival_function_at_times(24).values[0]

        results.append({
            "band": band,
            "n": len(subset),
            "defaults": subset[event_col].sum(),
            "median_survival_months": median_time,
            "survival_12m": round(survival_12, 4),
            "survival_24m": round(survival_24, 4),
        })

        kmfs[band] = kmf

    return pd.DataFrame(results), kmfs


def plot_kaplan_meier(
    kmfs: dict,
    output_path: str = "reports/km_plot.png",
    figsize: tuple = (10, 7),
) -> None:
    """Plot Kaplan-Meier curves for all credit bands."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "Deep Subprime (< 580)": "#d62728",
        "Subprime (580-669)": "#ff7f0e",
        "Near Prime (670-739)": "#2ca02c",
        "Prime (740+)": "#1f77b4",
    }

    for band, kmf in kmfs.items():
        kmf.plot_survival_function(ax=ax, color=colors.get(band, None))

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% survival")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Kaplan-Meier plot saved to {output_path}")


def run(df: pd.DataFrame) -> tuple:
    """Run Kaplan-Meier analysis."""
    summary_df, kmfs = fit_by_credit_band(df)

    print("\n=== Kaplan-Meier Results by Credit Band ===")
    print(summary_df.to_string(index=False))

    plot_kaplan_meier(kmfs)

    return summary_df, kmfs


if __name__ == "__main__":
    from .data_loader import generate_loan_data

    df = generate_loan_data()
    summary, _ = run(df)
    print("\nSurvival probabilities:")
    print(summary[["band", "survival_12m", "survival_24m", "median_survival_months"]])