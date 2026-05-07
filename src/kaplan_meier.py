"""Kaplan-Meier survival analysis by credit score bands."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def fit_km_by_credit_band(
    df: pd.DataFrame,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> dict:
    """
    Fit Kaplan-Meier curves for each credit score band.

    Returns dict with:
    - kmfitters: {band: KaplanMeierFitter}
    - medians: {band: median_survival_time or None}
    - survival_probs: {band: {12: S(12), 24: S(24)}}
    """
    bands = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]

    # Assign credit score bands
    df = df.copy()
    df["credit_band"] = df["credit_score"].apply(lambda s: (
        "Poor (<580)" if s < 580
        else "Fair (580-669)" if s < 670
        else "Good (670-739)" if s < 740
        else "Excellent (740+)"
    ))

    kmfitters = {}
    medians = {}
    survival_probs = {}

    for band in bands:
        band_df = df[df["credit_band"] == band]
        if len(band_df) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df[duration_col],
            band_df[event_col],
            label=band,
        )
        kmfitters[band] = kmf

        # Median survival time
        try:
            median = median_survival_times(kmf)
            medians[band] = float(median) if not pd.isna(median) else None
        except Exception:
            medians[band] = None

        # Survival probabilities at 12 and 24 months
        survival_probs[band] = {
            12: float(kmf.survival_function_at_times([12]).values[0]),
            24: float(kmf.survival_function_at_times([24]).values[0]),
        }

    return {
        "kmfitters": kmfitters,
        "medians": medians,
        "survival_probs": survival_probs,
    }


def plot_km_curves(
    kmfitters: dict,
    output_path: str = "reports/km_curves.png",
) -> None:
    """Plot and save Kaplan-Meier curves for all credit bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Poor (<580)": "#d62728", "Fair (580-669)": "#ff7f0e",
              "Good (670-739)": "#2ca02c", "Excellent (740+)": "#1f77b4"}

    for band, kmf in kmfitters.items():
        color = colors.get(band, None)
        kmf.plot_survival_function(ax=ax, color=color)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14)
    ax.set_xlabel("Months since loan origination", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% survival")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved KM curves to {output_path}")


def get_results_text(results: dict) -> str:
    """Format results for display."""
    lines = ["\n=== Kaplan-Meier Results by Credit Score Band ===\n"]

    for band in ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]:
        if band not in results["medians"]:
            continue
        median = results["medians"][band]
        probs = results["survival_probs"][band]
        lines.append(f"{band}:")
        lines.append(f"  Median survival time: {median:.1f} months" if median else "  Median survival time: Not reached")
        lines.append(f"  12-month survival: {probs[12]:.1%}")
        lines.append(f"  24-month survival: {probs[24]:.1%}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_km_by_credit_band(df)
    print(get_results_text(results))
    plot_km_curves(results["kmfitters"])