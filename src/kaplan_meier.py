"""
Kaplan-Meier survival curves segmented by credit score band.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def assign_credit_band(score: int) -> str:
    if score < 580:
        return "< 580 (Deep Subprime)"
    elif score < 670:
        return "580-669 (Subprime)"
    elif score < 740:
        return "670-739 (Near Prime)"
    else:
        return "740+ (Prime)"


def fit_km_by_credit_band(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
) -> dict:
    """
    Fit KM curves for each credit score band and plot.
    Returns dict of results keyed by band label.
    """
    df = df.copy()
    df["credit_band"] = df["credit_score"].apply(assign_credit_band)

    bands = ["< 580 (Deep Subprime)", "580-669 (Subprime)", "670-739 (Near Prime)", "740+ (Prime)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, ax = plt.subplots(figsize=figsize)

    results = {}
    kmf = KaplanMeierFitter()

    for band, color in zip(bands, colors):
        sub = df[df["credit_band"] == band]
        if sub.empty:
            continue

        kmf.fit(sub["time_end"], sub["event_default"], label=band)
        median = kmf.median_survival_time_

        results[band] = {
            "n": int(len(sub)),
            "events": int(sub["event_default"].sum()),
            "median_survival_months": float(median) if not pd.isna(median) else None,
            "survival_12m": float(kmf.survival_function_at_times(12).values[0]),
            "survival_24m": float(kmf.survival_function_at_times(24).values[0]),
        }

        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Months since origination")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% survival")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path = Path("reports/km_survival_curves.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    results["_plot_path"] = str(output_path)

    return results


def print_km_summary(results: dict) -> str:
    lines = ["\n" + "=" * 60]
    lines.append("KAPLAN-MEIER RESULTS BY CREDIT SCORE BAND")
    lines.append("=" * 60)

    for band, res in results.items():
        if band.startswith("_"):
            continue
        median = res["median_survival_months"]
        if median is not None:
            median_str = f"{median:.1f} months"
        else:
            median_str = "N/A (curve never crossed 50%)"
        lines.append(
            f"{band}\n"
            f"  N={res['n']}  Events={res['events']}\n"
            f"  Median survival: {median_str}\n"
            f"  S(12)={res['survival_12m']:.1%}  S(24)={res['survival_24m']:.1%}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_km_by_credit_band(df)
    print(print_km_summary(results))