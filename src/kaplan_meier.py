"""Kaplan-Meier survival curves by credit score band."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from .data_loader import get_credit_band

def fit_km_by_credit_band(df):
    """
    Fit KM curves for each credit score band.
    Returns dict of fitted estimators and summary stats.
    """
    df = df.copy()
    df["credit_band"] = df["credit_score"].apply(get_credit_band)

    bands = ["Very Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Survival curves
    ax1 = axes[0]
    colors = {"Very Poor (<580)": "red", "Fair (580-669)": "orange",
              "Good (670-739)": "steelblue", "Excellent (740+)": "green"}

    for band in bands:
        subset = df[df["credit_band"] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(subset["time_end"], event_observed=subset["event_default"], label=band)
        results[band] = {
            "kmf": kmf,
            "median_survival": kmf.median_survival_time_,
            "survival_at_12": kmf.survival_function_at_times(12).values[0],
            "survival_at_24": kmf.survival_function_at_times(24).values[0],
        }
        kmf.plot_survival_function(ax=ax1, color=colors[band])

    ax1.set_title("Kaplan-Meier Survival Curves by Credit Band", fontsize=12)
    ax1.set_xlabel("Months")
    ax1.set_ylabel("Survival Probability")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)

    # Median survival bar chart
    ax2 = axes[1]
    medians = [results[b]["median_survival"] for b in bands]
    bars = ax2.bar(bands, medians, color=[colors[b] for b in bands], alpha=0.7)
    ax2.set_title("Median Survival Time by Credit Band", fontsize=12)
    ax2.set_ylabel("Months to Default")
    ax2.set_ylim(0, max(medians) * 1.3)
    for bar, val in zip(bars, medians):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.tick_params(axis="x", labelrotation=15)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = "/home/workspace/Projects/survival-analysis-time-to-default/reports/km_survival_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved KM plot: {out_path}")

    # Summary table
    summary = []
    for band in bands:
        r = results[band]
        summary.append({
            "credit_band": band,
            "n": len(df[df["credit_band"] == band]),
            "median_survival_months": round(r["median_survival"], 1),
            "survival_prob_12m": round(r["survival_at_12"], 4),
            "survival_prob_24m": round(r["survival_at_24"], 4),
        })

    return results, pd.DataFrame(summary)

if __name__ == "__main__":
    from .data_loader import generate_loan_data
    df = generate_loan_data()
    results, summary = fit_km_by_credit_band(df)
    print(summary.to_string(index=False))