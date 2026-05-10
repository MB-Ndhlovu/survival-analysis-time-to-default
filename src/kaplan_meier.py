"""Kaplan-Meier survival curves by credit score band."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def assign_credit_band(score: int) -> str:
    """Assign credit score band."""
    if score < 580:
        return "< 580 (Poor)"
    elif score < 670:
        return "580-669 (Fair)"
    elif score < 740:
        return "670-739 (Good)"
    else:
        return "740+ (Excellent)"


def fit_kaplan_meier(df: pd.DataFrame):
    """
    Fit Kaplan-Meier curves for each credit score band.
    Returns summary dict with median survival times and survival probabilities.
    """
    df = df.copy()
    df["credit_band"] = df["credit_score"].apply(assign_credit_band)

    bands = ["< 580 (Poor)", "580-669 (Fair)", "670-739 (Good)", "740+ (Excellent)"]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, band in enumerate(bands):
        subset = df[df["credit_band"] == band]
        if len(subset) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(subset["time_end"], event_observed=subset["event_default"], label=band)

        # 12 and 24 month survival
        s12 = kmf.survival_function_at_times(12).values[0]
        s24 = kmf.survival_function_at_times(24).values[0]

        # Median survival time (time at which S(t) <= 0.5)
        median_survival = kmf.median_survival_time_
        if np.isnan(median_survival):
            median_survival = "> 60 months"

        results[band] = {
            "n": len(subset),
            "defaults": int(subset["event_default"].sum()),
            "censored": int((subset["event_default"] == 0).sum()),
            "12m_survival": round(s12, 4),
            "24m_survival": round(s24, 4),
            "median_survival_months": median_survival,
        }

        # Plot
        kmf.plot_survival_function(ax=axes[0], show_censors=True)
        axes[0].set_title("Survival Curves by Credit Score Band")
        axes[0].set_xlabel("Months")
        axes[0].set_ylabel("Survival Probability")
        axes[0].legend(loc="lower left")
        axes[0].grid(True, alpha=0.3)

        # Cumulative hazard
        kmf.plot_cumulative_density(ax=axes[1], show_censors=True)
        axes[1].set_title("Cumulative Density (Defaults) by Credit Band")
        axes[1].set_xlabel("Months")
        axes[1].set_ylabel("Cumulative Default Probability")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/workspace/Projects/survival-analysis-time-to-default/reports/km_curves.png", dpi=150)
    plt.close()

    return results


if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    results = fit_kaplan_meier(df)
    for band, stats in results.items():
        print(f"\n{band}")
        for k, v in stats.items():
            print(f"  {k}: {v}")