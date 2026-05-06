import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def fit_kaplan_meier(df):
    """Fit Kaplan-Meier curves for credit score bands.

    Returns dict with:
        - kmf: fitted KaplanMeierFitter
        - credit_band_labels: list of band labels
        - median_survival: dict of band -> median months
        - survival_probs: dict of band -> {12mo, 24mo} survival
    """
    kmf = KaplanMeierFitter()

    bands = [
        ("Deep Subprime (<580)",  df["credit_score"] < 580),
        ("Subprime (580-669)",    (df["credit_score"] >= 580) & (df["credit_score"] <= 669)),
        ("Near Prime (670-739)",  (df["credit_score"] >= 670) & (df["credit_score"] <= 739)),
        ("Prime (740+)",          df["credit_score"] >= 740),
    ]

    plt.figure(figsize=(10, 6))
    median_survival = {}
    survival_probs = {}

    for label, mask in bands:
        band_df = df[mask]
        if len(band_df) == 0:
            continue
        kmf.fit(
            band_df["time_end"],
            event_observed=band_df["event_default"],
            label=label
        )
        median_survival[label] = kmf.median_survival_time_ if not np.isnan(kmf.median_survival_time_) else None

        # 12-month and 24-month survival
        timeline = np.arange(0, 37)
        sf = kmf.survival_function_at_times(timeline)
        probs = {t: float(sf[t]) if t in sf.index else None for t in [12, 24]}
        survival_probs[label] = probs

        kmf.plot_survival_function(ax=plt.gca())

    plt.title("Kaplan-Meier Survival Curves by Credit Score Band")
    plt.xlabel("Months")
    plt.ylabel("Survival Probability")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/workspace/Projects/survival-analysis-time-to-default/reports/km_survival_curves.png", dpi=150)
    plt.close()
    print("Saved: reports/km_survival_curves.png")

    return {
        "kmf": kmf,
        "credit_bands": bands,
        "median_survival": median_survival,
        "survival_probs_12_24": survival_probs,
    }

if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    result = fit_kaplan_meier(df)
    for label, med in result["median_survival"].items():
        print(f"{label}: median = {med} months")