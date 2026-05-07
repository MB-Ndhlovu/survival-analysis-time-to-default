"""
Kaplan-Meier survival curves by credit score band.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


CREDIT_BANDS = [
    ("< 580 (Deep Subprime)", 300, 579),
    ("580-669 (Subprime)", 580, 669),
    ("670-739 (Near Prime)", 670, 739),
    ("740+ (Prime)", 740, 850),
]


def fit_kaplan_meier(df: pd.DataFrame) -> dict:
    """
    Fit KM curves for each credit score band.
    Returns dict of fitted KMF objects keyed by band label.
    """
    kmf_by_band = {}

    for label, cs_min, cs_max in CREDIT_BANDS:
        mask = (df["credit_score"] >= cs_min) & (df["credit_score"] < cs_max)
        band_df = df[mask].copy()

        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df["time_end"],
            band_df["event_default"],
            label=label,
        )
        kmf_by_band[label] = kmf

    return kmf_by_band


def compute_median_times(kmf_by_band: dict) -> pd.DataFrame:
    """
    Compute median survival time (time to 50% default) per band.
    """
    rows = []
    for label, kmf in kmf_by_band.items():
        med = median_survival_times(kmf)
        rows.append({
            "band": label,
            "median_time_to_default": med if not np.isinf(med) else None,
        })
    return pd.DataFrame(rows)


def survival_at_months(kmf_by_band: dict, months: list) -> pd.DataFrame:
    """
    Survival probability at specific months per band.
    """
    rows = []
    for label, kmf in kmf_by_band.items():
        for m in months:
            rows.append({
                "band": label,
                "month": m,
                "survival_prob": round(kmf.survival_function_at_times(m).values[0], 4),
            })
    return pd.DataFrame(rows)


def plot_km_curves(kmf_by_band: dict, save_path: str = "reports/km_curves.png") -> None:
    """
    Plot KM survival curves for all credit score bands.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, kmf in kmf_by_band.items():
        kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved KM plot to {save_path}")


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data(5000)
    kmf_by_band = fit_kaplan_meier(df)

    print("=== Median Time to Default ===")
    print(compute_median_times(kmf_by_band))

    print("\n=== Survival Probabilities at 12 and 24 months ===")
    print(survival_at_months(kmf_by_band, [12, 24]))

    plot_km_curves(kmf_by_band)