import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def get_credit_score_band(score: int) -> str:
    if score < 580:
        return "< 580"
    elif score < 670:
        return "580-669"
    elif score < 740:
        return "670-739"
    else:
        return "740+"


def fit_kaplan_meier(df: pd.DataFrame) -> dict:
    kmf = KaplanMeierFitter()
    kmf.fit(df["time_end"], event_observed=df["event_default"], label="Overall")

    median_survival = kmf.median_survival_time_

    band_groups = {}
    for band in ["< 580", "580-669", "670-739", "740+"]:
        mask = df["credit_score_band"] == band
        if mask.sum() > 0:
            kmf_band = KaplanMeierFitter()
            kmf_band.fit(
                df.loc[mask, "time_end"],
                event_observed=df.loc[mask, "event_default"],
                label=band,
            )
            band_groups[band] = {
                "kmf": kmf_band,
                "median_survival": kmf_band.median_survival_time_,
                "survival_12m": kmf_band.predict(12),
                "survival_24m": kmf_band.predict(24),
            }

    return {"kmf": kmf, "median_survival": median_survival, "bands": band_groups}


def plot_kaplan_meier(df: pd.DataFrame, results: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    results["kmf"].plot_survival_function(ax=ax, color="black", linestyle="--", linewidth=2)

    colors = {"< 580": "#d62728", "580-669": "#ff7f0e", "670-739": "#2ca02c", "740+": "#1f77b4"}
    for band, data in results["bands"].items():
        data["kmf"].plot_survival_function(ax=ax, color=colors[band], linewidth=1.5)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_band_statistics(df: pd.DataFrame, results: dict) -> list:
    stats = []
    for band in ["< 580", "580-669", "670-739", "740+"]:
        if band in results["bands"]:
            data = results["bands"][band]
            count = (df["credit_score_band"] == band).sum()
            defaults = df.loc[df["credit_score_band"] == band, "event_default"].sum()
            stats.append(
                {
                    "band": band,
                    "n_loans": int(count),
                    "n_defaults": int(defaults),
                    "median_survival_months": (
                        float(data["median_survival"]) if not np.isnan(data["median_survival"]) else None
                    ),
                    "survival_12m": round(data["survival_12m"], 4),
                    "survival_24m": round(data["survival_24m"], 4),
                }
            )
    return stats


if __name__ == "__main__":
    from src.data_loader import generate_loan_data

    df = generate_loan_data()
    df["credit_score_band"] = df["credit_score"].apply(get_credit_score_band)

    results = fit_kaplan_meier(df)
    print(f"Overall median survival: {results['median_survival']}")
    for band, data in results["bands"].items():
        print(f"Band {band}: median={data['median_survival']:.1f}, 12m={data['survival_12m']:.3f}, 24m={data['survival_24m']:.3f}")