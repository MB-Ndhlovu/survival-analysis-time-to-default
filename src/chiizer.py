import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(series: pd.Series, n_bins: int, labels: list = None) -> pd.Series:
    return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")


def compute_risk_bands(df: pd.DataFrame) -> dict:
    bands = {}

    df["DTI_band"] = bin_variable(
        df["debt_to_income"], 3, labels=["Low DTI", "Medium DTI", "High DTI"]
    )
    df["LTV_band"] = bin_variable(df["LTV_ratio"], 3, labels=["Low LTV", "Medium LTV", "High LTV"])
    df["income_band"] = bin_variable(
        df["income"], 3, labels=["Low Income", "Medium Income", "High Income"]
    )
    df["rate_band"] = bin_variable(
        df["interest_rate"], 3, labels=["Low Rate", "Medium Rate", "High Rate"]
    )

    for band_col in ["DTI_band", "LTV_band", "income_band", "rate_band"]:
        band_name = band_col.replace("_band", "")
        bands[band_name] = {}
        for cat in df[band_col].unique():
            if pd.isna(cat):
                continue
            mask = df[band_col] == cat
            kmf = KaplanMeierFitter()
            kmf.fit(
                df.loc[mask, "time_end"],
                event_observed=df.loc[mask, "event_default"],
                label=str(cat),
            )
            bands[band_name][str(cat)] = {
                "kmf": kmf,
                "survival_12m": kmf.predict(12),
                "survival_24m": kmf.predict(24),
                "median": kmf.median_survival_time_,
            }

    return bands


def plot_risk_bands(df: pd.DataFrame, bands: dict, save_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    band_cols = ["DTI_band", "LTV_band", "income_band", "rate_band"]
    titles = ["Debt-to-Income Ratio", "LTV Ratio", "Income Level", "Interest Rate"]
    colors_list = [
        ["#1f77b4", "#ff7f0e", "#d62728"],
        ["#2ca02c", "#ff7f0e", "#d62728"],
        ["#d62728", "#ff7f0e", "#2ca02c"],
        ["#d62728", "#ff7f0e", "#2ca02c"],
    ]

    for idx, (col, title) in enumerate(zip(band_cols, titles)):
        ax = axes[idx]
        band_name = col.replace("_band", "")
        if band_name not in bands:
            continue

        for i, (cat, data) in enumerate(bands[band_name].items()):
            color = colors_list[idx][min(i, len(colors_list[idx]) - 1)]
            data["kmf"].plot_survival_function(ax=ax, color=color, linewidth=1.5)

        ax.set_title(f"Survival by {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_risk_summary(bands: dict) -> dict:
    summary = {}
    for var, categories in bands.items():
        summary[var] = {}
        for cat, data in categories.items():
            summary[var][cat] = {
                "survival_12m": round(data["survival_12m"], 4),
                "survival_24m": round(data["survival_24m"], 4),
                "median_survival": float(data["median"]) if not np.isnan(data["median"]) else None,
            }
    return summary


if __name__ == "__main__":
    from src.data_loader import generate_loan_data

    df = generate_loan_data()
    bands = compute_risk_bands(df)
    for var, cats in bands.items():
        print(f"\n{var.upper()} Bands:")
        for cat, data in cats.items():
            print(f"  {cat}: 12m={data['survival_12m']:.3f}, 24m={data['survival_24m']:.3f}")