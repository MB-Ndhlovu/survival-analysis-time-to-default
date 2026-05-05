"""Risk Chiizer: bin continuous variables into risk categories and analyze survival."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def build_chiizer(df, variable, bins, labels):
    """
    Bin a continuous variable and compute survival curves per bin.
    Args:
        df: loan dataframe
        variable: column name to bin
        bins: list of bin edges
        labels: list of bin labels
    Returns: dict of KM curves per bin
    """
    df = df.copy()
    df["bin"] = pd.cut(df[variable], bins=bins, labels=labels, include_lowest=True)

    kmf_dict = {}
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))

    for i, label in enumerate(labels):
        subset = df[df["bin"] == label]
        if len(subset) < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(subset["time_end"], event_observed=subset["event_default"], label=f"{label} (n={len(subset)})")
        kmf_dict[label] = {
            "kmf": kmf,
            "survival_12": kmf.survival_function_at_times(12).values[0],
            "survival_24": kmf.survival_function_at_times(24).values[0],
            "median": kmf.median_survival_time_,
        }
        kmf.plot_survival_function(ax=ax, color=colors[i])

    ax.set_title(f"Survival Curves by {variable}", fontsize=12)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = f"/home/workspace/Projects/survival-analysis-time-to-default/reports/chiizer_{variable.replace(' ','_').lower()}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved chiizer plot: {out_path}")

    return kmf_dict

def run_all_chiizers(df):
    """Run chiizer on key credit risk variables."""
    results = {}

    # DTI bins
    results["debt_to_income"] = build_chiizer(
        df, "debt_to_income",
        bins=[0, 15, 25, 35, 60],
        labels=["Low DTI (<15%)", "Moderate (15-25%)", "Elevated (25-35%)", "High (>35%)"]
    )

    # Employment years
    results["employment_years"] = build_chiizer(
        df, "employment_years",
        bins=[0, 2, 5, 10, 30],
        labels=["New (<2yr)", "Early (2-5yr)", "Established (5-10yr)", "Senior (10+yr)"]
    )

    # Loan amount
    results["loan_amount"] = build_chiizer(
        df, "loan_amount",
        bins=[0, 200000, 500000, 1000000, 2_000_000],
        labels=["Small (<200k)", "Medium (200-500k)", "Large (500k-1M)", "Very Large (>1M)"]
    )

    # LTV ratio
    results["LTV_ratio"] = build_chiizer(
        df, "LTV_ratio",
        bins=[0, 0.5, 0.7, 0.85, 1.2],
        labels=["Low LTV (<0.5)", "Moderate (0.5-0.7)", "High (0.7-0.85)", "Very High (>0.85)"]
    )

    return results

if __name__ == "__main__":
    from .data_loader import generate_loan_data
    df = generate_loan_data()
    run_all_chiizers(df)