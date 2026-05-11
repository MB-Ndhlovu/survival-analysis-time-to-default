"""
Risk Chiizer: bin continuous variables into risk categories,
compute and compare survival curves for each bin.
Helps identify which risk buckets drive default timing differences.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


BINS_DTI = {
    "Low DTI (≤20%)":  (0.0,  0.20),
    "Mid DTI (20–35%)": (0.20, 0.35),
    "High DTI (>35%)":  (0.35, 1.0),
}

BINS_LTV = {
    "Low LTV (≤70%)":  (0.0,  0.70),
    "Mid LTV (70–85%)": (0.70, 0.85),
    "High LTV (>85%)":  (0.85, 1.5),
}

BINS_LOAN = {
    "Small (≤R100k)":   (0,    100_000),
    "Medium (100k–400k)": (100_000, 400_000),
    "Large (>R400k)":   (400_000, 10_000_000),
}


def assign_bin(value: float, bins: dict) -> str:
    for label, (lo, hi) in bins.items():
        if lo <= value < hi:
            return label
    return list(bins.keys())[-1]


def chiize(
    df: pd.DataFrame,
    var_name: str,
    bins: dict,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> dict:
    """
    For each bin of `var_name`, fit a Kaplan-Meier curve.
    Returns {bin_label: {"kmf": KaplanMeierFitter, "n": int}}.
    """
    results = {}
    for label, (lo, hi) in bins.items():
        mask = (df[var_name] >= lo) & (df[var_name] < hi)
        bin_df = df[mask].copy()

        kmf = KaplanMeierFitter()
        kmf.fit(bin_df[duration_col], bin_df[event_col], label=label)
        results[label] = {"kmf": kmf, "n": len(bin_df)}

    return results


def plot_chiizer(
    chiizer_results: dict,
    title: str,
    output_path: str,
):
    """Plot survival curves for each bin of a chiized variable."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, data in chiizer_results.items():
        data["kmf"].plot_survival_function(ax=ax, ci_show=True)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chiizer plot: {output_path}")


def chiizer_table(
    chiizer_results: dict, times: list
) -> pd.DataFrame:
    """Build summary table of S(t) at each time horizon per bin."""
    rows = []
    for label, data in chiizer_results.items():
        kmf = data["kmf"]
        row = {"bin": label, "n": data["n"]}
        for t in times:
            row[f"S({t}m)"] = round(kmf.survival_function_at_times(t).values[0], 4)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    for var, bins in [("debt_to_income", BINS_DTI), ("LTV_ratio", BINS_LTV)]:
        res = chiize(df, var, bins)
        plot_chiizer(res, f"Survival by {var}", f"chiizer_{var}.png")
        print(chiizer_table(res, times=[12, 24]))