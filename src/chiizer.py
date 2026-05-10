"""
Risk Chiizer — bin continuous variables into risk categories
and compute survival curves for each bin.
This creates interpretable risk segments for business use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from .data_loader import get_credit_score_band


def bin_variable(series, n_bins=4, labels=None):
    """
    Bin a continuous variable into quantile-based categories.

    Parameters
    ----------
    series : pd.Series
        Continuous variable to bin.
    n_bins : int
        Number of bins.
    labels : list, optional
        Custom bin labels.

    Returns
    -------
    pd.Series
        Categorical bin labels.
    """
    try:
        binned, bins = pd.qcut(series, q=n_bins, labels=labels, duplicates="drop", retbins=True)
        actual_bins = len(bins) - 1
        if labels is None and actual_bins < n_bins:
            binned, bins = pd.qcut(series, q=n_bins,
                                   labels=[f"Q{i+1}" for i in range(actual_bins)],
                                   duplicates="drop", retbins=True)
        return binned
    except ValueError:
        # Fallback to equal-width bins if qcut fails
        width = (series.max() - series.min()) / n_bins
        edges = [series.min() + i * width for i in range(n_bins + 1)]
        if labels is None:
            labels = [f"Q{i+1}" for i in range(n_bins)]
        return pd.cut(series, bins=edges, labels=labels[:n_bins], include_lowest=True)


def chiize(df, variable, n_bins=4, labels=None, event_col="event_default", duration_col="time_end"):
    """
    Bin a continuous variable and compute survival curves per bin.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data.
    variable : str
        Column name to bin and analyze.
    n_bins : int
        Number of bins.
    labels : list, optional
        Bin labels.
    event_col : str
        Event indicator column.
    duration_col : str
        Duration/time column.

    Returns
    -------
    dict
        Bin label -> survival metrics (n, events, 12m survival, 24m survival)
    """
    binned = bin_variable(df[variable], n_bins=n_bins, labels=labels)
    df_work = df.copy()
    df_work["bin"] = binned

    results = {}
    for bin_label in df_work["bin"].dropna().unique():
        sub = df_work[df_work["bin"] == bin_label]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], sub[event_col], label=str(bin_label))

        s12 = float(kmf.survival_function_at_times(12).values[0])
        s24 = float(kmf.survival_function_at_times(24).values[0])
        median = kmf.median_survival_time_
        if median == np.inf:
            median = None

        results[str(bin_label)] = {
            "n": int(len(sub)),
            "events": int(sub[event_col].sum()),
            "survival_12_month": round(s12, 4),
            "survival_24_month": round(s24, 4),
            "median_survival": float(median) if median is not None and median != np.inf else None,
        }

    return results


def chiize_all(df, variables=None):
    """
    Run chiizer on multiple variables.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data.
    variables : list, optional
        Variables to chiize. Defaults to key credit risk variables.

    Returns
    -------
    dict
        variable -> bin results dict
    """
    if variables is None:
        variables = [
            "debt_to_income", "LTV_ratio", "interest_rate",
            "employment_years", "loan_amount"
        ]
    all_results = {}
    for var in variables:
        if var not in df.columns:
            continue
        all_results[var] = chiize(df, var, n_bins=4)
    return all_results


def print_chiizer_summary(all_results):
    """Print formatted summary of chiizer results."""
    print("\n" + "=" * 70)
    print("RISK CHIIZER — SURVIVAL BY CONTINUOUS VARIABLE BINS".center(70))
    print("=" * 70)
    for var, bins in all_results.items():
        print(f"\n{var.upper()}".center(70))
        print(f"{'Bin':<10} {'N':>6} {'Events':>7} {'12m Surv':>10} {'24m Surv':>10} {'Median':>10}")
        print("-" * 55)
        for bin_label, data in bins.items():
            med = f"{data['median_survival']:.0f}m" if data["median_survival"] else "∞"
            print(f"{bin_label:<10} {data['n']:>6} {data['events']:>7} "
                  f"{data['survival_12_month']:>10.3f} {data['survival_24_month']:>10.3f} {med:>10}")
    print("=" * 70)


def plot_chiizer_bins(df, variable, results, save_path=None):
    """Plot survival curves for each bin of a variable."""
    binned = bin_variable(df[variable], n_bins=4)
    df_work = df.copy()
    df_work["bin"] = binned

    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()

    for bin_label in df_work["bin"].dropna().unique():
        sub = df_work[df_work["bin"] == bin_label]
        kmf.fit(sub["time_end"], sub["event_default"], label=str(bin_label))
        kmf.plot_survival_function(ax=ax, linewidth=2)

    ax.set_title(f"Survival Curves by {variable.upper()} Quartile", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved chiizer plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    from .data_loader import generate_loan_data

    df = generate_loan_data()
    results = chiize_all(df)
    print_chiizer_summary(results)