"""Risk Chiizer: Bin continuous variables into risk categories."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from .data_loader import get_credit_score_band


def chiize_variable(
    df: pd.DataFrame,
    variable: str,
    n_bins: int = 4,
    labels: list = None,
    time_col: str = "time_end",
    event_col: str = "event_default",
) -> tuple:
    """Bin a continuous variable into risk categories and compute survival curves.

    Args:
        df: Input DataFrame
        variable: Column name to bin
        n_bins: Number of bins
        labels: Custom bin labels
        time_col: Time column
        event_col: Event indicator column

    Returns:
        (binned DataFrame, survival summary DataFrame, kmf dict)
    """
    df = df.copy()

    # Create bins
    if labels is None:
        percentiles = np.linspace(0, 100, n_bins + 1)
        labels = [f"Q{i+1}" for i in range(n_bins)]

    df[f"{variable}_bin"] = pd.qcut(df[variable], q=n_bins, labels=labels, duplicates="drop")

    kmf_dict = {}
    results = []

    for bin_val in df[f"{variable}_bin"].unique():
        subset = df[df[f"{variable}_bin"] == bin_val]
        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], subset[event_col], label=f"{variable}: {bin_val}")

        survival_12 = kmf.survival_function_at_times(12).values[0]
        survival_24 = kmf.survival_function_at_times(24).values[0]
        median = kmf.median_survival_time_
        if pd.isna(median):
            median = ">24 months"

        results.append({
            "bin": bin_val,
            "n": len(subset),
            "defaults": int(subset[event_col].sum()),
            "default_rate": round(subset[event_col].mean(), 4),
            "survival_12m": round(survival_12, 4),
            "survival_24m": round(survival_24, 4),
            "median_survival": median,
        })

        kmf_dict[str(bin_val)] = kmf

    return df, pd.DataFrame(results), kmf_dict


def plot_chiized_variable(
    kmf_dict: dict,
    title: str,
    output_path: str,
    color_scheme: str = "viridis",
) -> None:
    """Plot survival curves for binned variable."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.get_cmap(color_scheme)
    for i, (bin_val, kmf) in enumerate(kmf_dict.items()):
        color = cmap(i / max(len(kmf_dict) - 1, 1))
        kmf.plot_survival_function(ax=ax, color=color)

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(f"Survival Curves by {title}", fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Chiizer plot saved to {output_path}")


def run(df: pd.DataFrame) -> dict:
    """Run chiizer on key variables."""
    print("\n=== Risk Chiizer Analysis ===")

    variables = ["debt_to_income", "LTV_ratio", "loan_amount", "interest_rate"]

    results = {}

    for var in variables:
        print(f"\n--- {var.upper()} ---")

        # Determine appropriate labels based on variable
        if var == "debt_to_income":
            labels = ["Low (<15%)", "Medium (15-25%)", "High (25-35%)", "Very High (>35%)"]
        elif var == "LTV_ratio":
            labels = ["Low (<60%)", "Medium (60-75%)", "High (75-85%)", "Very High (>85%)"]
        elif var == "loan_amount":
            labels = ["Small (<50k)", "Medium (50-100k)", "Large (100-200k)", "Very Large (>200k)"]
        elif var == "interest_rate":
            labels = ["Low (<6%)", "Medium (6-9%)", "High (9-12%)", "Very High (>12%)"]
        else:
            labels = None

        df, summary, kmfs = chiize_variable(
            df, var, n_bins=4, labels=labels
        )

        print(summary.to_string(index=False))
        results[var] = {"summary": summary, "kmfs": kmfs, "df": df}

        # Plot
        plot_chiized_variable(
            kmfs,
            title=var.replace("_", " ").title(),
            output_path=f"reports/{var}_chiized.png",
        )

    return results


if __name__ == "__main__":
    from .data_loader import generate_loan_data

    df = generate_loan_data()
    results = run(df)