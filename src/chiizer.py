"""Risk Chiizer — bin continuous variables into risk categories."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def chiize(df: pd.DataFrame, variable: str, n_bins: int = 4) -> pd.DataFrame:
    """
    Bin a continuous variable into quantiles and compute survival curves per bin.
    Returns a DataFrame with survival stats per bin.
    """
    df = df.copy()

    # Create quantile bins
    try:
        df[f"{variable}_bin"] = pd.qcut(df[variable], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback: equal-width bins
        df[f"{variable}_bin"] = pd.cut(df[variable], bins=n_bins, labels=False)

    kmf = KaplanMeierFitter()
    results = []

    for bin_val in sorted(df[f"{variable}_bin"].dropna().unique()):
        subset = df[df[f"{variable}_bin"] == bin_val]
        kmf.fit(
            subset["time_end"],
            event_observed=subset["event_default"],
            label=f"Q{int(bin_val)+1}",
        )

        s12 = kmf.survival_function_at_times(12).values[0]
        s24 = kmf.survival_function_at_times(24).values[0]
        median = kmf.median_survival_time_
        if np.isnan(median):
            median = "> 60"

        results.append({
            "bin": int(bin_val),
            "label": f"Q{int(bin_val)+1}",
            "n": len(subset),
            "defaults": int(subset["event_default"].sum()),
            "12m_survival": round(s12, 4),
            "24m_survival": round(s24, 4),
            "median_survival": median,
        })

    result_df = pd.DataFrame(results)
    return result_df


def plot_chiized_curves(df: pd.DataFrame, variable: str, result_df: pd.DataFrame):
    """Plot survival curves for each bin of a chiized variable."""
    df = df.copy()
    df[f"{variable}_bin"] = pd.qcut(df[variable], q=4, labels=False, duplicates="drop")

    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for bin_val in sorted(df[f"{variable}_bin"].dropna().unique()):
        subset = df[df[f"{variable}_bin"] == bin_val]
        kmf.fit(
            subset["time_end"],
            event_observed=subset["event_default"],
            label=f"Q{int(bin_val)+1}",
        )
        kmf.plot_survival_function(ax=ax, show_censors=True)

    ax.set_title(f"Survival Curves by {variable.upper()} Quartiles")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"/home/workspace/Projects/survival-analysis-time-to-default/reports/km_{variable}.png",
        dpi=150,
    )
    plt.close()


def run_chiizer(df: pd.DataFrame) -> dict:
    """Run chiizer on key continuous variables."""
    variables = ["credit_score", "debt_to_income", "LTV_ratio", "interest_rate"]
    results = {}

    for var in variables:
        try:
            result_df = chiize(df, var)
            plot_chiized_curves(df, var, result_df)
            results[var] = result_df.to_dict(orient="records")
        except Exception as e:
            print(f"Warning: could not chiize {var}: {e}")

    return results


if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    results = run_chiizer(df)
    for var, res in results.items():
        print(f"\n=== {var} ===")
        for row in res:
            print(row)