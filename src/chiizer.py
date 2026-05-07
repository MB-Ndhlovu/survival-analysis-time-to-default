"""
Risk Chiizer: bin continuous variables into risk categories and compute survival curves.
"""
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


def bin_variable(series: pd.Series, n_bins: int = 4, method: str = "quantile") -> pd.Series:
    """
    Bin a continuous variable into risk categories.

    method='quantile': equal-size bins (recommended for credit scores, income)
    method='equal': equal-width bins
    """
    if method == "quantile":
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    else:
        return pd.cut(series, bins=n_bins, labels=False, include_lowest=True)


def chiize(df: pd.DataFrame, var: str, n_bins: int = 4) -> dict:
    """
    Bin a variable, fit KM curves per bin, return results.

    Returns dict with:
    - bins_df: bin boundaries and counts
    - kmf_by_bin: fitted KMFs per bin
    - survival_df: survival probabilities at key months
    """
    binned = bin_variable(df[var], n_bins=n_bins)
    df_work = df.copy()
    df_work["bin"] = binned

    bin_labels = sorted(df_work["bin"].dropna().unique())
    kmf_by_bin = {}
    bin_stats = []

    for b in bin_labels:
        mask = df_work["bin"] == b
        sub = df_work[mask]
        if len(sub) < 10:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(sub["time_end"], sub["event_default"], label=f"Bin {b}")
        kmf_by_bin[b] = kmf

        # Bin boundaries
        bin_stats.append({
            "bin": int(b),
            "n": len(sub),
            f"{var}_min": sub[var].min(),
            f"{var}_max": sub[var].max(),
            f"{var}_mean": round(sub[var].mean(), 2),
            "default_rate": round(sub["event_default"].mean(), 4),
        })

    return {
        "bins_df": pd.DataFrame(bin_stats),
        "kmf_by_bin": kmf_by_bin,
    }


def chiize_all_vars(df: pd.DataFrame) -> dict:
    """
    Chiize all continuous covariates.
    """
    vars_to_chiize = [
        "credit_score",
        "income",
        "debt_to_income",
        "loan_amount",
        "LTV_ratio",
    ]

    results = {}
    for var in vars_to_chiize:
        results[var] = chiize(df, var, n_bins=4)
        print(f"\n=== {var.upper()} Risk Bins ===")
        print(results[var]["bins_df"].to_string(index=False))

    return results


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data(5000)
    chiize_all_vars(df)