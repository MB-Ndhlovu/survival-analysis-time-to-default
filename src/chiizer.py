"""Risk chiizer: bin continuous variables into risk categories."""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


# Bin definitions
BIN_DEFINITIONS = {
    "credit_score": {
        "bins": [0, 580, 670, 740, 850],
        "labels": ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"],
    },
    "debt_to_income": {
        "bins": [0, 0.2, 0.35, 0.5, 2.0],
        "labels": ["Low (<20%)", "Moderate (20-35%)", "High (35-50%)", "Very High (>50%)"],
    },
    "LTV_ratio": {
        "bins": [0, 0.6, 0.8, 0.95, 1.5],
        "labels": ["Low (<60%)", "Moderate (60-80%)", "High (80-95%)", "Very High (>95%)"],
    },
}


def bin_variable(df: pd.DataFrame, var: str) -> pd.Series:
    """Bin a continuous variable into risk categories."""
    bins = BIN_DEFINITIONS[var]["bins"]
    labels = BIN_DEFINITIONS[var]["labels"]
    return pd.cut(df[var], bins=bins, labels=labels, include_lowest=True)


def compute_segment_survival(
    df: pd.DataFrame,
    segment_var: str,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> dict:
    """
    Compute survival statistics for each segment of a variable.

    Returns dict with segment -> {n, n_events, survival_12, survival_24, median}
    """
    df = df.copy()
    df["segment"] = bin_variable(df, segment_var)

    results = {}
    kmf = KaplanMeierFitter()

    for segment in df["segment"].dropna().unique():
        seg_df = df[df["segment"] == segment]
        n = len(seg_df)
        n_events = seg_df[event_col].sum()

        kmf.fit(seg_df[duration_col], seg_df[event_col], label=str(segment))

        s12 = float(kmf.survival_function_at_times([12]).values[0])
        s24 = float(kmf.survival_function_at_times([24]).values[0])

        try:
            median_vals = kmf.median_survival_time_
            if hasattr(median_vals, "iloc"):
                median = float(median_vals.iloc[0]) if len(median_vals) > 0 else None
            else:
                median = float(median_vals) if not pd.isna(median_vals) else None
        except Exception:
            median = None

        results[str(segment)] = {
            "n": n,
            "n_events": int(n_events),
            "survival_12": s12,
            "survival_24": s24,
            "median_survival": median,
        }

    return results


def build_risk_chiizer(
    df: pd.DataFrame,
    duration_col: str = "time_end",
    event_col: str = "event_default",
) -> dict:
    """
    Build full risk chiizer: compute survival curves for all risk variables.

    Returns dict with variable -> segment_stats
    """
    chiizer = {}

    for var in ["credit_score", "debt_to_income", "LTV_ratio"]:
        chiizer[var] = compute_segment_survival(df, var, duration_col, event_col)

    return chiizer


def print_chiizer_results(chiizer: dict) -> str:
    """Format chiizer results for display."""
    lines = ["\n=== Risk Chiizer Results ===\n"]

    var_labels = {
        "credit_score": "Credit Score Bands",
        "debt_to_income": "Debt-to-Income Ratio",
        "LTV_ratio": "Loan-to-Value Ratio",
    }

    for var, label in var_labels.items():
        if var not in chiizer:
            continue
        lines.append(f"\n{label}:\n")
        lines.append(f"{'Segment':<22} {'N':>6} {'Defaults':>8} {'S(12)':>8} {'S(24)':>8} {'Median':>10}\n")
        lines.append("-" * 64)

        for seg, stats in chiizer[var].items():
            median_str = f"{stats['median_survival']:.1f}" if stats["median_survival"] else "N/A"
            lines.append(
                f"{seg:<22} {stats['n']:>6} {stats['n_events']:>8} "
                f"{stats['survival_12']:>8.1%} {stats['survival_24']:>8.1%} {median_str:>10}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    chiizer = build_risk_chiizer(df)
    print(print_chiizer_results(chiizer))