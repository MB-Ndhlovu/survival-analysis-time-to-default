"""
Risk Chiizer — bin continuous variables into risk categories and compute
survival curves for each bin to surface risk drivers.
"""

import pandas as pd
from lifelines import KaplanMeierFitter


def bin_variable(series: pd.Series, bins: tuple, labels: tuple) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def chiize(df: pd.DataFrame) -> dict:
    """
    Bin key continuous variables and compute survival statistics per bin.
    """
    kmf = KaplanMeierFitter()

    configs = {
        "credit_score": {
            "bins": [0, 580, 670, 740, 850],
            "labels": ["Deep Subprime(<580)", "Subprime(580-669)", "Near Prime(670-739)", "Prime(740+)"],
        },
        "debt_to_income": {
            "bins": [0, 0.20, 0.35, 0.50, 1.0],
            "labels": ["Low(<20%)", "Medium(20-35%)", "High(35-50%)", "Very High(50%+)"],
        },
        "LTV_ratio": {
            "bins": [0, 0.60, 0.80, 0.95, 1.5],
            "labels": ["Low(<60%)", "Medium(60-80%)", "High(80-95%)", "Very High(95%+)"],
        },
        "interest_rate": {
            "bins": [0, 0.08, 0.12, 0.18, 1.0],
            "labels": ["Low(<8%)", "Medium(8-12%)", "High(12-18%)", "Very High(18%+)"],
        },
    }

    results = {}

    for var, cfg in configs.items():
        results[var] = {}
        binned = bin_variable(df[var], cfg["bins"], cfg["labels"])

        for label in cfg["labels"]:
            sub = df[binned == label]
            if sub.empty:
                continue

            kmf.fit(sub["time_end"], sub["event_default"])
            results[var][label] = {
                "n": int(len(sub)),
                "events": int(sub["event_default"].sum()),
                "survival_12m": float(kmf.survival_function_at_times(12).values[0]),
                "survival_24m": float(kmf.survival_function_at_times(24).values[0]),
            }

    return results


def print_chiizer_summary(results: dict) -> str:
    lines = ["\n" + "=" * 60]
    lines.append("RISK CHIIZER — SURVIVAL BY RISK CATEGORY")
    lines.append("=" * 60)

    for var, bins in results.items():
        lines.append(f"\n{var.upper().replace('_', ' ')}")
        lines.append("-" * 50)
        lines.append(f"{'Category':<28} {'N':>6} {'Events':>7} {'S(12)':>8} {'S(24)':>8}")
        lines.append("-" * 50)

        for label, stats in bins.items():
            lines.append(
                f"{label:<28} {stats['n']:>6} {stats['events']:>7} "
                f"{stats['survival_12m']:>7.1%} {stats['survival_24m']:>7.1%}"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = chiize(df)
    print(print_chiizer_summary(results))