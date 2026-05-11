"""Risk chiizer — bin continuous variables into risk categories and compare survival."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(series: pd.Series, n_bins: int = 4, labels: list = None) -> pd.Series:
    """Bin a continuous variable into quantiles."""
    bins = pd.qcut(series, q=n_bins, duplicates="drop")
    if labels:
        return bins.rename(bins.name + "_bin").map(dict(zip(bins.cat.categories, labels)))
    return bins


def build_risk_chiizer(df: pd.DataFrame, output_path: str = "reports/risk_chiizer.png") -> dict:
    """
    Bin continuous risk factors into categories, compute survival curves for each bin,
    and identify which factors drive the most separation in default timing.
    """
    kmf = KaplanMeierFitter()
    results = {}

    variables = ["credit_score", "debt_to_income", "LTV_ratio", "interest_rate"]
    variable_labels = {
        "credit_score": "Credit Score",
        "debt_to_income": "Debt-to-Income",
        "LTV_ratio": "LTV Ratio",
        "interest_rate": "Interest Rate",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, var in enumerate(variables):
        ax = axes[idx]

        # Create 4 bins
        try:
            binned = bin_variable(df[var], n_bins=4)
        except ValueError:
            # Handle duplicate bin edges
            binned = pd.cut(df[var], bins=4)

        bin_labels = sorted(binned.cat.categories.astype(str).tolist())
        df_temp = df.copy()
        df_temp[f"{var}_bin"] = binned.astype(str)

        # Fit KM for each bin
        for bin_val in bin_labels:
            subset = df_temp[df_temp[f"{var}_bin"] == bin_val]
            kmf.fit(subset["time_end"], subset["event_default"], label=bin_val)
            kmf.plot_survival_function(ax=ax, ci_show=False)

        ax.set_title(variable_labels.get(var, var))
        ax.set_xlabel("Months")
        ax.set_ylabel("Survival Probability")
        ax.set_xlim(0, 30)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Compute log-rank p-value for this variable
        from lifelines.statistics import logrank_test
        groups = df_temp[f"{var}_bin"].unique()
        if len(groups) >= 2:
            g1 = df_temp[df_temp[f"{var}_bin"] == groups[0]]
            g2 = df_temp[df_temp[f"{var}_bin"] == groups[-1]]
            try:
                test = logrank_test(g1["time_end"], g2["time_end"],
                                   g1["event_default"], g2["event_default"])
                results[var] = {
                    "logrank_p_value": round(test.p_value, 6),
                    "significant": test.p_value < 0.05,
                    "n_bins": len(groups),
                }
            except Exception:
                results[var] = {"logrank_p_value": None, "significant": False, "n_bins": len(groups)}

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return results


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = build_risk_chiizer(df)
    print("\nRisk Chiizer Results (log-rank test):")
    for var, res in results.items():
        sig = "SIGNIFICANT" if res["significant"] else "not significant"
        print(f"  {var}: p={res['logrank_p_value']} ({sig})")