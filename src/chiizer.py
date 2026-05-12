"""Risk Chiizer — bin continuous variables into risk categories and compute survival curves."""

import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter


def bin_variable(values, bins, labels):
    """Bin a continuous variable into categories."""
    import pandas as pd
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True)


def chiize(data, var_name, bins, labels, duration="time_end", event="event_default"):
    """Bin variable into risk categories and plot survival curves per bin."""
    data = data.copy()
    data["bin"] = bin_variable(data[var_name], bins, labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    results = {}
    for cat in labels:
        df_cat = data[data["bin"] == cat]
        kmf = KaplanMeierFitter()
        kmf.fit(df_cat[duration], df_cat[event], label=str(cat))
        s12 = float(kmf.survival_function_at_times(12).iloc[0])
        s24 = float(kmf.survival_function_at_times(24).iloc[0])
        results[str(cat)] = {
            "n": len(df_cat),
            "defaults": int(df_cat[event].sum()),
            "survival_12m": round(s12, 4),
            "survival_24m": round(s24, 4),
        }
        kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.set_title(f"Survival by {var_name}")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(title=var_name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"reports/km_{var_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()
    print(f"Saved reports/km_{var_name.replace(' ', '_').lower()}.png")

    return results


def run_chiizer(data):
    """Run chiizer on key credit risk variables."""
    print("\n=== Risk Chiizer ===")

    # DTI bins: Low < 0.20, Medium 0.20-0.35, High 0.35-0.45, Very High > 0.45
    dti_results = chiize(data, "debt_to_income",
                         bins=[-0.001, 0.20, 0.35, 0.45, 1.0],
                         labels=["Low(<20%)", "Medium(20-35%)", "High(35-45%)", "VeryHigh(>45%)"])

    # LTV bins
    ltv_results = chiize(data, "LTV_ratio",
                         bins=[0, 0.60, 0.80, 0.95, 2.0],
                         labels=["Low(<60%)", "Medium(60-80%)", "High(80-95%)", "VeryHigh(>95%)"])

    return {"debt_to_income": dti_results, "LTV_ratio": ltv_results}