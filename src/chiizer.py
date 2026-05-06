import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def build_risk_chiizer(df):
    """Bin continuous variables and compute survival curves per bin.

    Returns dict of chiizer results for:
        - debt_to_income bins (Low <0.20, Medium 0.20-0.35, High >0.35)
        - LTV_ratio bins (Low <0.50, Medium 0.50-0.75, High >0.75)
        - employment_years bins (0-2, 2-5, 5+)
    """
    kmf = KaplanMeierFitter()
    results = {}

    # --- DTI chiizer ---
    dti_bands = [
        ("Low DTI (<20%)",  df["debt_to_income"] < 0.20),
        ("Med DTI (20-35%)", (df["debt_to_income"] >= 0.20) & (df["debt_to_income"] <= 0.35)),
        ("High DTI (>35%)",  df["debt_to_income"] > 0.35),
    ]
    results["debt_to_income"] = _chiize(kmf, df, dti_bands, "Debt-to-Income Ratio")

    # --- LTV chiizer ---
    ltv_bands = [
        ("Low LTV (<50%)",  df["LTV_ratio"] < 0.50),
        ("Med LTV (50-75%)", (df["LTV_ratio"] >= 0.50) & (df["LTV_ratio"] <= 0.75)),
        ("High LTV (>75%)",  df["LTV_ratio"] > 0.75),
    ]
    results["LTV_ratio"] = _chiize(kmf, df, ltv_bands, "LTV Ratio")

    # --- Employment chiizer ---
    emp_bands = [
        ("0-2 years",    df["employment_years"] <= 2),
        ("2-5 years",    (df["employment_years"] > 2) & (df["employment_years"] <= 5)),
        ("5+ years",     df["employment_years"] > 5),
    ]
    results["employment_years"] = _chiize(kmf, df, emp_bands, "Employment Years")

    # Save combined plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (var, data) in zip(axes, results.items()):
        for label, km_data in data["curves"].items():
            ax.plot(km_data["timeline"], km_data["survival"], label=label, linewidth=2)
        ax.set_title(data["title"])
        ax.set_xlabel("Months")
        ax.set_ylabel("Survival Probability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/workspace/Projects/survival-analysis-time-to-default/reports/chiizer_curves.png", dpi=150)
    plt.close()
    print("Saved: reports/chiizer_curves.png")

    return results

def _chiize(kmf, df, bands, title):
    """Internal: fit KM for each bin and return curve data."""
    curves = {}
    for label, mask in bands:
        band_df = df[mask]
        if len(band_df) < 5:
            continue
        kmf.fit(band_df["time_end"], event_observed=band_df["event_default"], label=label)
        timeline = np.arange(0, 37)
        sf = kmf.survival_function_at_times(timeline)
        curves[label] = {
            "timeline": timeline.tolist(),
            "survival": [float(sf[t]) if t in sf.index else None for t in timeline],
        }
    return {"title": title, "curves": curves}

if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = build_risk_chiizer(df)
    print("Chiizer complete.")