"""
Kaplan-Meier survival analysis for credit score bands.

Fits Kaplan-Meier curves for different credit score bands:
- Score < 580: Poor
- Score 580-669: Fair
- Score 670-739: Good
- Score 740+: Excellent

Computes and plots survival functions, median survival times.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def assign_credit_band(score: int) -> str:
    """Assign credit score to a risk band."""
    if score < 580:
        return "Poor (<580)"
    elif score < 670:
        return "Fair (580-669)"
    elif score < 740:
        return "Good (670-739)"
    else:
        return "Excellent (740+)"


def fit_kaplan_meier(df: pd.DataFrame) -> dict:
    """
    Fit Kaplan-Meier curves for each credit score band.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with time_end, event_default, credit_score columns.

    Returns
    -------
    dict
        Dictionary with band names as keys and (kmf, median_time, survival_probs) as values.
    """
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(assign_credit_band)

    bands = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]
    results = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Poor (<580)': '#d62728', 'Fair (580-669)': '#ff7f0e',
              'Good (670-739)': '#2ca02c', 'Excellent (740+)': '#1f77b4'}

    for band in bands:
        band_df = df[df['credit_band'] == band]
        if len(band_df) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(durations=band_df['time_end'],
                event_observed=band_df['event_default'],
                label=band)

        # Median survival time (time where S(t) <= 0.50)
        median_time = kmf.median_survival_time_
        if pd.isna(median_time):
            median_time = "> 24 months (not reached)"

        # 12-month and 24-month survival probabilities
        survival_12 = kmf.predict(12)
        survival_24 = kmf.predict(24)

        results[band] = {
            'kmf': kmf,
            'median_time': median_time,
            'survival_12mo': survival_12,
            'survival_24mo': survival_24,
            'n_events': band_df['event_default'].sum(),
            'n_at_risk': len(band_df)
        }

        kmf.plot_survival_function(ax=ax, color=colors[band])

    ax.set_xlabel("Months since origination", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("reports/km_survival_curves.png", dpi=150)
    plt.close()

    return results


def print_km_summary(results: dict) -> None:
    """Print summary of Kaplan-Meier results."""
    print("\n" + "=" * 70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS BY CREDIT SCORE BAND")
    print("=" * 70)
    print(f"{'Band':<22} {'N':>6} {'Defaults':>8} {'Median Time':>18} {'12mo Surv':>10} {'24mo Surv':>10}")
    print("-" * 70)

    for band in ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]:
        if band not in results:
            continue
        r = results[band]
        median_str = str(r['median_time']) if isinstance(r['median_time'], str) else f"{r['median_time']:.1f} mo"
        print(f"{band:<22} {r['n_at_risk']:>6} {r['n_events']:>8} "
              f"{median_str:>18} {r['survival_12mo']:>10.1%} {r['survival_24mo']:>10.1%}")

    print("=" * 70)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_kaplan_meier(df)
    print_km_summary(results)