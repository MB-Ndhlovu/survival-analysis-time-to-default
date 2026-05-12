"""Kaplan-Meier survival analysis stratified by credit score band."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

CREDIT_BANDS = [
    ('Deep Subprime (< 580)', 300, 579),
    ('Subprime (580-669)', 580, 669),
    ('Near Prime (670-739)', 670, 739),
    ('Prime (740+)', 740, 850),
]

def assign_credit_band(score):
    for label, low, high in CREDIT_BANDS:
        if low <= score <= high:
            return label
    return 'Unknown'

def fit_kaplan_meier(df):
    """
    Fit KM curves per credit score band.
    Returns dict of {band_label: {'kmf': fitted KM, 'median': median survival time}}
    """
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(assign_credit_band)

    results = {}
    for label, low, high in CREDIT_BANDS:
        band_df = df[df['credit_band'] == label].copy()
        if len(band_df) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df['time_end'],
            band_df['event_default'],
            label=label
        )
        median = kmf.median_survival_time_
        results[label] = {'kmf': kmf, 'median': median, 'n': len(band_df)}

    return results

def plot_km_curves(results, save_path=None):
    """Plot KM survival curves by credit score band."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for (label, km_result), color in zip(results.items(), colors):
        kmf = km_result['kmf']
        median = km_result['median']
        n = km_result['n']
        ax.plot(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0],
               label=f"{label} (n={n}, median={median:.1f}m)", color=color, linewidth=2)

    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved KM plot to {save_path}")

    return fig

def compute_survival_at_t(results, t):
    """Compute survival probability at time t for each band."""
    out = {}
    for label, km_result in results.items():
        surv = km_result['kmf'].survival_function_at_times(t).values[0]
        out[label] = round(surv, 4)
    return out

def km_summary(results):
    """Print summary of KM results."""
    print("\n=== Kaplan-Meier Results by Credit Band ===")
    for label, km_result in results.items():
        kmf = km_result['kmf']
        median = km_result['median']
        n = km_result['n']
        s12 = kmf.survival_function_at_times(12).values[0]
        s24 = kmf.survival_function_at_times(24).values[0]
        print(f"\n{label}")
        print(f"  n = {n}")
        print(f"  12-month survival: {s12:.1%}")
        print(f"  24-month survival: {s24:.1%}")
        print(f"  Median survival time: {median:.1f} months" if not np.isinf(median) else f"  Median survival time: > 60 months (not reached)")
    return results