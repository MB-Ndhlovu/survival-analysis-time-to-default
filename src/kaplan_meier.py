"""
Kaplan-Meier survival analysis for credit score bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def fit_km_by_credit_band(df):
    """
    Fit Kaplan-Meier curves for each credit score band.

    Credit score bands:
    - Deep Subprime: < 580
    - Subprime: 580-669
    - Near Prime: 670-739
    - Prime: 740+
    """
    bands = [
        ('Deep Subprime (<580)', df['credit_score'] < 580),
        ('Subprime (580-669)', (df['credit_score'] >= 580) & (df['credit_score'] < 670)),
        ('Near Prime (670-739)', (df['credit_score'] >= 670) & (df['credit_score'] < 740)),
        ('Prime (740+)', df['credit_score'] >= 740),
    ]

    results = {}
    kmf_global = KaplanMeierFitter()

    for label, mask in bands:
        band_df = df[mask].copy()
        if len(band_df) < 10:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=band_df['time_end'],
            event_observed=band_df['event_default'],
            label=label
        )

        results[label] = {
            'kmf': kmf,
            'n_observations': int(mask.sum()),
            'n_defaults': int(band_df['event_default'].sum()),
            'n_censored': int((band_df['event_default'] == 0).sum()),
            'median_survival': kmf.median_survival_time_,
        }

        kmf_global.fit(
            durations=band_df['time_end'],
            event_observed=band_df['event_default'],
            label=label
        )

    return results, kmf_global


def plot_survival_curves(results, save_path=None):
    """Plot Kaplan-Meier survival curves for all credit bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Deep Subprime (<580)': '#d62728',
              'Subprime (580-669)': '#ff7f0e',
              'Near Prime (670-739)': '#2ca02c',
              'Prime (740+)': '#1f77b4'}

    for label, data in results.items():
        kmf = data['kmf']
        color = colors.get(label, None)
        kmf.plot_survival_function(ax=ax, color=color, label=label)

    ax.set_xlabel('Months Since Origination')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Kaplan-Meier plot to {save_path}")

    return fig


def compute_survival_probabilities(results, horizons=[12, 24]):
    """Compute survival probabilities at specific time horizons."""
    probs = {}
    for label, data in results.items():
        kmf = data['kmf']
        probs[label] = {}
        for t in horizons:
            try:
                prob = kmf.survival_function_at_times(t).values[0]
            except Exception:
                prob = np.nan
            probs[label][f'{t}_month'] = round(prob, 4)

    return probs


def print_km_summary(results, horizons=[12, 24]):
    """Print summary of Kaplan-Meier results."""
    print("\n" + "="*70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS BY CREDIT SCORE BAND")
    print("="*70)

    for label, data in results.items():
        kmf = data['kmf']
        n_obs = data['n_observations']
        n_def = data['n_defaults']
        n_cen = data['n_censored']
        median_survival = data['median_survival']

        print(f"\n{label}")
        print(f"  Loans: {n_obs} | Defaults: {n_def} | Censored: {n_cen}")
        print(f"  Median Survival Time: {median_survival:.1f} months")

        for t in horizons:
            try:
                prob = kmf.survival_function_at_times(t).values[0]
                print(f"  {t}-Month Survival Probability: {prob:.4f} ({prob*100:.2f}%)")
            except Exception:
                print(f"  {t}-Month Survival Probability: N/A")

    print("\n" + "="*70)


if __name__ == '__main__':
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results, _ = fit_km_by_credit_band(df)
    print_km_summary(results)
    plot_survival_curves(results)