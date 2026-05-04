"""Kaplan-Meier survival analysis for credit score bands."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def get_credit_band(score):
    """Assign credit score to risk band."""
    if score < 580:
        return 'Deep Subprime (<580)'
    elif score < 670:
        return 'Subprime (580-669)'
    elif score < 740:
        return 'Near Prime (670-739)'
    else:
        return 'Prime (740+)'

def fit_kaplan_meier(df):
    """Fit Kaplan-Meier curves for each credit score band.

    Args:
        df: DataFrame with time_start, time_end, event_default columns

    Returns:
        dict with bands as keys and (kmf, median_survival) as values
    """
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(get_credit_band)

    results = {}
    bands = ['Deep Subprime (<580)', 'Subprime (580-669)', 'Near Prime (670-739)', 'Prime (740+)']

    for band in bands:
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df['time_end'],
            band_df['event_default'],
            label=band
        )
        median = kmf.median_survival_time_
        results[band] = {'kmf': kmf, 'median_survival': median}

    return results

def plot_survival_curves(results, save_path='reports/km_survival_curves.png'):
    """Plot Kaplan-Meier survival curves for all credit bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Deep Subprime (<580)': '#d62728',
              'Subprime (580-669)': '#ff7f0e',
              'Near Prime (670-739)': '#2ca02c',
              'Prime (740+)': '#1f77b4'}

    for band, data in results.items():
        kmf = data['kmf']
        ax = kmf.plot_survival_function(ax=ax, color=colors[band], linewidth=2)

    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path

def get_survival_probabilities(results, months=[12, 24]):
    """Get survival probabilities at specified months for each band."""
    probs = {}
    for band, data in results.items():
        kmf = data['kmf']
        probs[band] = {}
        for m in months:
            try:
                probs[band][f'{m}_month'] = kmf.survival_function_at_times(m).values[0]
            except:
                probs[band][f'{m}_month'] = None
    return probs

if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_kaplan_meier(df)

    print("Median Survival Time by Credit Band:")
    for band, data in results.items():
        median = data['median_survival']
        print(f"  {band}: {median:.1f} months" if median else f"  {band}: Not reached")

    plot_survival_curves(results)
    print("Saved km_survival_curves.png")