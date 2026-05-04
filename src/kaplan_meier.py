"""
Kaplan-Meier survival analysis for loan default.
Estimates survival functions for different credit score bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from typing import Dict, Tuple


def fit_km_by_credit_band(df: pd.DataFrame) -> Dict[str, KaplanMeierFitter]:
    """
    Fit Kaplan-Meier curves for each credit score band.

    Args:
        df: DataFrame with time_end, event_default, credit_band columns

    Returns:
        Dict mapping band label to fitted KaplanMeierFitter
    """
    bands = ['< 580 (Subprime)', '580-669 (Near-prime)', '670-739 (Prime)', '740+ (Super-prime)']
    fitters = {}

    for band in bands:
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=band_df['time_end'],
            event_observed=band_df['event_default'],
            label=band
        )
        fitters[band] = kmf

    return fitters


def compute_median_survival_times(fitters: Dict[str, KaplanMeierFitter]) -> Dict[str, float]:
    """
    Extract median survival times from fitted models.

    Args:
        fitters: Dict of fitted KaplanMeierFitter objects

    Returns:
        Dict mapping band to median survival time (NaN if not reached)
    """
    medians = {}
    for band, kmf in fitters.items():
        medians[band] = kmf.median_survival_time_
    return medians


def compute_survival_probabilities(fitters: Dict[str, KaplanMeierFitter], times: list) -> pd.DataFrame:
    """
    Get survival probabilities at specific time points.

    Args:
        fitters: Dict of fitted KaplanMeierFitter objects
        times: List of time points (months)

    Returns:
        DataFrame with survival probabilities by band and time
    """
    results = {}
    for band, kmf in fitters.items():
        vals = kmf.survival_function_at_times(times).values.flatten()
        results[band] = list(vals)

    return pd.DataFrame(results, index=[f'{t}-month' for t in times]).T


def plot_km_curves(fitters: Dict[str, KaplanMeierFitter], save_path: str = None) -> plt.Figure:
    """
    Plot Kaplan-Meier survival curves for all credit bands.

    Args:
        fitters: Dict of fitted KaplanMeierFitter objects
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'< 580 (Subprime)': '#d62728', '580-669 (Near-prime)': '#ff7f0e',
              '670-739 (Prime)': '#2ca02c', '740+ (Super-prime)': '#1f77b4'}

    for band, kmf in fitters.items():
        kmf.plot_survival_function(ax=ax, color=colors.get(band, None))

    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 60)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def print_km_summary(fitters: Dict[str, KaplanMeierFitter], medians: Dict[str, float]):
    """Print formatted Kaplan-Meier summary."""
    print("\n" + "="*70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS")
    print("="*70)

    print("\n📊 Median Time-to-Default by Credit Band:")
    print("-" * 50)
    for band, median in medians.items():
        if np.isnan(median):
            print(f"  {band:<30} Not reached (>50% survive)")
        else:
            print(f"  {band:<30} {median:.1f} months")

    print("\n📈 Survival Probabilities by Time Horizon:")
    print("-" * 50)
    prob_df = compute_survival_probabilities(fitters, [12, 24, 36, 48])
    print(f"{'Credit Band':<30} {'12m':>8} {'24m':>8} {'36m':>8} {'48m':>8}")
    print("-" * 50)
    for band in prob_df.index:
        print(f"{band:<30} {prob_df.loc[band, '12-month']:>8.3f} {prob_df.loc[band, '24-month']:>8.3f} "
              f"{prob_df.loc[band, '36-month']:>8.3f} {prob_df.loc[band, '48-month']:>8.3f}")

    return prob_df


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands

    df = add_credit_bands(generate_loan_data())
    fitters = fit_km_by_credit_band(df)
    medians = compute_median_survival_times(fitters)
    prob_df = print_km_summary(fitters, medians)
    plot_km_curves(fitters, 'km_curves.png')
    print("\n✅ Kaplan-Meier analysis complete")