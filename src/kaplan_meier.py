"""
Kaplan-Meier survival analysis for credit score bands.
Fits KM curves, computes median survival times, and generates plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def fit_km_by_credit_band(df, time_col='time_end', event_col='event_default'):
    """
    Fit Kaplan-Meier curves for each credit score band.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time and event columns
    time_col : str
        Name of time-to-event column
    event_col : str
        Name of event indicator column (1=default, 0=censored)

    Returns
    -------
    dict
        Dictionary with band names as keys and KM fitters as values
    """
    bands = df['credit_band'].unique()
    kmfitters = {}

    for band in sorted(bands):
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df[time_col],
            band_df[event_col],
            label=band
        )
        kmfitters[band] = kmf

    return kmfitters


def compute_median_survival(kmfitters):
    """
    Compute median survival time for each KM fitter.

    Returns
    -------
    pd.DataFrame
        DataFrame with band, median_survival, and survival probabilities
    """
    results = []
    for band, kmf in kmfitters.items():
        median = kmf.median_survival_time_
        results.append({
            'band': band,
            'median_survival_months': round(median, 2) if np.isfinite(median) else None,
            'survival_at_12': round(kmf.survival_function_at_times(12).values[0], 4),
            'survival_at_24': round(kmf.survival_function_at_times(24).values[0], 4),
        })

    return pd.DataFrame(results)


def plot_km_curves(kmfitters, output_path=None):
    """
    Plot Kaplan-Meier survival curves for all credit bands.

    Parameters
    ----------
    kmfitters : dict
        Dictionary of fitted KM fitters
    output_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'Deep Subprime (<580)': '#d62728',
        'Subprime (580-669)': '#ff7f0e',
        'Near Prime (670-739)': '#2ca02c',
        'Prime (740+)': '#1f77b4',
    }

    for band, kmf in kmfitters.items():
        kmf.plot_survival_function(ax=ax, color=colors.get(band), linewidth=2)

    ax.set_xlabel('Months Since Loan Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band\n(Time to Default)', fontsize=14)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)

    # Add annotation for median survival
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% survival')
    ax.text(23.5, 0.52, '50%', fontsize=9, color='gray', va='bottom')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved Kaplan-Meier plot to {output_path}")

    return fig


def survival_table(kmf, time_points=None):
    """Generate survival table at specific time points."""
    if time_points is None:
        time_points = [6, 12, 18, 24]

    table = []
    for t in time_points:
        surv_prob = kmf.survival_function_at_times(t).values[0]
        ci_lower = kmf.confidence_interval_survival_function_.iloc[
            (kmf.survival_function_.index <= t).sum() - 1, 0
        ]
        ci_upper = kmf.confidence_interval_survival_function_.iloc[
            (kmf.survival_function_.index <= t).sum() - 1, 1
        ]
        table.append({
            'time_months': t,
            'survival_probability': round(surv_prob, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4)
        })

    return pd.DataFrame(table)


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_band

    df = generate_loan_data(5000)
    df = add_credit_band(df)

    kmfitters = fit_km_by_credit_band(df)
    medians = compute_median_survival(kmfitters)
    print("Median Survival by Credit Band:")
    print(medians.to_string(index=False))

    plot_km_curves(kmfitters, '/tmp/km_curves.png')
    print("\nSaved plot to /tmp/km_curves.png")