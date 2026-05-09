"""
Kaplan-Meier survival curves for credit score bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def fit_km_by_band(df: pd.DataFrame, time_col: str = 'time_end',
                   event_col: str = 'event_default') -> dict:
    """
    Fit Kaplan-Meier curves for each credit score band.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with credit_score and event columns
    time_col : str
        Column containing time to event or censoring
    event_col : str
        Column containing event indicator (1=default, 0=censored)

    Returns
    -------
    dict
        Dictionary mapping band labels to KaplanMeierFitter objects
    """
    bands = {
        'Subprime (<580)': (df['credit_score'] < 580),
        'Near-Prime (580-669)': (df['credit_score'] >= 580) & (df['credit_score'] < 670),
        'Prime (670-739)': (df['credit_score'] >= 670) & (df['credit_score'] < 740),
        'Super-Prime (740+)': (df['credit_score'] >= 740)
    }

    fitters = {}
    results = {}

    for band_name, mask in bands.items():
        band_df = df[mask].copy()
        if len(band_df) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(band_df[time_col], band_df[event_col], label=band_name)

        fitters[band_name] = kmf

        # Median survival time (time at which S(t) = 0.5)
        try:
            median_time = median_survival_times(kmf.survival_function_)
            if hasattr(median_time, 'iloc'):
                median_time = float(median_time.iloc[0])
            else:
                median_time = float(median_time)
        except Exception:
            median_time = None

        sf = kmf.survival_function_
        s12 = float(sf.iloc[11].values[0]) if len(sf) > 11 else None
        s24 = float(sf.iloc[23].values[0]) if len(sf) > 23 else None

        results[band_name] = {
            'n': len(band_df),
            'events': int(band_df[event_col].sum()),
            'median_survival_months': round(median_time, 2) if median_time else None,
            'survival_12m': round(s12, 4) if s12 else None,
            'survival_24m': round(s24, 4) if s24 else None
        }

    return fitters, results


def compute_survival_probabilities(kmf: KaplanMeierFitter,
                                    months: list = None) -> pd.DataFrame:
    """Extract survival probabilities at specific months."""
    if months is None:
        months = [6, 12, 18, 24, 36]

    sf = kmf.survival_function_
    result = {}
    for m in months:
        if m < len(sf):
            result[f'S({m}m)'] = round(float(sf.iloc[m].values[0]), 4)

    return pd.DataFrame([result])


def plot_km_curves(fitters: dict, save_path: str = None):
    """
    Plot Kaplan-Meier survival curves for all credit score bands.

    Parameters
    ----------
    fitters : dict
        Dictionary of fitted KaplanMeierFitter objects
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Subprime (<580)': '#d62728',
              'Near-Prime (580-669)': '#ff7f0e',
              'Prime (670-739)': '#2ca02c',
              'Super-Prime (740+)': '#1f77b4'}

    for band_name, kmf in fitters.items():
        kmf.plot_survival_function(ax=ax, color=colors.get(band_name, None))

    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band', fontsize=14)
    ax.set_xlabel('Months Since Loan Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add reference line at S=0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Median')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def main(df: pd.DataFrame):
    fitters, results = fit_km_by_band(df)

    print("\n=== Kaplan-Meier Results by Credit Score Band ===\n")
    for band, res in results.items():
        print(f"{band}")
        print(f"  N={res['n']}, Events={res['events']}")
        print(f"  Median Survival: {res['median_survival_months']} months")

    return fitters, results


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    main(df)