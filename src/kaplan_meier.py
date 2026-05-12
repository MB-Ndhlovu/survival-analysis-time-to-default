"""Kaplan-Meier survival analysis by credit score bands."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def assign_credit_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit score band column to DataFrame."""
    conditions = [
        df['credit_score'] < 580,
        (df['credit_score'] >= 580) & (df['credit_score'] < 670),
        (df['credit_score'] >= 670) & (df['credit_score'] < 740),
        df['credit_score'] >= 740,
    ]
    choices = ['Subprime (<580)', 'Near-Prime (580-669)', 'Prime (670-739)', 'Super-Prime (740+)']
    df['credit_band'] = np.select(conditions, choices, default='Unknown')
    return df


def fit_kaplan_meier(df: pd.DataFrame) -> dict:
    """
    Fit Kaplan-Meier curves for each credit score band.
    
    Returns
    -------
    dict
        Dictionary containing fitters, median times, and survival probabilities
    """
    df = assign_credit_band(df.copy())
    
    bands = ['Subprime (<580)', 'Near-Prime (580-669)', 'Prime (670-739)', 'Super-Prime (740+)']
    kmf_dict = {}
    median_times = {}
    survival_probs = {}
    
    for band in bands:
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df['time_end'],
            band_df['event_default'],
            label=band
        )
        kmf_dict[band] = kmf
        
        # Median survival time (time at which survival = 0.5)
        try:
            median = kmf.median_survival_time_
            median_times[band] = round(median, 2) if not np.isinf(median) else None
        except Exception:
            median_times[band] = None
        
        # 12-month and 24-month survival probabilities
        survival_probs[band] = {
            '12_month': round(kmf.survival_function_at_times(12).values[0], 4),
            '24_month': round(kmf.survival_function_at_times(24).values[0], 4),
        }
    
    return {
        'fitters': kmf_dict,
        'median_times': median_times,
        'survival_probs': survival_probs,
        'bands': bands,
    }


def plot_km_curves(km_results: dict, save_path: str = 'reports/km_survival_curves.png') -> None:
    """Plot and save Kaplan-Meier survival curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Subprime (<580)': '#d62728', 'Near-Prime (580-669)': '#ff7f0e',
              'Prime (670-739)': '#2ca02c', 'Super-Prime (740+)': '#1f77b4'}
    
    for band in km_results['bands']:
        kmf = km_results['fitters'][band]
        ax = kmf.plot_survival_function(ax=ax, color=colors.get(band, None))
    
    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Median survival')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Kaplan-Meier plot to {save_path}")


def print_km_summary(km_results: dict) -> None:
    """Print summary of Kaplan-Meier results."""
    print("\n" + "="*70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS BY CREDIT SCORE BAND")
    print("="*70)
    
    for band in km_results['bands']:
        print(f"\n{band}")
        print("-" * 40)
        probs = km_results['survival_probs'][band]
        print(f"  12-month survival: {probs['12_month']:.1%}")
        print(f"  24-month survival: {probs['24_month']:.1%}")
        
        median = km_results['median_times'][band]
        if median is not None:
            print(f"  Median time to default: {median:.1f} months")
        else:
            print(f"  Median time to default: Not reached (>50% still surviving)")


if __name__ == '__main__':
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    results = fit_kaplan_meier(df)
    print_km_summary(results)
    plot_km_curves(results)