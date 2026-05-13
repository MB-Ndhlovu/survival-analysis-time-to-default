"""
Kaplan-Meier survival curves for credit score bands.
Non-parametric estimation of survival functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def fit_kaplan_meier(df, duration_col='time_end', event_col='event_default'):
    """
    Fit Kaplan-Meier curves for each credit score band.

    Returns dict with fitted fitters and summary statistics.
    """
    bands = ['Very Poor', 'Fair', 'Good', 'Excellent']
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All bands on same plot
    ax1 = axes[0]
    colors = {'Very Poor': '#d62728', 'Fair': '#ff7f0e', 'Good': '#2ca02c', 'Excellent': '#1f77b4'}

    for band in bands:
        band_df = df[df['credit_band'] == band]

        kmf = KaplanMeierFitter()
        kmf.fit(band_df[duration_col], band_df[event_col], label=band)

        results[band] = {
            'fitter': kmf,
            'median_survival': kmf.median_survival_time_,
            'survival_at_12': kmf.predict(12),
            'survival_at_24': kmf.predict(24),
        }

        kmf.plot_survival_function(ax=ax1, color=colors[band])

    ax1.set_xlabel('Months')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative hazard
    ax2 = axes[1]
    for band in bands:
        kmf = results[band]['fitter']
        cumulative_hazard = 1 - kmf.survival_function_
        ax2.plot(kmf.survival_function_.index, cumulative_hazard,
                 label=band, color=colors[band])

    ax2.set_xlabel('Months')
    ax2.set_ylabel('Cumulative Default Probability')
    ax2.set_title('Cumulative Default by Credit Score Band')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/kaplan_meier_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    return results


def print_summary(results):
    """Print summary statistics for each band."""
    print("\n" + "="*70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS SUMMARY")
    print("="*70)

    for band in ['Very Poor', 'Fair', 'Good', 'Excellent']:
        r = results[band]
        kmf = r['fitter']

        print(f"\n{band} Credit Score Band:")
        print(f"  Median Survival Time: {r['median_survival']:.1f} months")
        print(f"  12-Month Survival:    {r['survival_at_12']*100:.1f}%")
        print(f"  24-Month Survival:    {r['survival_at_24']*100:.1f}%")

        # Confidence intervals
        ci = kmf.confidence_interval_
        print(f"  12-month 95% CI:      [{ci.iloc[11, 0]*100:.1f}%, {ci.iloc[11, 1]*100:.1f}%]")

    print("\n" + "="*70)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_kaplan_meier(df)
    print_summary(results)
    print("\nPlot saved to reports/kaplan_meier_curves.png")