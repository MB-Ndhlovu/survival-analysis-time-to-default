"""Kaplan-Meier survival curves by credit score band."""

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from src.data_loader import generate_loan_data, get_credit_score_band


def fit_kaplan_meier_by_band(df, time_col='time_end', event_col='event_default'):
    """Fit KM curves for each credit score band."""
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(get_credit_score_band)

    bands = ['Very Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All bands on same plot
    ax1 = axes[0]
    colors = {'Very Poor (<580)': 'red', 'Fair (580-669)': 'orange',
              'Good (670-739)': 'blue', 'Excellent (740+)': 'green'}

    for band in bands:
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(band_df[time_col], band_df[event_col], label=band)

        median_survival = kmf.median_survival_time_
        results[band] = {
            'median_survival_months': median_survival if not np.isinf(median_survival) else None,
            'survival_curve': kmf,
            'n_obs': len(band_df),
            'n_events': band_df[event_col].sum(),
        }

        kmf.plot_survival_function(ax=ax1, color=colors[band])

    ax1.set_xlabel('Months')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Plot 2: Cumulative hazard
    ax2 = axes[1]
    for band in bands:
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(band_df[time_col], band_df[event_col], label=band)
        kmf.plot_cumulative_density(ax=ax2, color=colors[band])

    ax2.set_xlabel('Months')
    ax2.set_ylabel('Cumulative Default Probability')
    ax2.set_title('Cumulative Default by Credit Score Band')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/kaplan_meier_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    return results


def compute_survival_probabilities(results, months=[12, 24]):
    """Compute survival probabilities at specific time points."""
    output = {}
    for band, data in results.items():
        kmf = data['survival_curve']
        probs = {}
        for m in months:
            # Get survival probability at month m
            timeline = kmf.survival_function_.index
            if m in timeline:
                probs[f'{m}_month'] = kmf.survival_function_.loc[m].values[0]
            else:
                # Interpolate or find closest
                closest = timeline[timeline <= m].max()
                probs[f'{m}_month'] = kmf.survival_function_.loc[closest].values[0]
        output[band] = {**data, **probs}
    return output


if __name__ == '__main__':
    df = generate_loan_data()
    results = fit_kaplan_meier_by_band(df)
    probs = compute_survival_probabilities(results)

    print("Kaplan-Meier Results by Credit Band:")
    print("-" * 60)
    for band, data in probs.items():
        median = data['median_survival_months']
        print(f"\n{band}")
        print(f"  N={data['n_obs']}, Events={data['n_events']}")
        print(f"  Median survival: {median if median else 'Not reached'}")
        print(f"  12-month survival: {data['12_month']:.1%}")
        print(f"  24-month survival: {data['24_month']:.1%}")