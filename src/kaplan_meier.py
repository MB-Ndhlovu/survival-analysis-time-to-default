"""Kaplan-Meier survival analysis by credit score bands."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def fit_km_by_credit_band(
    df: pd.DataFrame,
    output_dir: str = 'reports'
) -> dict:
    """Fit Kaplan-Meier curves for each credit score band.

    Credit bands:
    - Deep Subprime: < 580
    - Subprime: 580-669
    - Near Prime: 670-739
    - Prime: 740+

    Returns dict with median survival times and survival probabilities.
    """
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(lambda s: (
        'Deep Subprime' if s < 580 else
        'Subprime' if s < 670 else
        'Near Prime' if s < 740 else
        'Prime'
    ))

    bands = ['Deep Subprime', 'Subprime', 'Near Prime', 'Prime']
    band_colors = {'Deep Subprime': '#d62728', 'Subprime': '#ff7f0e', 'Near Prime': '#2ca02c', 'Prime': '#1f77b4'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results = {}
    median_survival = {}

    for band in bands:
        mask = df['credit_band'] == band
        band_df = df[mask]

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=band_df['time_end'],
            event_observed=band_df['event_default'],
            label=band
        )

        # Median survival time
        median = kmf.median_survival_time_
        median_survival[band] = median if not np.isinf(median) else "Not reached"

        # Survival probabilities at 12 and 24 months
        surv_12 = kmf.survival_function_at_times(12).values[0]
        surv_24 = kmf.survival_function_at_times(24).values[0]

        results[band] = {
            'n_obs': int(mask.sum()),
            'n_defaults': int(band_df['event_default'].sum()),
            'n_censored': int((band_df['event_default'] == 0).sum()),
            'median_survival_months': median if not np.isinf(median) else None,
            'survival_12_months': round(float(surv_12), 4),
            'survival_24_months': round(float(surv_24), 4),
        }

        # Plot KM curves
        kmf.plot_survival_function(ax=axes[0], ci_show=True, color=band_colors[band])

        # Plot cumulative density (hazard)
        kmf.plot_cumulative_density(ax=axes[1], ci_show=True, color=band_colors[band])

    axes[0].set_title('Survival Functions by Credit Band', fontsize=12)
    axes[0].set_xlabel('Months')
    axes[0].set_ylabel('Survival Probability')
    axes[0].legend(loc='lower left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_title('Cumulative Default Probability by Credit Band', fontsize=12)
    axes[1].set_xlabel('Months')
    axes[1].set_ylabel('Cumulative Default Probability')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/kaplan_meier_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print results
    print("\n=== Kaplan-Meier Results by Credit Band ===")
    print(f"{'Band':<20} {'N':>6} {'Defaults':>8} {'Censored':>9} {'Median Surv.':>13} {'12mo Surv':>10} {'24mo Surv':>10}")
    print("-" * 90)
    for band in bands:
        r = results[band]
        med = str(r['median_survival_months']) if r['median_survival_months'] else 'Not reached'
        print(f"{band:<20} {r['n_obs']:>6} {r['n_defaults']:>8} {r['n_censored']:>9} {med:>13} {r['survival_12_months']:>10.2%} {r['survival_24_months']:>10.2%}")

    return results


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = fit_km_by_credit_band(df)
    print("\nResults:", json.dumps(results, indent=2))