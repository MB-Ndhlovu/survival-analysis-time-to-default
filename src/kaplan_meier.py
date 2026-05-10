"""
Kaplan-Meier survival analysis by credit score bands.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def get_credit_band(score):
    if score < 580:
        return '< 580 (Deep Subprime)'
    elif score < 670:
        return '580-669 (Subprime)'
    elif score < 740:
        return '670-739 (Near Prime)'
    else:
        return '740+ (Prime)'


def fit_km_by_credit_band(df):
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(get_credit_band)

    bands = ['< 580 (Deep Subprime)', '580-669 (Subprime)', '670-739 (Near Prime)', '740+ (Prime)']
    results = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    for band in bands:
        subset = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(subset['time_end'], event_observed=subset['event_default'], label=band)

        # Median survival time
        median_survival = kmf.median_survival_time_
        results[band] = {
            'median_survival_months': median_survival,
            'survival_curve': kmf,
            'n_obs': len(subset),
            'n_events': subset['event_default'].sum(),
        }

        kmf.plot_survival_function(ax=ax, ci_show=True)

    plt.title('Kaplan-Meier Survival Curves by Credit Score Band')
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/km_survival_curves.png', dpi=150)
    plt.close()

    # 12-month and 24-month survival probabilities
    for band in bands:
        kmf = results[band]['survival_curve']
        results[band]['survival_12m'] = kmf.predict(12)
        results[band]['survival_24m'] = kmf.predict(24)

    return results


if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = fit_km_by_credit_band(df)
    for band, r in results.items():
        print(f"{band}: median={r['median_survival_months']:.1f}m, 12m={r['survival_12m']:.3f}, 24m={r['survival_24m']:.3f}")