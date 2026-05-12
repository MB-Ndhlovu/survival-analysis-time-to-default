"""Kaplan-Meier survival curves by credit score band."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def fit_kaplan_meier(df):
    """
    Fit Kaplan-Meier curves for credit score bands and compute median survival times.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from data_loader.py with time_start, time_end, event_default

    Returns
    -------
    dict
        Results containing survival curves, median times, and 12/24-month survival probs
    """
    # Define credit score bands
    bands = {
        'Very Poor (<580)': (df['credit_score'] < 580),
        'Fair (580-669)': (df['credit_score'] >= 580) & (df['credit_score'] < 670),
        'Good (670-739)': (df['credit_score'] >= 670) & (df['credit_score'] < 740),
        'Excellent (740+)': (df['credit_score'] >= 740),
    }

    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mask in bands.items():
        df_band = df[mask].copy()
        kmf = KaplanMeierFitter()
        kmf.fit(df_band['time_end'], df_band['event_default'], label=label)

        # Median survival time (time when S(t) <= 0.5)
        median_survival = kmf.median_survival_time_
        if pd.isna(median_survival):
            median_survival = "> 24 months (not reached)"

        # 12-month and 24-month survival probabilities
        surv_at_12 = kmf.survival_function_.reindex([12.0]).values[0][0] if 12.0 <= kmf.survival_function_.index.max() else None
        surv_at_24 = kmf.survival_function_.reindex([24.0]).values[0][0] if 24.0 <= kmf.survival_function_.index.max() else None

        results[label] = {
            'kmf': kmf,
            'median_survival_time': median_survival,
            'survival_12_month': float(surv_at_12) if surv_at_12 is not None else None,
            'survival_24_month': float(surv_at_24) if surv_at_24 is not None else None,
            'n_obs': int(mask.sum()),
            'n_defaults': int(df_band['event_default'].sum()),
        }

        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel('Months since loan origination')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/kaplan_meier_curves.png', dpi=150)
    plt.close()

    return results

if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = fit_kaplan_meier(df)
    for band, res in results.items():
        print(f"\n{band}:")
        print(f"  N={res['n_obs']}, Defaults={res['n_defaults']}")
        print(f"  Median survival: {res['median_survival_time']}")
        print(f"  12-mo survival: {res['survival_12_month']:.4f}")
        print(f"  24-mo survival: {res['survival_24_month']:.4f}")