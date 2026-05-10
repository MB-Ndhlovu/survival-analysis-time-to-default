"""
Kaplan-Meier survival curves by credit score band.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def fit_kaplan_meier(df, plot_path=None):
    """
    Fit KM curves for credit score bands and optionally plot.
    """
    # Define credit score bands
    bands = ['<580', '580-669', '670-739', '740+']
    band_edges = [0, 580, 670, 740, 850]
    
    df = df.copy()
    df['credit_band'] = pd.cut(df['credit_score'], bins=band_edges, labels=bands, right=False)

    results = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'<580': '#d62728', '580-669': '#ff7f0e', '670-739': '#2ca02c', '740+': '#1f77b4'}

    for band in bands:
        subset = df[df['credit_band'] == band]
        
        if len(subset) < 10:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=subset['time_end'],
            event_observed=subset['event_default'],
            label=band
        )

        # Store results
        results[band] = {
            'n': len(subset),
            'median_survival': kmf.median_survival_time_,
            'survival_curve': kmf.survival_function_.values.flatten(),
            'timeline': kmf.timeline,
            '12m_survival': kmf.predict(12),
            '24m_survival': kmf.predict(24),
            'events': subset['event_default'].sum()
        }

        kmf.plot_survival_function(ax=ax, color=colors[band])

    ax.set_xlabel('Months')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax.legend(title='Credit Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved KM plot to {plot_path}")

    plt.close()

    return results


def print_km_summary(results):
    print("\n" + "="*60)
    print("KAPLAN-MEIER SURVIVAL SUMMARY BY CREDIT BAND")
    print("="*60)
    print(f"{'Band':<12} {'N':>6} {'Events':>8} {'Median(Mo)':>12} {'12m Surv':>10} {'24m Surv':>10}")
    print("-"*60)
    
    for band, r in results.items():
        median = r['median_survival']
        median_str = f"{median:.1f}" if median != np.inf else "N/A"
        print(f"{band:<12} {r['n']:>6} {int(r['events']):>8} {median_str:>12} {r['12m_survival']:>10.3f} {r['24m_survival']:>10.3f}")
    
    print("="*60)


if __name__ == '__main__':
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    results = fit_kaplan_meier(df, 'reports/km_curves.png')
    print_km_summary(results)