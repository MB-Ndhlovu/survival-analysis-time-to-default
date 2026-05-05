import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def fit_kaplan_meier(df, duration_col='time_end', event_col='event_default'):
    """Fit Kaplan-Meier curves for credit score bands."""
    df = df.copy()
    df['credit_band'] = df['credit_score'].apply(
        lambda s: 'Deep Subprime (<580)' if s < 580 else
                  'Subprime (580-669)' if s < 670 else
                  'Near Prime (670-739)' if s < 740 else 'Prime (740+)'
    )

    kmf = KaplanMeierFitter()
    bands = ['Deep Subprime (<580)', 'Subprime (580-669)', 'Near Prime (670-739)', 'Prime (740+)']

    results = {}
    for band in bands:
        mask = df['credit_band'] == band
        if mask.sum() > 0:
            kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col], label=band)
            median_survival = kmf.median_survival_time_
            results[band] = {
                'n': mask.sum(),
                'median_survival': median_survival if not np.isinf(median_survival) else None,
                'survival_at_12': kmf.predict(12),
                'survival_at_24': kmf.predict(24)
            }

    return results, kmf, df

def plot_km_curves(kmf, df, output_path='reports/km_survival_curves.png'):
    """Plot Kaplan-Meier survival curves by credit score band."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bands = ['Deep Subprime (<580)', 'Subprime (580-669)', 'Near Prime (670-739)', 'Prime (740+)']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    for band, color in zip(bands, colors):
        mask = df['credit_band'] == band
        if mask.sum() > 0:
            kmf.fit(df.loc[mask, 'time_end'], df.loc[mask, 'event_default'], label=band)
            kmf.plot_survival_function(ax=ax, color=color)

    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Kaplan-Meier plot to {output_path}")

    return output_path

if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results, kmf, df_labeled = fit_kaplan_meier(df)
    for band, res in results.items():
        print(f"{band}: n={res['n']}, median={res['median_survival']}, 12m={res['survival_at_12']:.3f}, 24m={res['survival_at_24']:.3f}")