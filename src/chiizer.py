import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

def bin_variable(df, col, bins, labels):
    """Bin a continuous variable into categories."""
    return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

def build_risk_chiizer(df, duration_col='time_end', event_col='event_default'):
    """Build risk chiizer with binned variables and survival curves."""
    df = df.copy()

    # Bin key variables
    df['income_band'] = bin_variable(df, 'income', [0, 30000, 50000, 75000, 100000, np.inf],
                                     ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df['ltv_band'] = bin_variable(df, 'ltv_ratio', [0, 0.6, 0.75, 0.85, 0.95, np.inf],
                                   ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df['dti_band'] = bin_variable(df, 'debt_to_income', [0, 0.2, 0.3, 0.4, 0.5, np.inf],
                                   ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # For each bin, compute survival stats
    kmf = KaplanMeierFitter()

    chiizer_results = {}
    for band_col in ['income_band', 'ltv_band', 'dti_band']:
        chiizer_results[band_col] = {}
        for band in df[band_col].dropna().unique():
            mask = df[band_col] == band
            kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col])
            chiizer_results[band_col][band] = {
                'n': mask.sum(),
                'survival_12m': kmf.predict(12),
                'survival_24m': kmf.predict(24),
                'median_survival': kmf.median_survival_time_ if not np.isinf(kmf.median_survival_time_) else None
            }

    return chiizer_results

def print_chiizer_results(chiizer_results):
    """Print risk chiizer results."""
    print("\n=== Risk Chiizer Results ===")
    for var, bands in chiizer_results.items():
        var_name = var.replace('_band', '').replace('_', ' ').title()
        print(f"\n{var_name}:")
        print("-" * 60)
        for band, stats in bands.items():
            median_str = f"{stats['median_survival']:.1f}m" if stats['median_survival'] else "N/A"
            print(f"  {str(band):12s}: n={stats['n']:5d}, 12m survival={stats['survival_12m']:.3f}, 24m={stats['survival_24m']:.3f}, median={median_str}")

if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = build_risk_chiizer(df)
    print_chiizer_results(results)