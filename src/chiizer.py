"""
Risk Chiizer: bin continuous variables into risk categories and compute survival curves.
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def chiize(df, variable, n_bins=4):
    """
    Bin a continuous variable into quantile-based risk categories
    and compute survival statistics for each bin.
    """
    df = df.copy()

    # Create quantile bins
    try:
        df['bin'] = pd.qcut(df[variable], q=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')
    except ValueError:
        # Fallback to equal width bins
        df['bin'] = pd.cut(df[variable], bins=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])

    results = {}

    for bin_label in df['bin'].unique():
        if pd.isna(bin_label):
            continue
        subset = df[df['bin'] == bin_label]
        kmf = KaplanMeierFitter()
        kmf.fit(subset['time_end'], event_observed=subset['event_default'])

        results[str(bin_label)] = {
            'n': len(subset),
            'defaults': subset['event_default'].sum(),
            'survival_12m': kmf.predict(12),
            'survival_24m': kmf.predict(24),
            'median_survival': kmf.median_survival_time_,
        }

    return results


def run_chiizer(df):
    """
    Chiize multiple key variables and return risk segmentation.
    """
    variables = ['credit_score', 'debt_to_income', 'LTV_ratio', 'interest_rate']
    all_results = {}

    for var in variables:
        all_results[var] = chiize(df, var)

    # Print summary
    print("\n=== Risk Chiizer Results ===")
    for var, res in all_results.items():
        print(f"\n--- {var} ---")
        for bin_label, stats in res.items():
            print(f"  {bin_label}: n={stats['n']}, defaults={stats['defaults']}, "
                  f"12m survival={stats['survival_12m']:.3f}, 24m survival={stats['survival_24m']:.3f}")

    return all_results


if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = run_chiizer(df)