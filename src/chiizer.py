"""Risk chiizer — bin continuous variables into risk categories and compare survival."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def chiize(df, variable, n_bins=4, labels=None):
    """
    Bin a continuous variable into risk categories and compute survival curves.

    Parameters
    ----------
    df : pd.DataFrame
        Data from data_loader.py
    variable : str
        Column name to bin
    n_bins : int
        Number of bins
    labels : list or None
        Custom bin labels

    Returns
    -------
    dict
        Bin boundaries, survival statistics, and Kaplan-Meier fitters
    """
    if labels is None:
        labels = [f'Q{i+1}' for i in range(n_bins)]

    # Create quantile-based bins
    df_temp = df.copy()
    df_temp['bin'] = pd.qcut(df_temp[variable], q=n_bins, labels=labels, duplicates='drop')

    results = {
        'variable': variable,
        'bins': {},
        'overall_default_rate': float(df['event_default'].mean()),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    kmf_global = KaplanMeierFitter()
    kmf_global.fit(df['time_end'], df['event_default'], label='All Borrowers')

    for label in labels:
        mask = df_temp['bin'] == label
        if mask.sum() == 0:
            continue
        df_bin = df_temp[mask]
        kmf = KaplanMeierFitter()
        kmf.fit(df_bin['time_end'], df_bin['event_default'], label=label)

        # Survival at key horizons
        surv_12 = kmf.survival_function_.loc[12.0, label] if 12.0 in kmf.survival_function_.index else None
        surv_24 = kmf.survival_function_.loc[24.0, label] if 24.0 in kmf.survival_function_.index else None
        median = kmf.median_survival_time_
        if pd.isna(median):
            median = "> 24mo"

        results['bins'][label] = {
            'n': int(mask.sum()),
            'default_rate': float(df_bin['event_default'].mean()),
            'survival_12': float(surv_12) if surv_12 is not None else None,
            'survival_24': float(surv_24) if surv_24 is not None else None,
            'median_survival': median,
            'kmf': kmf,
        }

        kmf.plot_survival_function(ax=ax, ci_show=True)

    kmf_global.plot_survival_function(ax=ax, ci_show=False, linestyle='--', linewidth=2)

    ax.set_xlabel('Months since loan origination')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Survival Curves by {variable} Risk Categories')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = variable.replace(' ', '_').replace('/', '_')
    plt.savefig(f'/home/workspace/Projects/survival-analysis-time-to-default/reports/chiizer_{safe_name}.png', dpi=150)
    plt.close()

    return results

def run_full_chiizer(df):
    """Run chiizer on key credit risk variables."""
    variables = ['credit_score', 'debt_to_income', 'LTV_ratio', 'interest_rate']
    all_results = {}

    for var in variables:
        results = chiize(df, var)
        all_results[var] = results

    return all_results

if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = run_full_chiizer(df)
    for var, res in results.items():
        print(f"\n=== {var} ===")
        for bin_label, bin_data in res['bins'].items():
            print(f"  {bin_label}: n={bin_data['n']}, default_rate={bin_data['default_rate']:.3f}, "
                  f"12-mo survival={bin_data['survival_12']:.4f if bin_data['survival_12'] else 'N/A'}")