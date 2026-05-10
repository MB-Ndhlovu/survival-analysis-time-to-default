"""
Risk Chiizer: bin continuous variables into risk categories,
compute and compare survival curves for each bin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(series, n_bins=4, strategy='quantile'):
    """Bin a continuous variable into categories."""
    if strategy == 'quantile':
        return pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
    else:
        return pd.cut(series, bins=n_bins, labels=False)


def chiize(df, variable, n_bins=4, duration_col='time_end', event_col='event_default'):
    """
    Build a risk chiizer for a continuous variable.
    
    Returns dict with bin boundaries, survival stats per bin.
    """
    df = df.copy()
    
    # Create bins
    if variable in ['credit_score']:
        # Custom bins for credit score
        if variable == 'credit_score':
            bins = [0, 580, 670, 740, 850]
            labels = ['Very Poor', 'Poor', 'Fair', 'Good']
            df['bin'] = pd.cut(df[variable], bins=bins, labels=labels, right=False)
        else:
            df['bin'] = bin_variable(df[variable], n_bins)
    else:
        df['bin'] = bin_variable(df[variable], n_bins)

    # Compute survival stats per bin
    kmf = KaplanMeierFitter()
    
    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))

    for bin_label in df['bin'].dropna().unique():
        subset = df[df['bin'] == bin_label]
        
        kmf.fit(
            durations=subset[duration_col],
            event_observed=subset[event_col],
            label=str(bin_label)
        )
        
        results[str(bin_label)] = {
            'n': len(subset),
            'events': subset[event_col].sum(),
            '12m_survival': kmf.predict(12),
            '24m_survival': kmf.predict(24),
            'median_survival': kmf.median_survival_time_
        }
        
        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel('Months')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Survival by {variable}')
    ax.legend(title=variable)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    return results, fig


def chiize_all_variables(df, variables, plot_dir='reports/chiizer'):
    """
    Run chiizer on multiple variables.
    Returns summary dict.
    """
    import os
    os.makedirs(plot_dir, exist_ok=True)

    all_results = {}

    for var in variables:
        results, fig = chiize(df, var)
        fig.savefig(f"{plot_dir}/{var}_survival.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        all_results[var] = results

    return all_results


def print_chiizer_summary(all_results):
    print("\n" + "="*70)
    print("RISK CHIIZER SUMMARY")
    print("="*70)

    for var, results in all_results.items():
        print(f"\n{var}:")
        print(f"  {'Category':<15} {'N':>6} {'Events':>8} {'12m Surv':>10} {'24m Surv':>10}")
        print("  " + "-"*55)
        
        for cat, stats in results.items():
            print(f"  {cat:<15} {stats['n']:>6} {int(stats['events']):>8} {stats['12m_survival']:>10.3f} {stats['24m_survival']:>10.3f}")

    print("="*70)


if __name__ == '__main__':
    from data_loader import generate_loan_data

    df = generate_loan_data()
    variables = ['credit_score', 'debt_to_income', 'LTV_ratio']
    all_results = chiize_all_variables(df, variables)
    print_chiizer_summary(all_results)