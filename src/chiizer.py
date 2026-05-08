"""
Risk Chiizer: Bin continuous variables into risk categories and compute survival curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(df, variable, n_bins=4, method='quantile'):
    """
    Bin a continuous variable into risk categories.

    Parameters:
    - variable: column name to bin
    - n_bins: number of bins
    - method: 'quantile' (equal count) or 'uniform'
    """
    if method == 'quantile':
        try:
            labels = [f'Q{i+1}' for i in range(n_bins)]
            df[f'{variable}_bin'] = pd.qcut(df[variable], q=n_bins, labels=labels, duplicates='drop')
        except ValueError:
            # Fallback to uniform if quantile fails
            df[f'{variable}_bin'] = pd.cut(df[variable], bins=n_bins, labels=labels)
    else:
        labels = [f'Bin {i+1}' for i in range(n_bins)]
        df[f'{variable}_bin'] = pd.cut(df[variable], bins=n_bins, labels=labels)

    return df


def chiizer_analysis(df, variables=None):
    """
    Run chiizer analysis on multiple variables.

    For each variable:
    1. Bin into risk categories
    2. Fit Kaplan-Meier curves per bin
    3. Compute survival probabilities
    """
    if variables is None:
        variables = ['credit_score', 'debt_to_income', 'interest_rate', 'LTV_ratio']

    results = {}
    kmf_global = KaplanMeierFitter()

    for var in variables:
        try:
            df = bin_variable(df, var, n_bins=4, method='quantile')
        except Exception as e:
            print(f"Warning: Could not bin {var}: {e}")
            continue

        bin_col = f'{var}_bin'
        bin_results = {}

        for bin_label in df[bin_col].dropna().unique():
            mask = df[bin_col] == bin_label
            bin_df = df[mask]

            if len(bin_df) < 10:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=bin_df['time_end'],
                event_observed=bin_df['event_default'],
                label=f'{var}_{bin_label}'
            )

            bin_results[str(bin_label)] = {
                'kmf': kmf,
                'n': int(mask.sum()),
                'n_defaults': int(bin_df['event_default'].sum()),
                'median_survival': kmf.median_survival_time_,
            }

            kmf_global.fit(
                durations=bin_df['time_end'],
                event_observed=bin_df['event_default'],
                label=f'{var}_{bin_label}'
            )

        results[var] = bin_results

    return results, kmf_global


def plot_chiizer_results(results, save_path=None):
    """Plot chiizer survival curves for all variables."""
    n_vars = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (var, bin_results) in enumerate(results.items()):
        ax = axes[idx]

        for i, (label, data) in enumerate(bin_results.items()):
            kmf = data['kmf']
            color = colors[i % len(colors)]
            kmf.plot_survival_function(ax=ax, color=color, label=label)

        ax.set_xlabel('Months')
        ax.set_ylabel('Survival Probability')
        ax.set_title(f'Survival by {var.upper()}')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved chiizer plot to {save_path}")

    return fig


def print_chiizer_summary(results):
    """Print chiizer results summary."""
    print("\n" + "="*70)
    print("RISK CHIIZER ANALYSIS")
    print("="*70)

    for var, bin_results in results.items():
        print(f"\n{var.upper()}")
        print("-" * 40)
        for label, data in sorted(bin_results.items()):
            n = data['n']
            n_def = data['n_defaults']
            median = data['median_survival']
            try:
                s12 = data['kmf'].survival_function_at_times(12).values[0]
                s24 = data['kmf'].survival_function_at_times(24).values[0]
                print(f"  {label}: n={n:4d}, defaults={n_def:3.0f}, median={median:6.1f}, 12m={s12:.3f}, 24m={s24:.3f}")
            except Exception:
                print(f"  {label}: n={n:4d}, defaults={n_def:3.0f}, median={median:6.1f}")


if __name__ == '__main__':
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results, _ = chiizer_analysis(df)
    print_chiizer_summary(results)