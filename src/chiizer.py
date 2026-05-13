"""
Risk Chiizer: Discretize continuous variables into risk categories
and compute survival curves for each bin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def chiize_variable(df, variable, n_bins=4, labels=None):
    """
    Bin a continuous variable into risk categories and compute survival curves.

    Returns binned data and summary statistics per bin.
    """
    if labels is None:
        labels = [f'Q{i+1}' for i in range(n_bins)]

    # Create quantile-based bins
    try:
        df['binned'] = pd.qcut(df[variable], q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # Fallback to equal width bins if quantile fails
        df['binned'] = pd.cut(df[variable], bins=n_bins, labels=labels)

    # Compute survival statistics per bin
    kmf = KaplanMeierFitter()
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Survival curves by bin
    ax1 = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_bins))

    for i, (label, color) in enumerate(zip(labels, colors)):
        bin_df = df[df['binned'] == label]

        kmf.fit(bin_df['time_end'], bin_df['event_default'], label=f'{variable} {label}')
        results[label] = {
            'n': len(bin_df),
            'median_survival': kmf.median_survival_time_,
            'survival_at_12': kmf.predict(12),
            'survival_at_24': kmf.predict(24),
        }

        kmf.plot_survival_function(ax=ax1, color=color)

    ax1.set_xlabel('Months')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title(f'Survival by {variable}')
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart of 24-month survival by bin
    ax2 = axes[1]
    bin_labels = list(results.keys())
    survival_24 = [results[l]['survival_at_24'] for l in bin_labels]

    bars = ax2.bar(bin_labels, survival_24, color=colors)
    ax2.set_xlabel(f'{variable} Quantile')
    ax2.set_ylabel('24-Month Survival Probability')
    ax2.set_title(f'24-Month Survival by {variable} Quintile')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, survival_24):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    safe_name = variable.replace(' ', '_').lower()
    plt.savefig(f'/home/workspace/Projects/survival-analysis-time-to-default/reports/chiizer_{safe_name}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Clean up temp column
    df.drop(columns=['binned'], inplace=True)

    return results


def chiize_all_variables(df):
    """Chiize all major continuous variables."""
    variables = ['income', 'debt_to_income', 'LTV_ratio', 'loan_amount']

    all_results = {}
    for var in variables:
        results = chiize_variable(df.copy(), var)
        all_results[var] = results

    return all_results


def print_chiizer_summary(all_results):
    """Print formatted summary of chiizer results."""
    print("\n" + "="*70)
    print("RISK CHIIZER SUMMARY")
    print("="*70)

    for var, results in all_results.items():
        print(f"\n{var}:")
        print(f"  {'Bin':<8} {'N':>6} {'Median (mo)':>12} {'12-mo Surv':>12} {'24-mo Surv':>12}")
        print("  " + "-" * 52)

        for label, data in results.items():
            print(f"  {label:<8} {data['n']:>6} {data['median_survival']:>12.1f} "
                  f"{data['survival_at_12']*100:>11.1f}% {data['survival_at_24']*100:>11.1f}%")


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    all_results = chiize_all_variables(df)
    print_chiizer_summary(all_results)
    print("\nPlots saved to reports/chiizer_*.png")