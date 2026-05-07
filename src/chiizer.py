"""Risk Chiizer - bin continuous variables into risk categories and compute survival curves."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from src.data_loader import generate_loan_data


def chiize_variable(df, variable, n_bins=4, time_col='time_end', event_col='event_default'):
    """Bin continuous variable into risk categories and compute survival by bin.

    Returns survival statistics for each bin category.
    """
    df = df.copy()

    # Create bins using quantiles
    df[f'{variable}_bin'] = pd.qcut(df[variable], q=n_bins, labels=False, duplicates='drop')

    # Map bin number to descriptive label
    unique_bins = sorted(df[f'{variable}_bin'].unique())
    bin_labels = []
    for b in unique_bins:
        low = df[df[f'{variable}_bin'] == b][variable].quantile(0.1)
        high = df[df[f'{variable}_bin'] == b][variable].quantile(0.9)
        bin_labels.append(f'{low:.1f}-{high:.1f}')

    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (bin_val, label) in enumerate(zip(unique_bins, bin_labels)):
        bin_df = df[df[f'{variable}_bin'] == bin_val]
        kmf = KaplanMeierFitter()
        kmf.fit(bin_df[time_col], bin_df[event_col], label=f'{variable}: {label}')

        median = kmf.median_survival_time_
        results[label] = {
            'median_survival': median if not np.isinf(median) else None,
            'n_obs': len(bin_df),
            'n_events': bin_df[event_col].sum(),
            'survival_curve': kmf,
        }

        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel('Months')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Survival Curves by {variable} Category')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = f'/home/workspace/Projects/survival-analysis-time-to-default/reports/chiizer_{variable}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return results


def run_risk_chiizer(df, time_col='time_end', event_col='event_default'):
    """Run chiizer on all key risk variables."""
    variables = ['debt_to_income', 'LTV_ratio', 'interest_rate', 'employment_years']

    all_results = {}
    for var in variables:
        all_results[var] = chiize_variable(df, var, n_bins=4, time_col=time_col, event_col=event_col)

    return all_results


def print_chiizer_summary(all_results):
    """Print summary of chiizer results."""
    print("\nRisk Chiizer Summary")
    print("=" * 80)

    for var, bins in all_results.items():
        print(f"\n{var}:")
        print("-" * 50)
        for label, data in bins.items():
            median = data['median_survival']
            n = data['n_obs']
            events = data['n_events']
            median_str = f"{median:.1f} mo" if median else "Not reached"
            print(f"  {label}: Median={median_str}, N={n}, Events={events}")

    print("=" * 80)


if __name__ == '__main__':
    df = generate_loan_data()
    all_results = run_risk_chiizer(df)
    print_chiizer_summary(all_results)