"""Risk Chiizer: Bin continuous variables into risk categories and compare survival curves."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def bin_variable(values, n_bins=4, labels=None):
    """Bin continuous variable into quantiles."""
    if labels is None:
        labels = [f'Q{i+1}' for i in range(n_bins)]
    try:
        bins = pd.qcut(values, q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        bins = pd.cut(values, bins=n_bins, labels=labels[:n_bins])
    return bins

def chiize_variable(df, variable, n_bins=4, time_col='time_end', event_col='event_default'):
    """Build risk chiizer for a continuous variable.

    Args:
        df: DataFrame with survival data
        variable: Column name to bin
        n_bins: Number of bins
        time_col: Column for time
        event_col: Column for event indicator

    Returns:
        dict with bin labels, kmf objects, and survival probabilities
    """
    df = df.copy()
    df['bin'] = bin_variable(df[variable], n_bins)

    results = {'variable': variable, 'bins': {}}

    for bin_label in df['bin'].unique():
        bin_df = df[df['bin'] == bin_label]
        kmf = KaplanMeierFitter()
        kmf.fit(bin_df[time_col], bin_df[event_col], label=f'{variable}: {bin_label}')

        median = kmf.median_survival_time_
        
        sf = kmf.survival_function_
        survival_12 = float(sf.loc[sf.index <= 12].iloc[-1].values[0]) if len(sf) > 0 and sf.index[-1] >= 12 else None
        survival_24 = float(sf.loc[sf.index <= 24].iloc[-1].values[0]) if len(sf) > 0 and sf.index[-1] >= 24 else None

        results['bins'][bin_label] = {
            'kmf': kmf,
            'median_survival': median,
            'survival_12m': survival_12,
            'survival_24m': survival_24,
            'n': len(bin_df)
        }

    return results

def plot_chiizer(results, save_path='reports/risk_chiizer.png'):
    """Plot survival curves for all bins of a chiized variable."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results['bins'])))

    for (bin_label, data), color in zip(results['bins'].items(), colors):
        kmf = data['kmf']
        ax = kmf.plot_survival_function(ax=ax, color=color, linewidth=2, label=bin_label)

    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Curves by {results["variable"].replace("_", " ").title()} Quartile', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path

def chiizer_summary(chiizer_results):
    """Summarize chiizer results as a table."""
    rows = []
    for bin_label, data in chiizer_results['bins'].items():
        rows.append({
            'bin': bin_label,
            'n': data['n'],
            'median_survival': f"{data['median_survival']:.1f}" if data['median_survival'] else 'Not reached',
            '12m_survival': f"{data['survival_12m']:.1%}" if data['survival_12m'] else 'N/A',
            '24m_survival': f"{data['survival_24m']:.1%}" if data['survival_24m'] else 'N/A'
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()

    for var in ['income', 'loan_amount', 'interest_rate']:
        result = chiize_variable(df, var)
        summary = chiizer_summary(result)
        print(f"\n{var.upper()} Chiizer:")
        print(summary.to_string(index=False))