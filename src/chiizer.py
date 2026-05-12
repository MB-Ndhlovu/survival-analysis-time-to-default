"""Risk Chiizer: bin continuous variables into risk categories and compare survival."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def bin_variable(series, bins=5):
    """Bin a continuous variable into quantile-based categories."""
    try:
        bin_edges = pd.qcut(series, q=bins, retbins=True, duplicates='drop')[1]
        actual_bins = len(bin_edges) - 1
    except ValueError:
        actual_bins = bins
    labels = [f'Q{i+1}' for i in range(actual_bins)]
    return pd.qcut(series, q=bins, labels=labels, duplicates='drop')

def chiize(df, variable, n_bins=5, event_col='event_default', duration_col='time_end'):
    """
    Bin a continuous variable and compute survival curves for each bin.
    Returns dict of {bin_label: {'kmf': KM fitter, 'survival_at_12': float, ...}}
    """
    df = df.copy()
    df['bin'] = bin_variable(df[variable], bins=n_bins)

    results = {}
    for bin_label in sorted(df['bin'].unique()):
        band_df = df[df['bin'] == bin_label]
        if len(band_df) < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(band_df[duration_col], band_df[event_col], label=str(bin_label))
        results[bin_label] = {
            'kmf': kmf,
            'n': len(band_df),
            'median': kmf.median_survival_time_,
            's12': kmf.survival_function_at_times(12).values[0],
            's24': kmf.survival_function_at_times(24).values[0],
        }

    return results

def plot_chiizer_results(results, variable_name, save_path=None):
    """Plot survival curves for each bin of a variable."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for (label, res), color in zip(results.items(), colors):
        kmf = res['kmf']
        ax.plot(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0],
                label=f"{label} (n={res['n']})", color=color, linewidth=2)

    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival by {variable_name} Risk Category', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

def chiizer_summary(results, variable_name):
    """Print a summary table for chiizer results."""
    print(f"\n=== Risk Chiizer: {variable_name} ===")
    print(f"{'Bin':<10} {'n':>6} {'S(12)':>8} {'S(24)':>8} {'Median':>10}")
    print("-" * 45)
    for label, res in results.items():
        median_str = f"{res['median']:.1f}m" if not np.isinf(res['median']) else "NR"
        print(f"{label:<10} {res['n']:>6} {res['s12']:>8.1%} {res['s24']:>8.1%} {median_str:>10}")
    return results