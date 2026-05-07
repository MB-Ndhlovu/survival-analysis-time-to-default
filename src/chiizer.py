"""
Risk Chiizer: Bin continuous variables into risk categories.
Compute survival curves for each bin to identify risk drivers.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


def create_risk_bins(df, variable, n_bins=4, strategy='quantile'):
    """
    Bin a continuous variable into risk categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    variable : str
        Column name to bin
    n_bins : int
        Number of bins to create
    strategy : str
        'quantile' for equal-sized bins, 'uniform' for equal-width

    Returns
    -------
    pd.Series
        Bin labels
    """
    if strategy == 'quantile':
        bins = pd.qcut(df[variable], q=n_bins, duplicates='drop')
    else:
        bins = pd.cut(df[variable], bins=n_bins, duplicates='drop')

    return bins


def chiize_variable(df, variable, n_bins=4, time_col='time_end', event_col='event_default'):
    """
    Build a risk chiizer for a continuous variable.

    Creates bins and computes survival statistics for each bin.

    Returns
    -------
    dict with keys:
        - bins_df: DataFrame with bin labels and boundaries
        - kmfitters: dict of fitted KM fitters per bin
        - survival_summary: DataFrame with survival stats per bin
    """
    bins = create_risk_bins(df, variable, n_bins=n_bins)

    df_copy = df.copy()
    df_copy[f'{variable}_bin'] = bins

    # Fit KM for each bin
    bin_labels = sorted(df_copy[f'{variable}_bin'].unique())
    kmfitters = {}
    survival_stats = []

    for label in bin_labels:
        mask = df_copy[f'{variable}_bin'] == label
        subset = df_copy[mask]

        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], subset[event_col], label=str(label))

        median_surv = kmf.median_survival_time_
        surv_12 = kmf.survival_function_at_times(12).values[0]
        surv_24 = kmf.survival_function_at_times(24).values[0]
        n_events = subset[event_col].sum()

        kmfitters[str(label)] = kmf

        survival_stats.append({
            'bin': str(label),
            'n_loans': len(subset),
            'n_defaults': n_events,
            'default_rate': round(n_events / len(subset), 4),
            'median_survival': round(median_surv, 2) if np.isfinite(median_surv) else None,
            'survival_12m': round(surv_12, 4),
            'survival_24m': round(surv_24, 4),
        })

    survival_df = pd.DataFrame(survival_stats)

    return {
        'bins': bins,
        'kmfitters': kmfitters,
        'survival_summary': survival_df,
    }


def plot_chiizer_results(chiizer_result, variable, output_path=None):
    """
    Plot survival curves for each bin of a chiized variable.

    Parameters
    ----------
    chiizer_result : dict
        Output from chiize_variable
    variable : str
        Variable name for title
    output_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(chiizer_result['kmfitters'])))

    for (label, kmf), color in zip(chiizer_result['kmfitters'].items(), colors):
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2)

    ax.set_xlabel('Months Since Loan Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Curves by {variable.replace("_", " ").title()} Bins\n(Risk Chiizer)', fontsize=13)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9, title='Bin')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def run_full_chiizer(df, variables=None, time_col='time_end', event_col='event_default'):
    """
    Run chiizer on multiple variables and return all results.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    variables : list, optional
        Variables to chiize. Defaults to key credit risk variables
    time_col : str
        Time column name
    event_col : str
        Event column name

    Returns
    -------
    dict
        Dictionary mapping variable name to chiizer result
    """
    if variables is None:
        variables = ['credit_score', 'debt_to_income', 'LTV_ratio', 'interest_rate']

    results = {}

    for var in variables:
        results[var] = chiize_variable(df, var, n_bins=4, time_col=time_col, event_col=event_col)

    return results


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_band

    df = generate_loan_data(5000)
    df = add_credit_band(df)

    print("Running Risk Chiizer on credit_score...")
    result = chiize_variable(df, 'credit_score', n_bins=4)
    print("\nSurvival by Credit Score Bins:")
    print(result['survival_summary'].to_string(index=False))