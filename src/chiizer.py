"""
Risk Chiizer — bin continuous variables into risk categories and compute survival curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def chiize_variable(df: pd.DataFrame, var: str, n_bins: int = 4,
                    time_col: str = 'time_end', event_col: str = 'event_default',
                    ascending: bool = True) -> dict:
    """
    Bin a continuous variable into risk categories and compute survival curves.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data
    var : str
        Variable name to bin
    n_bins : int
        Number of bins (default 4 = quartiles)
    time_col : str
        Time column
    event_col : str
        Event column
    ascending : bool
        If True, bins are ordered low-to-high risk
        (e.g., low income = higher risk for income)

    Returns
    -------
    dict
        Dictionary with bin labels, KM fitters, and summary statistics
    """
    # Create bins using quantiles
    try:
        df[f'{var}_bin'], bin_edges = pd.qcut(df[var], q=n_bins, labels=False,
                                               retbins=True, duplicates='drop')
    except ValueError:
        df[f'{var}_bin'], bin_edges = pd.cut(df[var], bins=n_bins, labels=False,
                                              retbins=True, duplicates='drop')

    n_actual_bins = int(df[f'{var}_bin'].max()) + 1
    labels = [f'Q{i+1}' for i in range(n_actual_bins)]

    results = {}
    fitters = {}

    for bin_idx in range(n_actual_bins):
        label = labels[bin_idx]
        mask = df[f'{var}_bin'] == bin_idx
        subset = df[mask].copy()

        if len(subset) < 10:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], subset[event_col], label=f'{var}: {label}')

        fitters[f'{var}: {label}'] = kmf
        results[f'{var}: {label}'] = {
            'n': len(subset),
            'events': int(subset[event_col].sum()),
            'mean_value': round(float(subset[var].mean()), 2)
        }

    return fitters, results


def plot_chiizer_results(fitters_dict: dict, var: str, save_path: str = None):
    """
    Plot survival curves from chiizer for a single variable.

    Parameters
    ----------
    fitters_dict : dict
        Dictionary of KaplanMeierFitter objects
    var : str
        Variable name for title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(fitters_dict)))

    for i, (label, kmf) in enumerate(fitters_dict.items()):
        kmf.plot_survival_function(ax=ax, color=colors[i])

    ax.set_title(f'Survival Curves by {var} Quartile', fontsize=14)
    ax.set_xlabel('Months Since Loan Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chiizer plot saved to {save_path}")

    plt.close()


def run_full_chiizer(df: pd.DataFrame, variables: list = None,
                      time_col: str = 'time_end', event_col: str = 'event_default') -> dict:
    """
    Run chiizer on multiple variables.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data
    variables : list
        List of variable names to analyze
    time_col : str
        Time column
    event_col : str
        Event column

    Returns
    -------
    dict
        Combined results for all variables
    """
    if variables is None:
        variables = ['income', 'debt_to_income', 'LTV_ratio', 'employment_years']

    all_results = {}
    all_fitters = {}

    for var in variables:
        print(f"\nChiizing: {var}")
        fitters, results = chiize_variable(df, var, n_bins=4,
                                           time_col=time_col, event_col=event_col)
        all_fitters[var] = fitters
        all_results[var] = results

        print(f"  Bins created: {len(results)}")
        for label, res in results.items():
            print(f"    {label}: N={res['n']}, Events={res['events']}, Mean={res['mean_value']}")

    return all_fitters, all_results


def main(df: pd.DataFrame):
    variables = ['income', 'debt_to_income', 'LTV_ratio', 'employment_years']
    all_fitters, all_results = run_full_chiizer(df, variables)

    for var, fitters in all_fitters.items():
        plot_chiizer_results(fitters, var)

    return all_fitters, all_results


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    main(df)