"""Risk Chiizer - bin continuous variables into risk categories and analyze survival."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(series: pd.Series, n_bins: int = 4, labels: list = None) -> pd.Series:
    """
    Bin a continuous variable into categories.
    
    Parameters
    ----------
    series : pd.Series
        Continuous variable to bin
    n_bins : int
        Number of bins
    labels : list, optional
        Custom bin labels
    
    Returns
    -------
    pd.Series
        Binned categorical variable
    """
    try:
        binned = pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # Fall back to equal width bins if qcut fails
        binned = pd.cut(series, bins=n_bins, labels=labels, duplicates='drop')
    return binned


def compute_survival_by_bin(df: pd.DataFrame, variable: str, n_bins: int = 4) -> dict:
    """
    Compute survival statistics for each bin of a continuous variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Loan data with survival columns
    variable : str
        Variable name to analyze
    n_bins : int
        Number of bins
    
    Returns
    -------
    dict
        Survival statistics by bin
    """
    df = df.copy()
    
    if variable == 'credit_score':
        bins = [0, 580, 670, 740, 1000]
        labels = ['Very Low (<580)', 'Low (580-669)', 'Medium (670-739)', 'High (740+)']
        df['binned'] = pd.cut(df[variable], bins=bins, labels=labels)
    elif variable == 'debt_to_income':
        bins = [0, 0.2, 0.35, 0.45, 1.0]
        labels = ['Low (<20%)', 'Medium (20-35%)', 'High (35-45%)', 'Very High (45%+)']
        df['binned'] = pd.cut(df[variable], bins=bins, labels=labels)
    elif variable == 'LTV_ratio':
        bins = [0, 0.6, 0.75, 0.85, 2.0]
        labels = ['Low (<60%)', 'Medium (60-75%)', 'High (75-85%)', 'Very High (85%+)']
        df['binned'] = pd.cut(df[variable], bins=bins, labels=labels)
    elif variable == 'interest_rate':
        bins = [0, 5, 8, 12, 30]
        labels = ['Low (<5%)', 'Medium (5-8%)', 'High (8-12%)', 'Very High (12%+)']
        df['binned'] = pd.cut(df[variable], bins=bins, labels=labels)
    else:
        df['binned'] = bin_variable(df[variable], n_bins)
    
    results = {}
    kmf = KaplanMeierFitter()
    
    for bin_label in df['binned'].dropna().unique():
        bin_df = df[df['binned'] == bin_label]
        
        kmf_temp = KaplanMeierFitter()
        kmf_temp.fit(bin_df['time_end'], bin_df['event_default'])
        
        results[str(bin_label)] = {
            'n_obs': len(bin_df),
            'n_events': bin_df['event_default'].sum(),
            'surv_12m': round(kmf_temp.survival_function_at_times(12).values[0], 4),
            'surv_24m': round(kmf_temp.survival_function_at_times(24).values[0], 4),
        }
        
        try:
            median = kmf_temp.median_survival_time_
            results[str(bin_label)]['median_time'] = round(median, 2) if not np.isinf(median) else None
        except Exception:
            results[str(bin_label)]['median_time'] = None
    
    return results


def print_chiizer_summary(df: pd.DataFrame) -> None:
    """Print risk chiizer summary for key variables."""
    variables = ['credit_score', 'debt_to_income', 'LTV_ratio', 'interest_rate']
    
    print("\n" + "="*70)
    print("RISK CHIIZER ANALYSIS - SURVIVAL BY RISK CATEGORY")
    print("="*70)
    
    for var in variables:
        print(f"\n{var.upper().replace('_', ' ')}")
        print("-" * 60)
        
        results = compute_survival_by_bin(df, var)
        
        # Sort by survival (descending - lowest risk first)
        sorted_bins = sorted(results.items(), 
                           key=lambda x: x[1]['surv_12m'], 
                           reverse=True)
        
        print(f"{'Category':<25} {'N':>6} {'Defaults':>8} {'12m Surv':>10} {'24m Surv':>10}")
        print("-" * 60)
        
        for bin_label, stats in sorted_bins:
            print(f"{bin_label:<25} {stats['n_obs']:>6} {stats['n_events']:>8} "
                  f"{stats['surv_12m']:>10.1%} {stats['surv_24m']:>10.1%}")


def plot_chiizer_curves(df: pd.DataFrame, variable: str, 
                       save_path: str = None) -> None:
    """Plot survival curves for each bin of a variable."""
    df = df.copy()
    
    if variable == 'credit_score':
        bins = [0, 580, 670, 740, 1000]
        labels = ['Very Low (<580)', 'Low (580-669)', 'Medium (670-739)', 'High (740+)']
        df['binned'] = pd.cut(df[variable], bins=bins, labels=labels)
    elif variable == 'debt_to_income':
        bins = [0, 0.2, 0.35, 0.45, 1.0]
        labels = ['Low (<20%)', 'Medium (20-35%)', 'High (35-45%)', 'Very High (45%+)']
        df['binned'] = pd.cut(df[variable], bins=bins, labels=labels)
    else:
        df['binned'] = bin_variable(df[variable], 4)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'][:len(df['binned'].dropna().unique())]
    
    kmf = KaplanMeierFitter()
    for i, bin_label in enumerate(sorted(df['binned'].dropna().unique())):
        bin_df = df[df['binned'] == bin_label]
        kmf.fit(bin_df['time_end'], bin_df['event_default'], label=str(bin_label))
        kmf.plot_survival_function(ax=ax, color=colors[i] if i < len(colors) else None)
    
    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Curves by {variable.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved chiizer plot to {save_path}")


if __name__ == '__main__':
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    print_chiizer_summary(df)