"""
Risk Chiizer — discretize continuous variables into risk categories and 
compute survival curves for each bin.

The "chiizer" name references the chi-squared statistic used to test
whether survival curves across bins are statistically different.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def bin_variable(df, col, n_bins=4, strategy='quantile'):
    """
    Bin a continuous variable into categories.
    
    Parameters:
    - col: column name
    - n_bins: number of bins
    - strategy: 'quantile' (equal count) or 'equal' (equal width)
    
    Returns DataFrame with new 'col_binned' column.
    """
    df = df.copy()
    col_data = df[col].dropna()
    
    if strategy == 'quantile':
        try:
            binned = pd.qcut(col_data, q=n_bins, duplicates='drop')
            bin_edges = binned.cat.categories.left.tolist() + [binned.cat.categories.right.tolist()[-1]]
        except ValueError:
            bin_edges = np.linspace(col_data.min(), col_data.max(), n_bins + 1).tolist()
    else:
        bin_edges = np.linspace(col_data.min(), col_data.max(), n_bins + 1).tolist()
    
    # Remove duplicate edges and ensure we have valid bins
    bin_edges = sorted(list(set(bin_edges)))
    if len(bin_edges) < 2:
        bin_edges = [col_data.min(), col_data.max()]
    
    n_actual_bins = len(bin_edges) - 1
    
    # Create unique labels
    if n_actual_bins <= 4:
        suffixes = ['Low', 'Med-Low', 'Med-High', 'High'][:n_actual_bins]
    else:
        suffixes = [str(i) for i in range(n_actual_bins)]
    
    labels = [f'{i+1}' for i in range(n_actual_bins)]
    
    df.loc[df[col].notna(), f'{col}_bin'] = pd.cut(
        df.loc[df[col].notna(), col], 
        bins=bin_edges, 
        labels=labels, 
        include_lowest=True,
        ordered=False
    )
    
    return df


def fit_survival_by_bin(df, duration_col, event_col, bin_col):
    """
    Fit Kaplan-Meier survival curves for each bin of a variable.
    
    Returns dict: bin_label -> {kmf, survival_at_times, median}
    """
    unique_bins = df[bin_col].dropna().unique()
    results = {}
    
    for bin_val in sorted(unique_bins, key=lambda x: float(str(x).split('-')[0])):
        bin_df = df[df[bin_col] == bin_val]
        
        if len(bin_df) < 10:  # Skip bins with too few observations
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(bin_df[duration_col], bin_df[event_col], label=str(bin_val))
        
        results[str(bin_val)] = {
            'kmf': kmf,
            'n_obs': len(bin_df),
            'n_defaults': bin_df[event_col].sum(),
            'survival_12': kmf.survival_function_at_times(12).values[0],
            'survival_24': kmf.survival_function_at_times(24).values[0],
        }
    
    return results


def chiize(df, duration_col='time_end', event_col='event_default'):
    """
    Run the full risk chiizer: bin key variables and compute survival curves.
    
    Variables to chiize: income, debt_to_income, loan_amount, LTV_ratio, employment_years
    """
    variables_config = {
        'income': {'n_bins': 4, 'strategy': 'quantile', 'label': 'Income (ZAR)'},
        'debt_to_income': {'n_bins': 4, 'strategy': 'quantile', 'label': 'Debt-to-Income'},
        'loan_amount': {'n_bins': 4, 'strategy': 'quantile', 'label': 'Loan Amount (ZAR)'},
        'LTV_ratio': {'n_bins': 4, 'strategy': 'quantile', 'label': 'LTV Ratio'},
        'employment_years': {'n_bins': 3, 'strategy': 'quantile', 'label': 'Employment Years'},
    }
    
    all_results = {}
    
    for var, config in variables_config.items():
        df_binned = bin_variable(df, var, n_bins=config['n_bins'], strategy=config['strategy'])
        bin_col = f'{var}_bin'
        
        results = fit_survival_by_bin(df_binned, duration_col, event_col, bin_col)
        
        all_results[var] = {
            'label': config['label'],
            'bin_results': results,
            'df': df_binned,
            'bin_col': bin_col,
        }
    
    return all_results


def plot_chiizer_results(all_results, save_path=None):
    """Plot survival curves for each binned variable."""
    n_vars = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (var, data) in enumerate(all_results.items()):
        ax = axes[i]
        label = data['label']
        bin_results = data['bin_results']
        
        for j, (bin_val, bin_data) in enumerate(bin_results.items()):
            kmf = bin_data['kmf']
            n = bin_data['n_obs']
            ax.plot(kmf.survival_function_.index, 
                    kmf.survival_function_.iloc[:, 0],
                    label=f'{bin_val}\n(n={n})',
                    color=colors[j % len(colors)],
                    linewidth=2)
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title(f'Survival by {label}', fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    # Hide last empty subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved chiizer plot to {save_path}")
    
    return fig


def print_chiizer_results(all_results):
    """Print chiizer results with logrank test for statistical significance."""
    print("\n" + "="*70)
    print("RISK CHIIZER — SURVIVAL ANALYSIS BY VARIABLE BINS")
    print("="*70)
    
    for var, data in all_results.items():
        label = data['label']
        bin_results = data['bin_results']
        
        print(f"\n{'='*70}")
        print(f"Variable: {label}")
        print(f"{'='*70}")
        
        print(f"\n{'Bin':<20} {'N':>6} {'Defaults':>9} {'12m Surv':>10} {'24m Surv':>10}")
        print("-"*60)
        
        for bin_val, bin_data in sorted(bin_results.items(), 
                                       key=lambda x: float(str(x[0]).split('-')[0])):
            n = bin_data['n_obs']
            d = bin_data['n_defaults']
            s12 = bin_data['survival_12']
            s24 = bin_data['survival_24']
            print(f"{str(bin_val):<20} {n:>6} {d:>9.0f} {s12:>10.1%} {s24:>10.1%}")
        
        # Logrank test to see if bins are statistically different
        bins = list(bin_results.keys())
        if len(bins) >= 2:
            print("\nLogrank test for equality of survival curves:")
            try:
                from lifelines.statistics import multivariate_logrank_test
                df = data['df']
                bin_col = data['bin_col']
                results = multivariate_logrank_test(
                    df[duration_col := 'time_end'], 
                    df[event_col := 'event_default'], 
                    df[bin_col]
                )
                print(f"  Test statistic: {results.test_statistic:.4f}")
                print(f"  p-value: {results.p_value:.4f}")
                if results.p_value < 0.05:
                    print("  → Bins are statistically different (p < 0.05)")
                else:
                    print("  → Bins are NOT statistically different (p >= 0.05)")
            except Exception as e:
                print(f"  (Could not compute logrank test: {e})")
    
    print("\n" + "="*70)


def chiizer_summary_table(all_results):
    """Build a consolidated summary DataFrame."""
    rows = []
    for var, data in all_results.items():
        label = data['label']
        for bin_val, bin_data in data['bin_results'].items():
            rows.append({
                'variable': label,
                'bin': str(bin_val),
                'n_obs': bin_data['n_obs'],
                'n_defaults': bin_data['n_defaults'],
                'survival_12m': round(bin_data['survival_12'], 4),
                'survival_24m': round(bin_data['survival_24'], 4),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands
    
    df = generate_loan_data()
    df = add_credit_bands(df)
    
    results = chiize(df)
    print_chiizer_results(results)
    
    summary = chiizer_summary_table(results)
    print("\n\nSummary Table:")
    print(summary.to_string(index=False))