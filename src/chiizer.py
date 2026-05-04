"""
Risk Chiizer - bin continuous variables into risk categories
and compute survival curves for each bin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def create_risk_bins(df: pd.DataFrame, variable: str, n_bins: int = 4, strategy: str = 'quantile') -> pd.Series:
    """
    Bin a continuous variable into risk categories.

    Args:
        df: DataFrame containing the variable
        variable: Column name to bin
        n_bins: Number of bins to create
        strategy: 'quantile' for equal-frequency, 'uniform' for equal-width

    Returns:
        Series with bin labels
    """
    if strategy == 'quantile':
        labels = pd.qcut(df[variable], q=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')
    else:
        labels = pd.cut(df[variable], bins=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])

    return labels


def build_chiizer_matrix(df: pd.DataFrame, variables: list, n_bins: int = 4) -> pd.DataFrame:
    """
    Create a chiizer matrix showing survival characteristics by risk bin.

    Args:
        df: DataFrame with survival data
        variables: List of variable names to chiize
        n_bins: Number of bins for each variable

    Returns:
        DataFrame with summary statistics by variable and bin
    """
    results = []

    for var in variables:
        try:
            bins = create_risk_bins(df, var, n_bins)
            df_temp = df.copy()
            df_temp['bin'] = bins

            for bin_label in bins.unique():
                if pd.isna(bin_label):
                    continue
                bin_df = df_temp[df_temp['bin'] == bin_label]
                n_obs = len(bin_df)
                n_events = bin_df['event_default'].sum()
                event_rate = n_events / n_obs if n_obs > 0 else 0

                results.append({
                    'Variable': var,
                    'Bin': str(bin_label),
                    'N': n_obs,
                    'Events': n_events,
                    'Event Rate': event_rate
                })
        except Exception as e:
            print(f"Warning: Could not chiize {var}: {e}")
            continue

    return pd.DataFrame(results)


def plot_survival_by_bin(df: pd.DataFrame, variable: str, n_bins: int = 4,
                        title: str = None, save_path: str = None) -> plt.Figure:
    """
    Plot Kaplan-Meier survival curves for each bin of a variable.

    Args:
        df: DataFrame with survival data
        variable: Column to bin and plot
        n_bins: Number of bins
        title: Plot title
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = create_risk_bins(df, variable, n_bins)
    df_plot = df.copy()
    df_plot['bin'] = bins

    kmf = KaplanMeierFitter()

    # Color map for bins
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bins))

    for i, bin_label in enumerate(sorted(bins.unique(), key=str)):
        if pd.isna(bin_label):
            continue
        bin_df = df_plot[df_plot['bin'] == bin_label]
        kmf.fit(bin_df['time_end'], bin_df['event_default'], label=f'{variable} {bin_label}')
        kmf.plot_survival_function(ax=ax, color=[colors[i]])

    ax.set_title(title or f'Survival Curves by {variable} Risk Bins', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def chiize_income(df: pd.DataFrame) -> pd.DataFrame:
    """Chiize income into risk categories."""
    df = df.copy()
    df['income_bin'] = pd.cut(df['income'],
                              bins=[0, 50000, 80000, 120000, float('inf')],
                              labels=['Low (<50k)', 'Medium (50-80k)', 'High (80-120k)', 'Very High (120k+)'])
    return df


def chiize_dti(df: pd.DataFrame) -> pd.DataFrame:
    """Chiize debt-to-income into risk categories."""
    df = df.copy()
    df['dti_bin'] = pd.cut(df['debt_to_income'],
                          bins=[0, 0.15, 0.25, 0.35, 1.0],
                          labels=['Low (<15%)', 'Medium (15-25%)', 'High (25-35%)', 'Very High (>35%)'])
    return df


def chiize_loan_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Chiize loan amount into risk categories."""
    df = df.copy()
    df['loan_bin'] = pd.cut(df['loan_amount'],
                           bins=[0, 20000, 40000, 70000, float('inf')],
                           labels=['Small (<20k)', 'Medium (20-40k)', 'Large (40-70k)', 'Very Large (70k+)'])
    return df


def chiize_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all chiizations to a DataFrame."""
    df = chiize_income(df)
    df = chiize_dti(df)
    df = chiize_loan_amount(df)
    return df


def print_chiizer_summary(matrix: pd.DataFrame):
    """Print formatted chiizer matrix summary."""
    print("\n" + "="*70)
    print("RISK CHIIZER SUMMARY")
    print("="*70)

    for var in matrix['Variable'].unique():
        var_df = matrix[matrix['Variable'] == var].sort_values('Bin')
        print(f"\n📊 {var.upper()} Risk Bins:")
        print("-" * 50)
        print(f"{'Bin':<20} {'N':>8} {'Defaults':>10} {'Event Rate':>12}")
        print("-" * 50)
        for _, row in var_df.iterrows():
            bar = '█' * int(row['Event Rate'] * 20)
            print(f"{row['Bin']:<20} {row['N']:>8} {int(row['Events']):>10} {row['Event Rate']:>11.1%} {bar}")

    return matrix


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands

    df = add_credit_bands(generate_loan_data())
    df = chiize_all(df)

    variables = ['income', 'debt_to_income', 'loan_amount']
    matrix = build_chiizer_matrix(df, variables)

    print_chiizer_summary(matrix)

    for var in variables:
        plot_survival_by_bin(df, var, title=f'Survival by {var} Risk')

    print("\n✅ Risk chiizer complete")