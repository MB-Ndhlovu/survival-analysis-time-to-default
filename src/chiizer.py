"""
Risk Chiizer: bin continuous variables into risk categories.

Transforms continuous risk factors into discretized survival curves.
This creates interpretable risk stratification from raw covariates.

For each variable, bins the data and computes Kaplan-Meier curves per bin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(series: pd.Series, n_bins: int = 4, labels: list = None) -> pd.Series:
    """
    Discretize a continuous variable into bins with optional labels.

    Parameters
    ----------
    series : pd.Series
        Continuous variable to bin.
    n_bins : int
        Number of quantile bins.
    labels : list, optional
        Custom bin labels.

    Returns
    -------
    pd.Series
        Discretized bin labels.
    """
    if labels is None:
        labels = [f"Q{i+1}" for i in range(n_bins)]

    # Use qcut for equal-frequency bins (fail-safe on duplicates)
    try:
        binned = pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # Fallback to cut with equal width
        binned = pd.cut(series, bins=n_bins, labels=labels[:n_bins])

    return binned


def chiize_variable(df: pd.DataFrame, var_name: str, n_bins: int = 4,
                   labels: list = None, color_map: dict = None) -> pd.DataFrame:
    """
    Compute survival curves for each bin of a variable.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with time_end, event_default, and the variable column.
    var_name : str
        Name of the continuous variable to chiize.
    n_bins : int
        Number of bins.
    labels : list, optional
        Bin labels.
    color_map : dict, optional
        Mapping of bin labels to colors.

    Returns
    -------
    pd.DataFrame
        Summary stats per bin with survival probabilities.
    """
    if var_name not in df.columns:
        raise ValueError(f"Variable '{var_name}' not in DataFrame")

    df = df.copy()
    df['bin'] = bin_variable(df[var_name], n_bins, labels)

    kmf = KaplanMeierFitter()

    results_rows = []
    fig, ax = plt.subplots(figsize=(9, 5))

    if color_map is None:
        cmap = plt.cm.viridis
        colors = {label: cmap(i / max(n_bins - 1, 1)) for i, label in enumerate(labels or [f"Q{i+1}" for i in range(n_bins)])}
    else:
        colors = color_map
        cmap = None

    for i, (bin_label, group) in enumerate(df.groupby('bin', observed=True)):
        if len(group) < 10:
            continue

        kmf.fit(durations=group['time_end'],
                event_observed=group['event_default'],
                label=str(bin_label))

        surv_12 = kmf.predict(12)
        surv_24 = kmf.predict(24)
        median_surv = kmf.median_survival_time_

        results_rows.append({
            'bin': bin_label,
            'n': len(group),
            'n_defaults': group['event_default'].sum(),
            'survival_12mo': surv_12,
            'survival_24mo': surv_24,
            'median_time': median_surv if not pd.isna(median_surv) else ">24mo"
        })

        if color_map:
            color = colors.get(bin_label, '#1f77b4')
        else:
            color = cmap(i / max(n_bins - 1, 1))
        kmf.plot_survival_function(ax=ax, color=[color])

    ax.set_xlabel("Months since origination", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_title(f"Survival by {var_name} Risk Categories (Risk Chiizer)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title=var_name, loc="lower left")

    plt.tight_layout()
    plt.savefig(f"reports/chiizer_{var_name.replace('/', '_')}.png", dpi=150)
    plt.close()

    return pd.DataFrame(results_rows)


def chiize_all_variables(df: pd.DataFrame) -> dict:
    """
    Run risk chiizer on all continuous risk variables.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with time_end, event_default, and risk covariates.

    Returns
    -------
    dict
        Dictionary mapping variable name to chiizer summary DataFrame.
    """
    variables = ['debt_to_income', 'LTV_ratio', 'interest_rate']
    labels = {
        'debt_to_income': ['Low DTI', 'Med DTI', 'High DTI', 'V.High DTI'],
        'LTV_ratio': ['Low LTV', 'Med LTV', 'High LTV', 'V.High LTV'],
        'interest_rate': ['Low Rate', 'Med Rate', 'High Rate', 'V.High Rate']
    }

    color_maps = {
        'debt_to_income': {'Low DTI': '#2ca02c', 'Med DTI': '#1f77b4',
                           'High DTI': '#ff7f0e', 'V.High DTI': '#d62728'},
        'LTV_ratio': {'Low LTV': '#2ca02c', 'Med LTV': '#1f77b4',
                      'High LTV': '#ff7f0e', 'V.High LTV': '#d62728'},
        'interest_rate': {'Low Rate': '#2ca02c', 'Med Rate': '#1f77b4',
                          'High Rate': '#ff7f0e', 'V.High Rate': '#d62728'}
    }

    results = {}
    for var in variables:
        results[var] = chiize_variable(df, var, n_bins=4, labels=labels[var],
                                       color_map=color_maps[var])

    return results


def print_chiizer_summary(results: dict) -> None:
    """Print risk chiizer summary for all variables."""
    print("\n" + "=" * 80)
    print("RISK CHIIZER: SURVIVAL BY RISK CATEGORIES")
    print("=" * 80)

    for var, df_result in results.items():
        print(f"\n{var.upper().replace('_', ' ')}")
        print("-" * 60)
        print(f"{'Bin':<15} {'N':>6} {'Defaults':>8} {'12mo Surv':>10} {'24mo Surv':>10}")
        print("-" * 60)
        for _, row in df_result.iterrows():
            print(f"{str(row['bin']):<15} {row['n']:>6} {int(row['n_defaults']):>8} "
                  f"{row['survival_12mo']:>10.1%} {row['survival_24mo']:>10.1%}")
    print("=" * 80)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    results = chiize_all_variables(df)
    print_chiizer_summary(results)