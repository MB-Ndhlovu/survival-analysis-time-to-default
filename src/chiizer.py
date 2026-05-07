"""Risk Chiizer: bin continuous variables into risk categories and compute survival curves."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def bin_variable(df: pd.DataFrame, var: str, n_bins: int = 4, labels: list = None) -> pd.Series:
    """Bin a continuous variable into quantiles."""
    # Use retbins=True to get actual number of bins after dropping duplicates
    try:
        bins = pd.qcut(df[var], q=n_bins, duplicates='drop', retbins=True)[1]
        actual_bins = len(bins) - 1
    except ValueError:
        actual_bins = n_bins
    
    if labels is None:
        labels = [f'Q{i+1}' for i in range(actual_bins)]
    else:
        labels = labels[:actual_bins]
    
    return pd.qcut(df[var], q=n_bins, labels=labels, duplicates='drop')


def chiize_risk(
    df: pd.DataFrame,
    output_dir: str = 'reports'
) -> dict:
    """Build a risk chiizer by bining key variables and computing survival curves.

    Creates risk categories for:
    - Debt-to-income ratio
    - LTV ratio
    - Employment years
    - Loan amount

    Returns dict with survival stats for each bin.
    """
    df = df.copy()

    # Bin key risk factors
    df['DTI_bin'] = bin_variable(df, 'debt_to_income', n_bins=4,
                                 labels=['Low DTI (<20%)', 'Medium DTI (20-35%)', 
                                        'High DTI (35-50%)', 'Very High DTI (>50%)'])
    
    df['LTV_bin'] = bin_variable(df, 'LTV_ratio', n_bins=3,
                                 labels=['Low LTV (<0.70)', 'Medium LTV (0.70-0.90)', 'High LTV (>0.90)'])
    
    df['Emp_bin'] = bin_variable(df, 'employment_years', n_bins=3,
                                 labels=['<2 Years', '2-8 Years', '8+ Years'])
    
    df['Loan_bin'] = bin_variable(df, 'loan_amount', n_bins=4,
                                  labels=['Small (<$50k)', 'Medium ($50-150k)', 
                                         'Large ($150-300k)', 'Jumbo (>300k)'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    chiizer_results = {}

    # DTI analysis
    kmf = KaplanMeierFitter()
    dti_results = {}
    for i, (cat, group_df) in enumerate(df.groupby('DTI_bin', observed=True)):
        kmf.fit(group_df['time_end'], group_df['event_default'], label=cat)
        surv_12 = kmf.survival_function_at_times(12).values[0]
        surv_24 = kmf.survival_function_at_times(24).values[0]
        med = kmf.median_survival_time_
        med = med if not np.isinf(med) else None
        dti_results[cat] = {'survival_12': round(float(surv_12), 4), 
                            'survival_24': round(float(surv_24), 4),
                            'median': med}
        kmf.plot_survival_function(ax=axes[0], ci_show=True)

    axes[0].set_title('Survival by DTI Ratio', fontsize=11)
    axes[0].set_xlabel('Months'); axes[0].set_ylabel('Survival Probability')
    axes[0].legend(loc='lower left'); axes[0].grid(True, alpha=0.3)
    chiizer_results['DTI'] = dti_results

    # LTV analysis
    kmf = KaplanMeierFitter()
    ltv_results = {}
    for cat, group_df in df.groupby('LTV_bin', observed=True):
        kmf.fit(group_df['time_end'], group_df['event_default'], label=cat)
        surv_12 = kmf.survival_function_at_times(12).values[0]
        surv_24 = kmf.survival_function_at_times(24).values[0]
        med = kmf.median_survival_time_
        med = med if not np.isinf(med) else None
        ltv_results[cat] = {'survival_12': round(float(surv_12), 4),
                           'survival_24': round(float(surv_24), 4),
                           'median': med}
        kmf.plot_survival_function(ax=axes[1], ci_show=True)

    axes[1].set_title('Survival by LTV Ratio', fontsize=11)
    axes[1].set_xlabel('Months'); axes[1].set_ylabel('Survival Probability')
    axes[1].legend(loc='lower left'); axes[1].grid(True, alpha=0.3)
    chiizer_results['LTV'] = ltv_results

    # Employment analysis
    kmf = KaplanMeierFitter()
    emp_results = {}
    for cat, group_df in df.groupby('Emp_bin', observed=True):
        kmf.fit(group_df['time_end'], group_df['event_default'], label=cat)
        surv_12 = kmf.survival_function_at_times(12).values[0]
        surv_24 = kmf.survival_function_at_times(24).values[0]
        med = kmf.median_survival_time_
        med = med if not np.isinf(med) else None
        emp_results[cat] = {'survival_12': round(float(surv_12), 4),
                            'survival_24': round(float(surv_24), 4),
                            'median': med}
        kmf.plot_survival_function(ax=axes[2], ci_show=True)

    axes[2].set_title('Survival by Employment Length', fontsize=11)
    axes[2].set_xlabel('Months'); axes[2].set_ylabel('Survival Probability')
    axes[2].legend(loc='lower left'); axes[2].grid(True, alpha=0.3)
    chiizer_results['Employment'] = emp_results

    # Loan amount analysis
    kmf = KaplanMeierFitter()
    loan_results = {}
    for cat, group_df in df.groupby('Loan_bin', observed=True):
        kmf.fit(group_df['time_end'], group_df['event_default'], label=cat)
        surv_12 = kmf.survival_function_at_times(12).values[0]
        surv_24 = kmf.survival_function_at_times(24).values[0]
        med = kmf.median_survival_time_
        med = med if not np.isinf(med) else None
        loan_results[cat] = {'survival_12': round(float(surv_12), 4),
                             'survival_24': round(float(surv_24), 4),
                             'median': med}
        kmf.plot_survival_function(ax=axes[3], ci_show=True)

    axes[3].set_title('Survival by Loan Amount', fontsize=11)
    axes[3].set_xlabel('Months'); axes[3].set_ylabel('Survival Probability')
    axes[3].legend(loc='lower left'); axes[3].grid(True, alpha=0.3)
    chiizer_results['Loan Amount'] = loan_results

    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_chiizer.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n=== Risk Chiizer Results ===")
    for factor, bins in chiizer_results.items():
        print(f"\n{factor}:")
        for cat, stats in bins.items():
            med = f"{stats['median']:.1f}" if stats['median'] else 'Not reached'
            print(f"  {cat:<25} 12mo: {stats['survival_12']:.2%}  24mo: {stats['survival_24']:.2%}  Median: {med}")

    return chiizer_results


if __name__ == '__main__':
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results = chiize_risk(df)
    import json; print(json.dumps(results, indent=2))