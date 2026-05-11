"""
Kaplan-Meier survival analysis for loan default by credit score band.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def fit_by_credit_band(df):
    """
    Fit Kaplan-Meier curves for each credit score band.
    
    Returns dict with band -> (kmf, median_survival, survival_at_12, survival_at_24)
    """
    bands = ['Deep Subprime (<580)', 'Subprime (580-669)', 
             'Near Prime (670-739)', 'Prime (740+)']
    
    results = {}
    
    for band in bands:
        band_df = df[df['credit_band'] == band]
        
        if len(band_df) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df['time_end'],
            band_df['event_default'],
            label=band
        )
        
        # Median survival time (time when survival probability hits 50%)
        median_time = median_survival_times(kmf.survival_function_)
        
        # Survival probabilities at 12 and 24 months
        survival_12 = kmf.survival_function_at_times(12).values[0]
        survival_24 = kmf.survival_function_at_times(24).values[0]
        
        results[band] = {
            'kmf': kmf,
            'median_survival': median_time,
            'survival_12m': survival_12,
            'survival_24m': survival_24,
            'n_obs': len(band_df),
            'n_defaults': band_df['event_default'].sum(),
        }
        
    return results


def plot_kaplan_meier(results, save_path=None):
    """
    Plot Kaplan-Meier curves for all credit bands.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme by risk
    colors = {
        'Deep Subprime (<580)': '#d62728',      # red
        'Subprime (580-669)': '#ff7f0e',       # orange
        'Near Prime (670-739)': '#2ca02c',     # green
        'Prime (740+)': '#1f77b4',             # blue
    }
    
    ax = axes[0]
    for band, data in results.items():
        kmf = data['kmf']
        label = kmf.survival_function_.columns[0]
        ax.plot(kmf.survival_function_.index, 
                kmf.survival_function_[label],
                label=f"{band}\n(n={data['n_obs']})",
                color=colors.get(band, None),
                linewidth=2)
    
    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title('Kaplan-Meier Survival Curves by Credit Band', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% survival')
    
    # Plot with confidence intervals
    ax2 = axes[1]
    for band, data in results.items():
        kmf = data['kmf']
        # Plot main curve using the label as column key
        label = kmf.survival_function_.columns[0]
        ax2.plot(kmf.survival_function_.index, 
                 kmf.survival_function_[label],
                 label=band,
                 color=colors.get(band, None),
                 linewidth=2)
        # Confidence intervals — columns are label_lower and label_upper
        ci_lower_col = kmf.confidence_interval_.columns[0]
        ci_upper_col = kmf.confidence_interval_.columns[1]
        ax2.fill_between(
            kmf.survival_function_.index,
            kmf.confidence_interval_[ci_lower_col],
            kmf.confidence_interval_[ci_upper_col],
            alpha=0.15,
            color=colors.get(band, None)
        )
    
    ax2.set_xlabel('Time (months)', fontsize=11)
    ax2.set_ylabel('Survival Probability', fontsize=11)
    ax2.set_title('Survival Curves with 95% Confidence Intervals', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Kaplan-Meier plot to {save_path}")
    
    return fig


def print_results(results):
    """Print summary of Kaplan-Meier results."""
    print("\n" + "="*70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS BY CREDIT BAND")
    print("="*70)
    
    for band, data in results.items():
        median = data['median_survival']
        s12 = data['survival_12m']
        s24 = data['survival_24m']
        n = data['n_obs']
        d = data['n_defaults']
        
        print(f"\n{band}")
        print(f"  Observations: {n} | Defaults observed: {d:.0f}")
        print(f"  Median survival time: {median:.1f} months")
        print(f"  12-month survival: {s12:.1%}")
        print(f"  24-month survival: {s24:.1%}")
    
    print("\n" + "-"*70)
    print("\nInterpretation:")
    print("- Deep subprime loans show fastest default timing")
    print("- Prime loans show sustained survival over 24 months")
    print("- Median survival time = when 50% of loans have defaulted")
    print("-"*70)


def get_summary_table(results):
    """Build a summary DataFrame of KM results."""
    rows = []
    for band, data in results.items():
        rows.append({
            'credit_band': band,
            'n_observations': data['n_obs'],
            'n_defaults': data['n_defaults'],
            'median_survival_months': round(data['median_survival'], 1),
            'survival_12m': round(data['survival_12m'], 4),
            'survival_24m': round(data['survival_24m'], 4),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands
    
    df = generate_loan_data()
    df = add_credit_bands(df)
    
    results = fit_by_credit_band(df)
    print_results(results)
    
    table = get_summary_table(results)
    print("\n\nSummary Table:")
    print(table.to_string(index=False))