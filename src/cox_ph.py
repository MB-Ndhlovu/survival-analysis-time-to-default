"""
Cox Proportional Hazards model for loan default analysis.
Examines how various factors affect the hazard of default.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from typing import Dict, Tuple


def fit_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    """
    Fit Cox Proportional Hazards model on loan features.

    Args:
        df: DataFrame with survival data and features

    Returns:
        Fitted CoxPHFitter model
    """
    cph = CoxPHFitter()

    # Select features for the model
    features = ['credit_score', 'income', 'employment_years', 'debt_to_income',
               'loan_amount', 'interest_rate', 'LTV_ratio']

    # Prepare data - drop rows with missing values
    model_df = df[features + ['time_end', 'event_default']].dropna()

    cph.fit(
        model_df,
        duration_col='time_end',
        event_col='event_default'
    )

    return cph


def extract_coefficients(cph: CoxPHFitter) -> pd.DataFrame:
    """
    Extract Cox model coefficients with hazard ratios and confidence intervals.

    Args:
        cph: Fitted CoxPHFitter

    Returns:
        DataFrame with coefficients, hazard ratios, and CI bounds
    """
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['hr_95CI_lower'] = np.exp(summary['coef lower 95%'])
    summary['hr_95CI_upper'] = np.exp(summary['coef upper 95%'])

    # Reorder columns
    summary = summary[['coef', 'hazard_ratio', 'hr_95CI_lower', 'hr_95CI_upper', 'p']]
    summary.columns = ['Coefficient', 'Hazard Ratio', 'HR 95% CI Lower', 'HR 95% CI Upper', 'P-value']

    return summary


def print_cox_summary(cph: CoxPHFitter, summary: pd.DataFrame):
    """Print formatted Cox PH model summary."""
    print("\n" + "="*70)
    print("COX PROPORTIONAL HAZARDS MODEL")
    print("="*70)

    print("\n🔍 Model Concordance:")
    print(f"   Concordance Index: {cph.concordance_index_:.4f}")

    print("\n📉 Coefficient Analysis (sorted by absolute magnitude):")
    print("-" * 80)
    print(f"{'Variable':<20} {'Coef':>10} {'HR':>10} {'95% CI':>18} {'P-value':>10}")
    print("-" * 80)

    for var in summary.index:
        row = summary.loc[var]
        ci_str = f"[{row['HR 95% CI Lower']:.2f}, {row['HR 95% CI Upper']:.2f}]"
        sig = "***" if row['P-value'] < 0.001 else "**" if row['P-value'] < 0.01 else "*" if row['P-value'] < 0.05 else ""
        print(f"{var:<20} {row['Coefficient']:>10.4f} {row['Hazard Ratio']:>10.3f} {ci_str:>18} {row['P-value']:>9.4f} {sig}")

    print("\n📊 Key Risk Drivers (HR interpretation):")
    print("-" * 50)
    hr_col = summary['Hazard Ratio']

    # Find biggest risk factors
    risks = hr_col.sort_values(ascending=False).head(3)
    print("   ↑ Top risk increase factors:")
    for var in risks.index:
        hr = hr_col[var]
        if hr > 1:
            pct = (hr - 1) * 100
            print(f"      - {var}: HR={hr:.3f} (+{pct:.1f}% hazard per unit)")

    # Find protective factors
    protective = hr_col.sort_values().head(3)
    print("\n   ↓ Top protective factors:")
    for var in protective.index:
        hr = hr_col[var]
        if hr < 1:
            pct = (1 - hr) * 100
            print(f"      - {var}: HR={hr:.3f} (-{pct:.1f}% hazard per unit)")

    return summary


def plot_cox_hazard_ratios(summary: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Plot hazard ratios with confidence intervals.

    Args:
        summary: DataFrame with hazard ratios and CI bounds
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(summary))
    hr = summary['Hazard Ratio']
    ci_low = summary['HR 95% CI Lower']
    ci_high = summary['HR 95% CI Upper']

    # Horizontal bar chart
    colors = ['#d62728' if v > 1 else '#2ca02c' for v in hr]
    ax.barh(y_pos, hr, color=colors, alpha=0.7)

    # Error bars for CI
    ax.errorbar(hr, y_pos, xerr=[hr - ci_low, ci_high - hr],
                fmt='none', color='black', capsize=5, alpha=0.7)

    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel('Hazard Ratio', fontsize=12)
    ax.set_title('Cox PH Model: Hazard Ratios with 95% CI', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d62728', alpha=0.7, label='Increases Risk'),
                      Patch(facecolor='#2ca02c', alpha=0.7, label='Decreases Risk')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands

    df = add_credit_bands(generate_loan_data())
    cph = fit_cox_model(df)
    summary = extract_coefficients(cph)
    print_cox_summary(cph, summary)
    plot_cox_hazard_ratios(summary, 'cox_hr.png')
    print("\n✅ Cox PH analysis complete")