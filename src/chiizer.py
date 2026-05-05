"""Risk Chiizer - bin continuous variables into risk categories."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def create_risk_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create risk categories for key variables."""
    df = df.copy()
    
    # Credit score bands
    df['credit_risk'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 670, 740, 850],
        labels=['High', 'Medium', 'Low', 'Prime']
    )
    
    # DTI bands
    df['dti_risk'] = pd.cut(
        df['debt_to_income'],
        bins=[0, 0.28, 0.36, 0.43, 1.0],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    # LTV bands
    df['ltv_risk'] = pd.cut(
        df['LTV_ratio'],
        bins=[0, 0.6, 0.8, 0.95, 1.0],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    # Employment stability
    df['employment_status'] = pd.cut(
        df['employment_years'],
        bins=[-1, 1, 3, 7, 100],
        labels=['New', 'Developing', 'Established', 'Senior']
    )
    
    return df


def plot_chiized_survival(df: pd.DataFrame, variable: str, save_path: str = None):
    """Plot survival curves for a chiized variable."""
    kmf = KaplanMeierFitter()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'credit_risk': {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c', 'Prime': '#1f77b4'},
        'dti_risk': {'Low': '#2ca02c', 'Moderate': '#1f77b4', 'High': '#ff7f0e', 'Very High': '#d62728'},
        'ltv_risk': {'Low': '#2ca02c', 'Moderate': '#1f77b4', 'High': '#ff7f0e', 'Very High': '#d62728'},
        'employment_status': {'New': '#d62728', 'Developing': '#ff7f0e', 'Established': '#2ca02c', 'Senior': '#1f77b4'}
    }
    
    for category in df[variable].cat.categories:
        subset = df[df[variable] == category]
        if len(subset) > 10:
            kmf.fit(subset['time_end'], subset['event_default'], label=f"{category} (n={len(subset)})")
            color = colors.get(variable, {}).get(category, '#333333')
            ax.plot(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0],
                   label=f"{category}", color=color, linewidth=2)
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Curves by {variable.replace("_", " ").title()}', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def chiizer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize survival by chiized risk categories."""
    df = create_risk_bins(df)
    
    rows = []
    for variable in ['credit_risk', 'dti_risk', 'ltv_risk', 'employment_status']:
        for category in df[variable].cat.categories:
            subset = df[df[variable] == category]
            if len(subset) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(subset['time_end'], subset['event_default'])
                
                rows.append({
                    'variable': variable,
                    'category': str(category),
                    'n': len(subset),
                    'defaults': subset['event_default'].sum(),
                    'default_rate': round(subset['event_default'].mean() * 100, 1),
                    'survival_24m': round(kmf.predict(24) * 100, 1),
                    'median_months': kmf.median_survival_time_ if not np.isinf(kmf.median_survival_time_) else 'N/A'
                })
    
    return pd.DataFrame(rows)


def run_chiizer_analysis(df: pd.DataFrame, output_dir: str = 'reports') -> dict:
    """Run full chiizer analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = create_risk_bins(df)
    
    for variable in ['credit_risk', 'dti_risk', 'ltv_risk']:
        plot_chiized_survival(df, variable, f"{output_dir}/chiizer_{variable}.png")
    
    summary_df = chiizer_summary(df)
    
    return {
        'chiized_data': df,
        'summary': summary_df.to_dict('records')
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    
    df = load_data()
    result = run_chiizer_analysis(df)
    
    print("=== Risk Chiizer Summary ===")
    summary_df = pd.DataFrame(result['summary'])
    print(summary_df.to_string(index=False))