import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def build_risk_chiizer(df):
    """Build a risk chiizer by binning continuous variables into risk categories."""
    kmf = KaplanMeierFitter()
    
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Variable bins and labels
    bin_configs = {
        'income': {
            'bins': [0, 40000, 70000, 100000, float('inf')],
            'labels': ['Low (<40k)', 'Medium (40-70k)', 'High (70-100k)', 'Very High (100k+)'],
            'name': 'Income'
        },
        'debt_to_income': {
            'bins': [0, 0.3, 0.5, 0.7, float('inf')],
            'labels': ['Low (<0.3)', 'Medium (0.3-0.5)', 'High (0.5-0.7)', 'Very High (>0.7)'],
            'name': 'Debt-to-Income'
        },
        'employment_years': {
            'bins': [0, 2, 5, 10, float('inf')],
            'labels': ['New (<2y)', 'Established (2-5y)', 'Senior (5-10y)', 'Veteran (10y+)'],
            'name': 'Employment'
        },
        'loan_amount': {
            'bins': [0, 50000, 150000, 300000, float('inf')],
            'labels': ['Small (<50k)', 'Medium (50-150k)', 'Large (150-300k)', ' jumbo (300k+)'],
            'name': 'Loan Amount'
        },
        'LTV_ratio': {
            'bins': [0, 0.6, 0.8, 1.0, float('inf')],
            'labels': ['Low (<0.6)', 'Medium (0.6-0.8)', 'High (0.8-1.0)', 'Very High (>1.0)'],
            'name': 'LTV Ratio'
        },
        'interest_rate': {
            'bins': [0, 6, 10, 14, float('inf')],
            'labels': ['Low (<6%)', 'Medium (6-10%)', 'High (10-14%)', 'Very High (>14%)'],
            'name': 'Interest Rate'
        }
    }
    
    for i, (var, config) in enumerate(bin_configs.items()):
        ax = axes[i]
        
        df[f'{var}_bin'] = pd.cut(df[var], bins=config['bins'], labels=config['labels'], include_lowest=True)
        
        results[var] = {}
        
        for cat in config['labels']:
            cat_data = df[df[f'{var}_bin'] == cat]
            if len(cat_data) > 10:
                kmf.fit(cat_data['time_end'], cat_data['event_default'], label=cat)
                kmf.plot_survival_function(ax=ax, ci_show=True)
                
                s24 = kmf.predict(24)
                results[var][cat] = {'n': len(cat_data), 'survival_24m': round(s24, 4)}
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title(f'Survival by {config["name"]}')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/risk_chiizer.png', dpi=150)
    plt.close()
    
    return results

if __name__ == "__main__":
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    results = build_risk_chiizer(df)
    
    print("=== Risk Chiizer Results ===")
    for var, data in results.items():
        print(f"\n{var}:")
        for cat, stats in data.items():
            print(f"  {cat}: n={stats['n']}, 24m survival={stats['survival_24m']}")