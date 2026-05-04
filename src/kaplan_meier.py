import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def create_credit_bands(df):
    """Create credit score bands."""
    bins = [0, 580, 670, 740, 900]
    labels = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']
    df['credit_band'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    return df

def fit_kaplan_meier(df):
    """Fit Kaplan-Meier curves for each credit band."""
    kmf = KaplanMeierFitter()
    
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Survival curves by credit band
    ax1 = axes[0]
    for band in ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']:
        band_data = df[df['credit_band'] == band]
        if len(band_data) > 0:
            kmf.fit(band_data['time_end'], band_data['event_default'], label=band)
            kmf.plot_survival_function(ax=ax1)
            
            # Get median survival time
            median = kmf.median_survival_time_
            results[band] = {
                'n': len(band_data),
                'events': band_data['event_default'].sum(),
                'median_survival': median if not np.isinf(median) else 'Not reached'
            }
    
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Kaplan-Meier Survival Curves by Credit Score Band')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Cumulative hazard
    ax2 = axes[1]
    from lifelines import NelsonAalenFitter
    naf = NelsonAalenFitter()
    
    for band in ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']:
        band_data = df[df['credit_band'] == band]
        if len(band_data) > 0:
            naf.fit(band_data['time_end'], band_data['event_default'], label=band)
            naf.plot_cumulative_hazard(ax=ax2)
    
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Cumulative Hazard')
    ax2.set_title('Cumulative Hazard by Credit Score Band')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/km_survival_curves.png', dpi=150)
    plt.close()
    
    return results, kmf

def compute_survival_probabilities(df, kmf):
    """Compute 12 and 24 month survival probabilities by segment."""
    results = {}
    
    for band in ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']:
        band_data = df[df['credit_band'] == band]
        if len(band_data) > 0:
            kmf.fit(band_data['time_end'], band_data['event_default'])
            
            s12 = kmf.predict(12)
            s24 = kmf.predict(24)
            
            results[band] = {
                'survival_12m': round(s12, 4),
                'survival_24m': round(s24, 4)
            }
    
    return results

if __name__ == "__main__":
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    df = create_credit_bands(df)
    results, kmf = fit_kaplan_meier(df)
    probs = compute_survival_probabilities(df, kmf)
    
    print("=== Kaplan-Meier Results ===")
    for band, data in results.items():
        print(f"\n{band}:")
        print(f"  n={data['n']}, events={data['events']}")
        print(f"  Median survival: {data['median_survival']}")
        print(f"  12m survival: {probs[band]['survival_12m']}")
        print(f"  24m survival: {probs[band]['survival_24m']}")