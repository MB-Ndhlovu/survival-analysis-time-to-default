"""Kaplan-Meier survival analysis by credit score bands."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def assign_credit_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Assign credit score bands."""
    df = df.copy()
    bands = ['<580', '580-669', '670-739', '740+']
    df['credit_band'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 670, 740, 1000],
        labels=bands
    )
    return df


def fit_km_by_band(df: pd.DataFrame) -> dict:
    """Fit Kaplan-Meier curves for each credit score band."""
    df = assign_credit_bands(df)
    results = {}
    
    for band in ['<580', '580-669', '670-739', '740+']:
        band_df = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(
            band_df['time_end'],
            band_df['event_default'],
            label=band
        )
        results[band] = {
            'kmf': kmf,
            'median_survival': kmf.median_survival_time_,
            'n_obs': len(band_df),
            'n_events': band_df['event_default'].sum()
        }
    
    return results


def plot_km_curves(results: dict, save_path: str = None):
    """Plot Kaplan-Meier survival curves by credit band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'<580': '#d62728', '580-669': '#ff7f0e', '670-739': '#2ca02c', '740+': '#1f77b4'}
    
    for band, data in results.items():
        kmf = data['kmf']
        ax.plot(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            label=f"{band} (n={data['n_obs']}, events={data['n_events']})",
            color=colors[band],
            linewidth=2
        )
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Credit Score Band', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_survival_probabilities(results: dict) -> pd.DataFrame:
    """Compute 12-month and 24-month survival probabilities."""
    rows = []
    for band, data in results.items():
        kmf = data['kmf']
        surv_func = kmf.survival_function_
        
        s12 = kmf.predict(12) if 12 in surv_func.index else np.nan
        s24 = kmf.predict(24)
        
        rows.append({
            'credit_band': band,
            'n_obs': data['n_obs'],
            'n_events': data['n_events'],
            'median_survival_months': data['median_survival'],
            'survival_12m': s12,
            'survival_24m': s24
        })
    
    return pd.DataFrame(rows)


def run_km_analysis(df: pd.DataFrame, output_dir: str = 'reports') -> dict:
    """Run full Kaplan-Meier analysis."""
    results = fit_km_by_band(df)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plot_km_curves(results, f"{output_dir}/km_survival_curves.png")
    
    probs_df = compute_survival_probabilities(results)
    
    return {
        'results': results,
        'survival_probabilities': probs_df.to_dict('records')
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    
    df = load_data()
    results = fit_km_by_band(df)
    
    print("=== Kaplan-Meier Results by Credit Score Band ===")
    for band, data in results.items():
        print(f"\n{band}:")
        print(f"  Observations: {data['n_obs']}")
        print(f"  Defaults: {data['n_events']}")
        print(f"  Median survival: {data['median_survival']:.1f} months")
    
    probs = compute_survival_probabilities(results)
    print("\n=== Survival Probabilities ===")
    print(probs.to_string(index=False))