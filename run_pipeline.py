"""Time-to-Default Survival Analysis Pipeline."""

import json
import os
from src.data_loader import load_data
from src.kaplan_meier import run_km_analysis, fit_km_by_band, compute_survival_probabilities
from src.cox_ph import run_cox_analysis, get_hazard_ratios
from src.chiizer import run_chiizer_analysis, chiizer_summary
from src.predict_survival import run_prediction_demo


def run_pipeline(output_dir: str = 'reports') -> dict:
    """Execute full survival analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS")
    print("=" * 60)
    
    print("\n[1/5] Loading loan data...")
    df = load_data()
    print(f"  Generated {len(df)} loan records")
    default_rate = df['event_default'].mean() * 100
    censor_rate = (1 - df['event_default'].mean()) * 100
    print(f"  Defaults: {int(df['event_default'].sum())} ({default_rate:.1f}%)")
    print(f"  Censored: {int(len(df) - df['event_default'].sum())} ({censor_rate:.1f}%)")
    
    print("\n[2/5] Running Kaplan-Meier analysis...")
    km_results = run_km_analysis(df, output_dir)
    print("  Kaplan-Meier curves saved to reports/km_survival_curves.png")
    print("\n  Survival by Credit Band:")
    for band, data in km_results['results'].items():
        median = data['median_survival'] if not (data['median_survival'] is None or
              (isinstance(data['median_survival'], float) and (data['median_survival'] == float('inf') or
               isinstance(data['median_survival'], float) and data['median_survival'] < 0))) else 'N/A'
        print(f"    {band}: median survival = {median} months")
    
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results = run_cox_analysis(df)
    cph = cox_results['cph']
    print(f"  Concordance Index: {cox_results['concordance_index']}")
    print("\n  Top Hazard Ratios (risk factors):")
    sorted_hrs = sorted(cox_results['hazard_ratios'], key=lambda x: x['hazard_ratio'], reverse=True)
    for i, hr in enumerate(sorted_hrs[:5]):
        sig = "*" if hr['p'] < 0.05 else ""
        var_name = hr.get('covariate', hr.get('variable', 'unknown'))
        print(f"    {i+1}. {var_name}: HR={hr['hazard_ratio']:.3f}{sig} (p={hr['p']:.4f})")
    
    print("\n[4/5] Building risk chiizer...")
    chiizer_results = run_chiizer_analysis(df, output_dir)
    print("  Chiizer plots saved to reports/")
    chiizer_df = chiizer_summary(df)
    
    print("\n[5/5] Generating applicant predictions...")
    pred_results = run_prediction_demo(df, output_dir)
    print("  Sample applicant survival predictions:")
    for pred in pred_results['predictions']:
        med_str = f"{pred['median_months']} months" if pred['median_months'] != 'N/A' else 'N/A'
        print(f"    {pred['name']}: 12m={pred['survival_12m']:.1%}, 24m={pred['survival_24m']:.1%}, median={med_str}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    results = {
        'data_summary': {
            'n_samples': len(df),
            'n_defaults': int(df['event_default'].sum()),
            'default_rate': round(default_rate, 2),
            'censor_rate': round(censor_rate, 2)
        },
        'km_results': {
            'survival_probabilities': km_results['survival_probabilities'],
            'median_survival_by_band': {
                band: data['median_survival'] if not (data['median_survival'] is None or
                       (isinstance(data['median_survival'], float) and (data['median_survival'] == float('inf') or
                        isinstance(data['median_survival'], float) and data['median_survival'] < 0))) else None
                for band, data in km_results['results'].items()
            }
        },
        'cox_ph': {
            'concordance_index': cox_results['concordance_index'],
            'hazard_ratios': cox_results['hazard_ratios'],
            'interpretations': cox_results['interpretations']
        },
        'chiizer_summary': chiizer_results['summary'],
        'predictions': pred_results['predictions']
    }
    
    output_path = f"{output_dir}/survival_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_pipeline()