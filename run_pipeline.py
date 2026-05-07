"""Execute full survival analysis pipeline."""

import json
import os
from datetime import datetime

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_km_by_credit_band
from src.cox_ph import fit_cox_ph
from src.chiizer import chiize_risk
from src.predict_survival import demonstrate_predictions


def run_pipeline(output_dir: str = 'reports'):
    """Run complete survival analysis pipeline.

    Steps:
    1. Generate synthetic loan data
    2. Fit Kaplan-Meier curves by credit band
    3. Fit Cox Proportional Hazards model
    4. Build risk chiizer (binned survival analysis)
    5. Predict survival for sample applicants
    6. Save results to JSON
    """
    print("=" * 65)
    print("  TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 65)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Generating loan data...")
    df = generate_loan_data(n_records=5000, seed=42)
    print(f"  Generated {len(df)} loan records")
    print(f"  Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"  Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")

    # 2. Kaplan-Meier
    print("\n[2/5] Fitting Kaplan-Meier survival curves...")
    km_results = fit_km_by_credit_band(df, output_dir)
    print(f"  Saved: {output_dir}/kaplan_meier_curves.png")

    # 3. Cox PH
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results, cph = fit_cox_ph(df)
    print(f"  Concordance Index: {cox_results['concordance_index']:.4f}")

    # 4. Risk Chiizer
    print("\n[4/5] Building risk chiizer...")
    chiizer_results = chiize_risk(df, output_dir)
    print(f"  Saved: {output_dir}/risk_chiizer.png")

    # 5. Predictions
    print("\n[5/5] Generating applicant predictions...")
    predictions = demonstrate_predictions(df, cph, output_dir)
    print(f"  Saved: {output_dir}/applicant_survival_curve.png")

    # Save results JSON
    results = {
        'generated_at': datetime.now().isoformat(),
        'n_records': len(df),
        'n_defaults': int(df['event_default'].sum()),
        'n_censored': int((df['event_default'] == 0).sum()),
        'km_by_credit_band': km_results,
        'cox_ph': {
            'concordance_index': cox_results['concordance_index'],
            'log_likelihood': cox_results['log_likelihood'],
            'AIC': cox_results['AIC'],
            'hazard_ratios': cox_results['hazard_ratios'],
        },
        'chiizer': chiizer_results,
        'predictions': {
            name: {
                'hazard_ratio': p['hazard_ratio'],
                'survival_12_months': p['survival_probs'][12],
                'survival_24_months': p['survival_probs'][24],
            }
            for name, p in predictions.items()
        }
    }

    results_path = f'{output_dir}/survival_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 65}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Output directory: {output_dir}/")
    print(f"  Results JSON: survival_results.json")
    print(f"  KM Plot: kaplan_meier_curves.png")
    print(f"  Chiizer: risk_chiizer.png")
    print(f"  Applicant curves: applicant_survival_curve.png")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    results = run_pipeline()