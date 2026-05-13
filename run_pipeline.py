"""
Execute full survival analysis pipeline.
Generates data, fits models, produces outputs and saves results.
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier, print_summary as print_km_summary
from src.cox_ph import fit_cox_ph, print_cox_summary, get_top_risk_factors
from src.chiizer import chiize_all_variables, print_chiizer_summary
from src.predict_survival import demo_prediction


def run_pipeline():
    """Run complete survival analysis pipeline."""

    print("="*70)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("="*70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic loan data...")
    df = generate_loan_data(n=5000, censor_at=24)
    print(f"  Generated {len(df)} loans")
    print(f"  Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"  Censored: {(~df['event_default'].astype(bool)).sum()} ({(1-df['event_default'].mean())*100:.1f}%)")

    # 2. Kaplan-Meier analysis
    print("\n[2/5] Fitting Kaplan-Meier survival curves...")
    km_results = fit_kaplan_meier(df)
    print_km_summary(km_results)
    print("  Saved: reports/kaplan_meier_curves.png")

    # 3. Cox PH analysis
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cph = fit_cox_ph(df)
    cox_summary = print_cox_summary(cph)
    top_risks = get_top_risk_factors(cox_summary)
    print("\n  Top 3 Risk Factors:")
    for i, risk in enumerate(top_risks, 1):
        print(f"    {i}. {risk['variable']}: {risk['interpretation']}")

    # 4. Chiizer analysis
    print("\n[4/5] Computing risk chiizer analysis...")
    chiizer_results = chiize_all_variables(df)
    print_chiizer_summary(chiizer_results)
    print("  Saved: reports/chiizer_*.png")

    # 5. Prediction for new applicants
    print("\n[5/5] Generating applicant predictions...")
    predictions = demo_prediction(df)
    print("  Saved: reports/applicant_predictions.png")

    # Compile results for JSON export
    results = {
        "dataset_info": {
            "n_loans": int(len(df)),
            "n_defaults": int(df['event_default'].sum()),
            "n_censored": int((~df['event_default'].astype(bool)).sum()),
            "censorship_rate": float(1 - df['event_default'].mean())
        },
        "kaplan_meier": {
            band: {
                "median_survival_months": float(r['median_survival']),
                "survival_at_12_months": float(r['survival_at_12']),
                "survival_at_24_months": float(r['survival_at_24'])
            }
            for band, r in km_results.items()
        },
        "cox_ph_hazard_ratios": {
            var: {
                "hazard_ratio": float(row['hazard_ratio']),
                "coef": float(row['coef']),
                "p_value": float(row['p'])
            }
            for var, row in cox_summary.iterrows()
        },
        "top_risk_factors": top_risks,
        "applicant_predictions": [
            {
                "months_12_survival": float(p['survival_at_12']),
                "months_24_survival": float(p['survival_at_24']),
                "median_survival_months": p['median_survival']
            }
            for p in predictions
        ]
    }

    # Save results
    output_path = '/home/workspace/Projects/survival-analysis-time-to-default/reports/survival_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")
    print("\nKey Insights:")
    print("  - Survival analysis reveals WHEN default occurs, not just IF")
    print("  - Lower credit bands show significantly higher default hazard")
    print("  - Cox PH identifies which factors accelerate default timing")
    print("  - Personalized survival curves enable risk-based pricing")

    return results


if __name__ == "__main__":
    results = run_pipeline()
    sys.exit(0)