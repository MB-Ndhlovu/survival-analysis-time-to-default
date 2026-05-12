"""Run the complete survival analysis pipeline."""

import sys
import json
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier, plot_km_curves, print_km_summary
from src.cox_ph import fit_cox_ph, print_cox_results
from src.chiizer import print_chiizer_summary
from src.predict_survival import predict_for_new_applicant, print_applicant_prediction


def run_pipeline():
    """Execute full survival analysis pipeline."""
    print("="*70)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("="*70)
    
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    # 1. Generate data
    print("\n[1/5] Generating synthetic loan data...")
    df = generate_loan_data(n_observations=5000)
    print(f"  Generated {len(df)} records")
    print(f"  Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"  Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")
    
    # 2. Kaplan-Meier analysis
    print("\n[2/5] Running Kaplan-Meier analysis...")
    km_results = fit_kaplan_meier(df)
    print_km_summary(km_results)
    plot_km_curves(km_results, 'reports/km_survival_curves.png')
    
    # 3. Cox PH model
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph(df)
    print_cox_results(cox_results)
    
    # 4. Risk chiizer analysis
    print("\n[4/5] Running risk chiizer analysis...")
    print_chiizer_summary(df)
    
    # 5. Predict for new applicant
    print("\n[5/5] Predicting survival for new applicant...")
    new_applicant = {
        'credit_score': 650,
        'income': 55000,
        'employment_years': 3.5,
        'debt_to_income': 0.32,
        'loan_amount': 180000,
        'interest_rate': 7.5,
        'LTV_ratio': 0.82,
    }
    applicant_result = predict_for_new_applicant(df, new_applicant)
    print_applicant_prediction(applicant_result, new_applicant)
    
    # Compile results for JSON
    results_json = {
        'data_summary': {
            'n_observations': int(len(df)),
            'n_defaults': int(df['event_default'].sum()),
            'n_censored': int((df['event_default']==0).sum()),
            'default_rate': float(df['event_default'].mean()),
        },
        'kaplan_meier': {
            'median_times': km_results['median_times'],
            'survival_probabilities': km_results['survival_probs'],
        },
        'cox_ph': {
            'concordance_index': float(cox_results['model'].concordance_index_),
            'log_likelihood': float(cox_results['model'].log_likelihood_),
            'coefficients': {
                var: {
                    'coef': float(row['coef']),
                    'hazard_ratio': float(row['hazard_ratio']),
                    'p_value': float(row['p']),
                }
                for var, row in cox_results['summary'].iterrows()
            }
        },
        'new_applicant_prediction': {
            'applicant': new_applicant,
            '12_month_survival': float(applicant_result['12_month_survival']),
            '24_month_survival': float(applicant_result['24_month_survival']),
        }
    }
    
    # Save results
    with open('reports/survival_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[Saved] Results to reports/survival_results.json")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nKey Outputs:")
    print(f"  - Kaplan-Meier plot: reports/km_survival_curves.png")
    print(f"  - Results JSON: reports/survival_results.json")
    
    return results_json


if __name__ == '__main__':
    results = run_pipeline()