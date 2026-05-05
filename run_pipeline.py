import json
import os
from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier, plot_km_curves
from src.cox_ph import fit_cox_ph, print_cox_results
from src.chiizer import build_risk_chiizer, print_chiizer_results
from src.predict_survival import predict_survival, print_prediction

def run_pipeline():
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # Ensure output dirs exist
    os.makedirs('reports', exist_ok=True)

    # 1. Load/generate data
    print("\n[1/6] Generating loan data...")
    df = generate_loan_data(n=5000, seed=42)
    print(f"  Generated {len(df)} loans")
    print(f"  Default rate: {df['event_default'].mean():.2%}")
    print(f"  Censored (no default in 24m): {(df['event_default'] == 0).mean():.2%}")

    # 2. Kaplan-Meier analysis
    print("\n[2/6] Running Kaplan-Meier analysis...")
    km_results, kmf, df_labeled = fit_kaplan_meier(df)
    plot_path = plot_km_curves(kmf, df_labeled, output_path='reports/km_survival_curves.png')

    print("\n  Kaplan-Meier Results by Credit Band:")
    for band, res in km_results.items():
        median_str = f"{res['median_survival']:.1f}m" if res['median_survival'] else "N/A (50% not defaulted)"
        print(f"  {band}: n={res['n']}, median survival={median_str}, 12m={res['survival_at_12']:.3f}, 24m={res['survival_at_24']:.3f}")

    # 3. Cox PH model
    print("\n[3/6] Fitting Cox Proportional Hazards model...")
    cox_results, cph = fit_cox_ph(df)
    coef_df = print_cox_results(cox_results)

    # 4. Risk Chiizer
    print("\n[4/6] Building risk chiizer...")
    chiizer_results = build_risk_chiizer(df)
    print_chiizer_results(chiizer_results)

    # 5. New applicant prediction
    print("\n[5/6] Predicting survival for new applicant...")
    new_applicant = {
        'income': 65000,
        'credit_score': 720,
        'employment_years': 7,
        'debt_to_income': 0.28,
        'loan_amount': 180000,
        'interest_rate': 7.5,
        'ltv_ratio': 0.75
    }
    prediction = predict_survival(new_applicant, cph, df)
    print_prediction(prediction, new_applicant)

    # 6. Compile and save results
    print("\n[6/6] Saving results to reports/survival_results.json...")

    results_json = {
        'km_results': {
            band: {
                'n': res['n'],
                'median_survival': res['median_survival'],
                'survival_at_12m': res['survival_at_12'],
                'survival_at_24m': res['survival_at_24']
            } for band, res in km_results.items()
        },
        'cox_ph': {
            'concordance_index': cox_results['concordance_index'],
            'coefficients': {
                var: {
                    'coefficient': float(c['coefficient']),
                    'hazard_ratio': float(c['hazard_ratio']),
                    'p_value': float(c['p_value']),
                    'significant': c['significant']
                } for var, c in cox_results['coefficients'].items()
            }
        },
        'chiizer': chiizer_results,
        'new_applicant_prediction': {
            'applicant': new_applicant,
            'survival_12m': prediction['survival_12m'],
            'survival_24m': prediction['survival_24m'],
            'median_survival_months': prediction['median_survival_months'],
            'similar_profiles_n': prediction['similar_profiles_n']
        },
        'summary': {
            'total_loans': len(df),
            'default_rate': float(df['event_default'].mean()),
            'top_default_risk_factors': list(coef_df.head(3).index) if len(coef_df) >= 3 else list(coef_df.index)
        }
    }

    with open('reports/survival_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved:")
    print(f"  - reports/km_survival_curves.png")
    print(f"  - reports/survival_results.json")
    print("\nKey insight: Survival analysis reveals WHEN default is likely,")
    print("not just IF — enabling better pricing and provisioning.")

    return results_json

if __name__ == "__main__":
    results = run_pipeline()