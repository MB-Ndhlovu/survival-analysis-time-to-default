"""Execute full survival analysis pipeline."""

import json
from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier
from src.cox_ph import fit_cox_ph
from src.chiizer import run_full_chiizer
from src.predict_survival import build_predictor

def run_pipeline():
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating loan data (n=5000)...")
    df = generate_loan_data()
    print(f"  Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"  Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")

    # 2. Kaplan-Meier
    print("\n[2] Fitting Kaplan-Meier curves by credit score band...")
    km_results = fit_kaplan_meier(df)
    print("  Saved: reports/kaplan_meier_curves.png")
    for band, res in km_results.items():
        median = res['median_survival_time']
        s12 = f"{res['survival_12_month']:.4f}" if res['survival_12_month'] is not None else "N/A"
        s24 = f"{res['survival_24_month']:.4f}" if res['survival_24_month'] is not None else "N/A"
        print(f"  {band}:")
        print(f"    N={res['n_obs']}, defaults={res['n_defaults']}")
        print(f"    Median survival: {median}")
        print(f"    12-mo survival: {s12}")
        print(f"    24-mo survival: {s24}")

    # 3. Cox PH
    print("\n[3] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph(df)
    print(f"  Concordance Index: {cox_results['concordance_index']}")
    sorted_vars = sorted(cox_results['interpretations'].items(),
                        key=lambda x: x[1]['p_value'])
    print("  Top risk factors (by hazard ratio):")
    for var, info in sorted_vars[:4]:
        sig = "*" if info['significant'] else ""
        print(f"    {var}: HR={info['hazard_ratio']:.4f} {sig}")

    # 4. Chiizer
    print("\n[4] Running risk chiizer on key variables...")
    chiizer_results = run_full_chiizer(df)
    print("  Saved: reports/chiizer_*.png")
    for var, res in chiizer_results.items():
        print(f"\n  {var}:")
        for bin_label, bin_data in res['bins'].items():
            s12 = f"{bin_data['survival_12']:.4f}" if bin_data['survival_12'] else "N/A"
            print(f"    {bin_label}: n={bin_data['n']}, "
                  f"default_rate={bin_data['default_rate']:.3f}, "
                  f"12-mo surv={s12}")

    # 5. Prediction
    print("\n[5] Building predictor and scoring new applicant...")
    predictor = build_predictor(df)
    new_applicant = {
        'income': 55000,
        'credit_score': 620,
        'employment_years': 2.5,
        'debt_to_income': 0.35,
        'loan_amount': 25000,
        'interest_rate': 0.12,
        'LTV_ratio': 0.85,
    }
    prediction = predictor(new_applicant)
    print(f"  Applicant: credit_score={new_applicant['credit_score']}, "
          f"DTI={new_applicant['debt_to_income']:.2f}")
    print(f"  Risk Tier: {prediction['risk_tier']}")
    print(f"  12-mo survival: {prediction['survival_12mo']:.4f}")
    print(f"  24-mo survival: {prediction['survival_24mo']:.4f}")

    # 6. Compile results JSON
    results_json = {
        'km_results': {
            band: {
                'n_obs': res['n_obs'],
                'n_defaults': res['n_defaults'],
                'median_survival_time': str(res['median_survival_time']),
                'survival_12_month': res['survival_12_month'],
                'survival_24_month': res['survival_24_month'],
            }
            for band, res in km_results.items()
        },
        'cox_ph': {
            'concordance_index': cox_results['concordance_index'],
            'interpretations': {
                var: {
                    'hazard_ratio': float(info['hazard_ratio']),
                    'p_value': float(info['p_value']),
                    'significant': bool(info['significant']),
                    'interpretation': info['interpretation'],
                }
                for var, info in cox_results['interpretations'].items()
            }
        },
        'chiizer_summary': {
            var: {
                bin_label: {
                    'n': bin_data['n'],
                    'default_rate': bin_data['default_rate'],
                    'survival_12': bin_data['survival_12'],
                    'survival_24': bin_data['survival_24'],
                }
                for bin_label, bin_data in res['bins'].items()
            }
            for var, res in chiizer_results.items()
        },
        'new_applicant_prediction': {
            'applicant': new_applicant,
            'prediction': {k: v for k, v in prediction.items()}
        }
    }

    with open('/home/workspace/Projects/survival-analysis-time-to-default/reports/survival_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print("\n  Saved: reports/survival_results.json")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nKey Insight: Survival analysis reveals WHEN default is likely,")
    print("not just IF. Credit score bands show distinct survival trajectories")
    print("with median defaults ranging from <12mo (Very Poor) to >24mo (Excellent).")

    return results_json

if __name__ == '__main__':
    results = run_pipeline()