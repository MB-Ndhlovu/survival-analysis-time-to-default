"""
Execute full survival analysis pipeline.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_loan_data, get_credit_score_band
from src.kaplan_meier import fit_km_by_band, plot_km_curves
from src.cox_ph import fit_cox_ph, get_hazard_ratios
from src.chiizer import run_full_chiizer
from src.predict_survival import predict_survival_for_applicant


def run_pipeline():
    """Execute the complete survival analysis pipeline."""
    print("=" * 60)
    print("PROJECT 6: TIME-TO-DEFAULT SURVIVAL ANALYSIS")
    print("=" * 60)

    # ── Step 1: Load / Generate Data ─────────────────────────────────
    print("\n[1/5] Generating loan portfolio data...")
    df = generate_loan_data(n=5000, seed=42)
    print(f"  → {len(df)} loans generated")
    print(f"  → Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"  → Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")

    # ── Step 2: Kaplan-Meier Curves ──────────────────────────────────
    print("\n[2/5] Fitting Kaplan-Meier curves by credit score band...")
    fitters, km_results = fit_km_by_band(df)

    # Plot KM curves
    plot_save_path = os.path.join(os.path.dirname(__file__), 'reports', 'km_survival_curves.png')
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plot_km_curves(fitters, save_path=plot_save_path)

    print("\n  Kaplan-Meier Results:")
    for band, res in km_results.items():
        s12 = None
        s24 = None
        try:
            s12 = fitters[band].survival_function_.iloc[min(11, len(fitters[band].survival_function_)-1)].values[0]
            s24 = fitters[band].survival_function_.iloc[min(23, len(fitters[band].survival_function_)-1)].values[0]
        except:
            pass
        print(f"  {band}:")
        print(f"    N={res['n']}, Events={res['events']}, Median={res['median_survival_months']}m")
        print(f"    S(12m)={s12:.2%}, S(24m)={s24:.2%}")

    # ── Step 3: Cox PH Model ─────────────────────────────────────────
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cph = fit_cox_ph(df)
    hr_df = get_hazard_ratios(cph)

    print("\n  Cox PH Results:")
    print("  Top default risk factors (by hazard ratio):")
    sorted_hr = hr_df.sort_values('hazard_ratio', ascending=False)
    for var, row in sorted_hr.head(5).iterrows():
        direction = "↑ increases" if row['hazard_ratio'] > 1 else "↓ decreases"
        print(f"    {var}: HR={row['hazard_ratio']:.4f} {direction} risk (p={row['p_value']:.4f})")

    # ── Step 4: Risk Chiizer ────────────────────────────────────────
    print("\n[4/5] Running risk chiizer (continuous variable binning)...")
    chi_fitters, chi_results = run_full_chiizer(df)

    # ── Step 5: New Applicant Prediction ─────────────────────────────
    print("\n[5/5] Predicting survival for new applicant...")
    applicant = {
        'income': 550000,
        'credit_score': 660,
        'employment_years': 3,
        'debt_to_income': 0.35,
        'loan_amount': 800000,
        'interest_rate': 0.14,
        'LTV_ratio': 0.75
    }

    survival_df = predict_survival_for_applicant(cph, applicant)
    s12 = survival_df[survival_df['month'] == 12]['survival_probability'].values[0]
    s24 = survival_df[survival_df['month'] == 24]['survival_probability'].values[0]

    print(f"\n  Applicant: Credit Score={applicant['credit_score']}, DTI={applicant['debt_to_income']}")
    print(f"  Predicted S(12m)={s12:.2%}, S(24m)={s24:.2%}")
    print(f"  Implied 12-month default: {1-s12:.2%}")
    print(f"  Implied 24-month default: {1-s24:.2%}")

    # ── Compile Results JSON ─────────────────────────────────────────
    print("\n[+] Saving results to reports/survival_results.json...")
    results = {
        'portfolio': {
            'n_loans': len(df),
            'n_defaults': int(df['event_default'].sum()),
            'default_rate': round(float(df['event_default'].mean()), 4),
            'censored_count': int((df['event_default'] == 0).sum()),
            'censored_rate': round(float((df['event_default'] == 0).mean()), 4)
        },
        'kaplan_meier': {},
        'cox_ph': {
            'hazard_ratios': hr_df.to_dict(),
            'concordance_index': round(float(cph.concordance_index_), 4)
        },
        'new_applicant': {
            'profile': applicant,
            'survival_12m': round(float(s12), 4),
            'survival_24m': round(float(s24), 4),
            'default_prob_12m': round(float(1 - s12), 4),
            'default_prob_24m': round(float(1 - s24), 4)
        }
    }

    # Add KM results by band
    for band, res in km_results.items():
        s12_val = None
        s24_val = None
        try:
            s12_val = round(float(fitters[band].survival_function_.iloc[11].values[0]), 4)
            s24_val = round(float(fitters[band].survival_function_.iloc[23].values[0]), 4)
        except:
            pass

        results['kaplan_meier'][band] = {
            'n': res['n'],
            'events': res['events'],
            'median_survival_months': res['median_survival_months'],
            'survival_12m': s12_val,
            'survival_24m': s24_val
        }

    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    results_path = os.path.join(reports_dir, 'survival_results.json')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  → Results saved to {results_path}")

    # ── Summary Output ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"""
Key Findings:
• Median time to default varies significantly by credit score band
• Cox PH identifies which factors drive default risk most
• Survival analysis reveals WHEN default is likely, not just IF

Files Generated:
• reports/km_survival_curves.png — Kaplan-Meier plot
• reports/survival_results.json — Full results JSON

Business Insight:
Survival analysis gives more information than binary default models —
it tells you WHEN default is likely, enabling better risk pricing,
capital reserving, and early warning systems.
""")

    return results


if __name__ == '__main__':
    results = run_pipeline()