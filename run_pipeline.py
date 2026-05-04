"""
Run full survival analysis pipeline.
Executes all analysis components and generates outputs.
"""

import json
import os
import sys
import math

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import generate_loan_data, add_credit_bands
from src.chiizer import chiize_all, build_chiizer_matrix, print_chiizer_summary
from src.kaplan_meier import (
    fit_km_by_credit_band, compute_median_survival_times,
    compute_survival_probabilities, plot_km_curves, print_km_summary
)
from src.cox_ph import fit_cox_model, extract_coefficients, print_cox_summary, plot_cox_hazard_ratios
from src.predict_survival import (
    create_applicant_profile, predict_survival_for_new_applicant,
    print_applicant_assessment, plot_applicant_survival
)


def save_results(results: dict, path: str):
    """Save results dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   Results saved to {path}")


def run_pipeline():
    """Execute full survival analysis pipeline."""
    print("\n" + "="*70)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("="*70)

    # ========================================
    # STEP 1: Load / Generate Data
    # ========================================
    print("\n[1/6] Generating loan data...")
    df = add_credit_bands(generate_loan_data(n_samples=5000))
    print(f"   Generated {len(df)} loan records")
    print(f"   Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"   Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")

    # Apply chiizer
    df = chiize_all(df)
    print("   Applied risk chiizer bins")

    # ========================================
    # STEP 2: Kaplan-Meier Analysis
    # ========================================
    print("\n[2/6] Running Kaplan-Meier analysis...")
    km_fitters = fit_km_by_credit_band(df)
    km_medians = compute_median_survival_times(km_fitters)
    prob_df = print_km_summary(km_fitters, km_medians)

    # Plot KM curves
    km_fig = plot_km_curves(km_fitters, save_path='reports/km_curves.png')
    print("   Kaplan-Meier curves saved to reports/km_curves.png")

    # ========================================
    # STEP 3: Chiizer Analysis
    # ========================================
    print("\n[3/6] Running risk chiizer analysis...")
    chiizer_matrix = build_chiizer_matrix(df, ['income', 'debt_to_income', 'loan_amount'])
    print_chiizer_summary(chiizer_matrix)

    # ========================================
    # STEP 4: Cox PH Model
    # ========================================
    print("\n[4/6] Fitting Cox Proportional Hazards model...")
    cph = fit_cox_model(df)
    cox_summary = extract_coefficients(cph)
    print_cox_summary(cph, cox_summary)

    # Plot hazard ratios
    hr_fig = plot_cox_hazard_ratios(cox_summary, save_path='reports/cox_hazard_ratios.png')
    print("   Cox PH hazard ratios saved to reports/cox_hazard_ratios.png")

    # ========================================
    # STEP 5: New Applicant Prediction
    # ========================================
    print("\n[5/6] Predicting survival for new applicant...")

    # Create example applicant
    new_applicant = create_applicant_profile(
        credit_score=710,
        income=78000,
        employment_years=6.2,
        debt_to_income=0.28,
        loan_amount=42000,
        interest_rate=0.095,
        LTV_ratio=0.82
    )

    applicant_results = predict_survival_for_new_applicant(cph, new_applicant)
    print_applicant_assessment(new_applicant, applicant_results)

    # Plot applicant survival curve
    app_fig = plot_applicant_survival(cph, new_applicant, df, save_path='reports/applicant_survival.png')
    print("   Applicant survival curve saved to reports/applicant_survival.png")

    # ========================================
    # STEP 6: Save Results to JSON
    # ========================================
    print("\n[6/6] Saving results to JSON...")

    # Build results dictionary
    results = {
        'data_summary': {
            'n_samples': len(df),
            'n_defaults': int(df['event_default'].sum()),
            'n_censored': int((df['event_default']==0).sum()),
            'default_rate': float(df['event_default'].mean()),
            'censoring_rate': float((df['event_default']==0).mean())
        },
        'median_survival_by_credit_band': {
            band: float(median) if not math.isnan(median) else None
            for band, median in km_medians.items()
        },
        'survival_probabilities': {
            band: {
                '12_month': float(prob_df.loc[band, '12-month']),
                '24_month': float(prob_df.loc[band, '24-month']),
                '36_month': float(prob_df.loc[band, '36-month']),
                '48_month': float(prob_df.loc[band, '48-month'])
            }
            for band in prob_df.index
        },
        'cox_ph_coefficients': {
            var: {
                'coefficient': float(cox_summary.loc[var, 'Coefficient']),
                'hazard_ratio': float(cox_summary.loc[var, 'Hazard Ratio']),
                'hr_95_ci_lower': float(cox_summary.loc[var, 'HR 95% CI Lower']),
                'hr_95_ci_upper': float(cox_summary.loc[var, 'HR 95% CI Upper']),
                'p_value': float(cox_summary.loc[var, 'P-value'])
            }
            for var in cox_summary.index
        },
        'new_applicant_prediction': {
            'profile': {
                col: float(new_applicant.iloc[0][col])
                for col in new_applicant.columns
            },
            'hazard_ratio': float(applicant_results['hazard_ratio']),
            'survival_probabilities': {
                '12_month': float(applicant_results['12_month_survival']),
                '24_month': float(applicant_results['24_month_survival']),
                '36_month': float(applicant_results['36_month_survival']),
                '48_month': float(applicant_results['48_month_survival'])
            }
        },
        'chiizer_summary': chiizer_matrix.to_dict('records')
    }

    save_results(results, 'reports/survival_results.json')

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\n📁 Output files:")
    print("   - reports/km_curves.png")
    print("   - reports/cox_hazard_ratios.png")
    print("   - reports/applicant_survival.png")
    print("   - reports/survival_results.json")

    print("\n💡 Key Business Insights:")
    print("   - Survival analysis reveals WHEN default occurs, not just IF")
    print("   - Credit score bands show clear separation in survival curves")
    print("   - Cox PH identifies the biggest risk drivers in your portfolio")
    print("   - Individual predictions enable risk-based pricing")

    return results


if __name__ == '__main__':
    results = run_pipeline()