"""
Run full survival analysis pipeline:
1. Generate loan data
2. Fit Kaplan-Meier curves by credit band
3. Fit Cox PH model
4. Run risk chiizer
5. Predict survival for new applicants
6. Save results to reports/
"""

import json
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier, print_km_summary
from src.cox_ph import fit_cox_ph, print_cox_summary
from src.chiizer import chiize_all_variables, print_chiizer_summary
from src.predict_survival import simulate_applicant_survival, print_applicant_summary


def serialize_for_json(obj):
    """Convert non-serializable objects for JSON output."""
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return None
        return round(obj, 6)
    if hasattr(obj, '__float__'):
        return round(float(obj), 6)
    return str(obj)


def run_pipeline():
    """Execute full survival analysis pipeline."""
    print("=" * 80)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 80)

    # 1. Load/generate data
    print("\n[1/5] Generating synthetic loan data (n=5000)...")
    df = generate_loan_data(n_samples=5000)
    print(f"  -> {len(df)} loans generated")
    print(f"  -> {df['event_default'].eq(1).sum()} defaults observed "
          f"({df['event_default'].eq(1).mean():.1%})")
    print(f"  -> {df['event_default'].eq(0).sum()} censored "
          f"({df['event_default'].eq(0).mean():.1%})")

    # 2. Kaplan-Meier by credit band
    print("\n[2/5] Fitting Kaplan-Meier curves by credit score band...")
    km_results = fit_kaplan_meier(df)
    print_km_summary(km_results)

    # 3. Cox PH model
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph(df)
    print_cox_summary(cox_results)

    # 4. Risk Chiizer
    print("\n[4/5] Running risk chiizer on key variables...")
    chiizer_results = chiize_all_variables(df)
    print_chiizer_summary(chiizer_results)

    # 5. New applicant predictions
    print("\n[5/5] Predicting survival for new applicants...")
    applicant_results = simulate_applicant_survival(df, cox_results['cph'])
    print_applicant_summary(applicant_results)

    # Save results to JSON
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Build serializable results dict
    summary_results = {
        "pipeline": "Time-to-Default Survival Analysis",
        "n_samples": len(df),
        "n_defaults": int(df['event_default'].sum()),
        "censoring_rate": round(df['event_default'].eq(0).mean(), 4),
        "concordance_index": round(cox_results['concordance'], 4),
        "km_curves_by_band": {},
        "cox_hazard_ratios": {},
        "chiizer_summaries": {},
        "applicant_predictions": {}
    }

    # KM by band
    band_order = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]
    for band in band_order:
        if band in km_results:
            r = km_results[band]
            mt = r['median_time']
            if not isinstance(mt, str):
                mt = round(float(mt), 2)
            summary_results["km_curves_by_band"][band] = {
                "n_at_risk": int(r['n_at_risk']),
                "n_defaults": int(r['n_events']),
                "median_time_months": mt,
                "survival_12mo": round(float(r['survival_12mo']), 4),
                "survival_24mo": round(float(r['survival_24mo']), 4)
            }

    # Cox HR
    summary_df = cox_results['summary']
    for var in summary_df.index:
        row = summary_df.loc[var]
        summary_results["cox_hazard_ratios"][var] = {
            "coefficient": round(float(row['coef']), 4),
            "hazard_ratio": round(float(row['hazard_ratio']), 4),
            "hr_lower_95": round(float(row['hr_lower']), 4),
            "hr_upper_95": round(float(row['hr_upper']), 4),
            "p_value": round(float(row['p']), 4)
        }

    # Chiizer
    for var, chi_df in chiizer_results.items():
        rows = []
        for _, row in chi_df.iterrows():
            rows.append({
                "bin": str(row['bin']),
                "n": int(row['n']),
                "n_defaults": int(row['n_defaults']),
                "survival_12mo": round(float(row['survival_12mo']), 4),
                "survival_24mo": round(float(row['survival_24mo']), 4)
            })
        summary_results["chiizer_summaries"][var] = rows

    # Applicant predictions
    for label, data in applicant_results.items():
        profile = data['profile']
        summary_results["applicant_predictions"][label] = {
            "income": profile['income'],
            "credit_score": profile['credit_score'],
            "debt_to_income": round(profile['debt_to_income'], 4),
            "loan_amount": profile['loan_amount'],
            "interest_rate": round(profile['interest_rate'], 4),
            "LTV_ratio": round(profile['LTV_ratio'], 4),
            "survival_12mo": round(float(data['survival_12mo']), 4),
            "survival_24mo": round(float(data['survival_24mo']), 4)
        }

    # Write JSON
    os.makedirs("reports", exist_ok=True)
    with open("reports/survival_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    print("  -> reports/survival_results.json saved")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print("  - reports/km_survival_curves.png     (Kaplan-Meier by credit band)")
    print("  - reports/chiizer_*.png              (Risk chiizer plots)")
    print("  - reports/*_risk.png                 (Applicant survival curves)")
    print("  - reports/survival_results.json     (Full results JSON)")

    return summary_results


if __name__ == "__main__":
    run_pipeline()