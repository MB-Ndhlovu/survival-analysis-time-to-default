"""Execute full survival analysis pipeline."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier
from src.cox_ph import fit_cox_ph
from src.chiizer import build_risk_chiizer
from src.predict_survival import predict_survival_for_applicant


def run_pipeline():
    print("=" * 70)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 70)

    # Ensure output directories exist
    os.makedirs("reports", exist_ok=True)

    # Step 1: Load/generate data
    print("\n[1/5] Generating loan data...")
    df = generate_loan_data(n=5000, seed=42)
    n_censored = (df["event_default"] == 0).sum()
    n_defaults = (df["event_default"] == 1).sum()
    print(f"  Generated {len(df)} loans")
    print(f"  Defaults: {n_defaults} ({n_defaults/len(df)*100:.1f}%)")
    print(f"  Censored: {n_censored} ({n_censored/len(df)*100:.1f}%)")

    # Step 2: Kaplan-Meier curves
    print("\n[2/5] Fitting Kaplan-Meier curves by credit band...")
    km_results = fit_kaplan_meier(df, output_path="reports/km_curves.png")
    print("\n  Kaplan-Meier Results by Credit Band:")
    for band, metrics in km_results.items():
        med = metrics["median_survival_time"]
        med_str = f"{med} months" if med else "> 24 months (not reached)"
        print(f"  {band}: median={med_str}, 12m={metrics['survival_12m']:.1%}, 24m={metrics['survival_24m']:.1%}")

    # Step 3: Cox PH model
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph(df)

    # Step 4: Risk chiizer
    print("\n[4/5] Building risk chiizer (survival curves by binned factors)...")
    chiizer_results = build_risk_chiizer(df, output_path="reports/risk_chiizer.png")
    print("\n  Log-rank p-values (significance of factor effect):")
    for var, res in chiizer_results.items():
        sig = "SIGNIFICANT" if res["significant"] else "not significant"
        p_val = res["logrank_p_value"]
        p_str = f"{p_val:.2e}" if p_val and p_val < 0.001 else f"{p_val:.4f}" if p_val else "N/A"
        print(f"  {var}: p={p_str} ({sig})")

    # Step 5: New applicant prediction
    print("\n[5/5] Predicting survival for new applicant...")
    new_applicant = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.22,
        "loan_amount": 80000,
        "interest_rate": 6.5,
        "LTV_ratio": 0.75,
    }
    pred_results = predict_survival_for_applicant(df, new_applicant, output_path="reports/applicant_survival.png")
    print(f"  Applicant credit score: {new_applicant['credit_score']}")
    print(f"  Median survival: {pred_results['median_survival_months']} months")
    print(f"  12m survival: {pred_results['survival_12m']:.1%}")
    print(f"  24m survival: {pred_results['survival_24m']:.1%}")
    print(f"  36m survival: {pred_results['survival_36m']:.1%}")

    # Compile results for JSON output
    results = {
        "data_summary": {
            "n_total": len(df),
            "n_defaults": int(n_defaults),
            "n_censored": int(n_censored),
            "censoring_rate": round(n_censored / len(df), 4),
        },
        "kaplan_meier_by_band": km_results,
        "cox_ph_model": {
            "concordance_index": round(cox_results["concordance_index"], 4),
            "log_likelihood": round(cox_results["log_likelihood"], 2),
            "hazard_ratios": cox_results["hazard_ratios"],
            "coefficients": cox_results["coefficients"],
            "p_values": cox_results["p_values"],
        },
        "risk_chiizer": chiizer_results,
        "new_applicant_prediction": pred_results,
    }

    # Save results
    with open("reports/survival_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nOutputs saved:")
    print("  reports/km_curves.png        - Kaplan-Meier curves by credit band")
    print("  reports/risk_chiizer.png     - Survival curves by binned factors")
    print("  reports/applicant_survival.png - New applicant prediction curve")
    print("  reports/survival_results.json - Full results in JSON format")
    print("\nKey Business Insight:")
    print("  Survival analysis reveals WHEN default is likely,")
    print("  not just IF. Credit score bands show clear separation")
    print("  in time-to-default curves, enabling more precise risk pricing.")

    return results


if __name__ == "__main__":
    results = run_pipeline()