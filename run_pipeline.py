"""Full pipeline: load data, run survival analysis, print results."""

import json
import sys

# Add src to path for imports
sys.path.insert(0, ".")

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_km_by_credit_band, plot_km_curves, get_results_text
from src.cox_ph import fit_cox_ph_model, print_cox_results, get_top_risk_factors
from src.chiizer import build_risk_chiizer, print_chiizer_results
from src.predict_survival import predict_survival_for_applicant, print_prediction_results


def run_pipeline():
    """Execute full survival analysis pipeline."""
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating synthetic loan data...")
    df = generate_loan_data(n_samples=5000, seed=42)
    print(f"    Generated {len(df)} records")
    print(f"    Default rate: {df['event_default'].mean():.1%}")
    print(f"    Censored rate: {(df['event_default'] == 0).mean():.1%}")

    # 2. Kaplan-Meier analysis
    print("\n[2] Fitting Kaplan-Meier curves by credit score band...")
    km_results = fit_km_by_credit_band(df)
    km_text = get_results_text(km_results)
    print(km_text)
    plot_km_curves(km_results["kmfitters"], "reports/km_curves.png")

    # 3. Cox PH model
    print("\n[3] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph_model(df)
    cox_text = print_cox_results(cox_results)
    print(cox_text)

    # 4. Risk chiizer
    print("\n[4] Building risk chiizer...")
    chiizer = build_risk_chiizer(df)
    chiizer_text = print_chiizer_results(chiizer)
    print(chiizer_text)

    # 5. Prediction for new applicant
    print("\n[5] Predicting survival for new applicant...")
    new_applicant = {
        "credit_score": 650,
        "income": 55.0,
        "employment_years": 3.5,
        "debt_to_income": 0.40,
        "loan_amount": 75.0,
        "interest_rate": 8.5,
        "LTV_ratio": 0.85,
    }
    prediction = predict_survival_for_applicant(cox_results["cph"], new_applicant)
    pred_text = print_prediction_results(prediction, new_applicant)
    print(pred_text)

    # 6. Compile results JSON
    print("\n[6] Saving results to reports/survival_results.json...")
    results_json = {
        "data_summary": {
            "n_samples": len(df),
            "default_rate": float(df["event_default"].mean()),
            "censored_rate": float((df["event_default"] == 0).mean()),
        },
        "kaplan_meier": {
            "credit_bands": {},
        },
        "cox_ph": {
            "concordance_index": float(cox_results["concordance_index"]),
            "coefficients": {},
        },
        "chiizer": {},
        "new_applicant_prediction": {
            "applicant": new_applicant,
            "survival_at_12_months": float(prediction["survival_probabilities"][11]),
            "survival_at_24_months": float(prediction["survival_probabilities"][23]),
            "survival_at_36_months": float(prediction["survival_probabilities"][35]),
        },
    }

    # Fill in KM results
    for band in ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]:
        if band in km_results["medians"]:
            results_json["kaplan_meier"]["credit_bands"][band] = {
                "median_survival_months": km_results["medians"][band],
                "survival_12_months": km_results["survival_probs"][band][12],
                "survival_24_months": km_results["survival_probs"][band][24],
            }

    # Fill in Cox PH coefficients
    summary = cox_results["summary"]
    for var in summary.index:
        results_json["cox_ph"]["coefficients"][var] = {
            "coef": float(summary.loc[var, "coef"]),
            "hazard_ratio": float(summary.loc[var, "hazard_ratio"]),
            "p_value": float(summary.loc[var, "p"]),
        }

    # Fill in chiizer results
    for var in ["credit_score", "debt_to_income", "LTV_ratio"]:
        results_json["chiizer"][var] = {}
        if var in chiizer:
            for seg, stats in chiizer[var].items():
                results_json["chiizer"][var][seg] = {
                    "n": stats["n"],
                    "n_events": stats["n_events"],
                    "survival_12": stats["survival_12"],
                    "survival_24": stats["survival_24"],
                    "median_survival": stats["median_survival"],
                }

    with open("reports/survival_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("    Saved to reports/survival_results.json")

    # 7. Business insight summary
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)
    print("""
Survival analysis gives more information than binary default models:
- Traditional models: "Will this borrower default? Yes/No"
- Survival models: "When is default most likely, and what's the probability over time?"

Key findings from this analysis:
""")

    # Top risk factors
    top_risks = get_top_risk_factors(summary, n=3)
    if top_risks:
        print("Top risk factors increasing time-to-default hazard:")
        for var, hr, direction, p in top_risks:
            print(f"  - {var}: HR={hr:.3f} ({direction} default risk, p={p:.4f})")
        print()

    # Credit band comparison
    print("Credit score band comparison (12-month survival):")
    for band in ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]:
        if band in km_results["survival_probs"]:
            s12 = km_results["survival_probs"][band][12]
            print(f"  - {band}: {s12:.1%} survive 12 months without default")
    print()
    print("=" * 60)

    return results_json


if __name__ == "__main__":
    results = run_pipeline()