"""
Execute full survival analysis pipeline:
1. Load/generate data
2. Kaplan-Meier by credit band
3. Cox PH hazard ratios
4. Risk Chiizer
5. New applicant prediction
6. Save results to JSON and plot to PNG
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_km_by_credit_band, print_km_summary
from src.cox_ph import fit_cox_ph, print_cox_summary
from src.chiizer import chiize, print_chiizer_summary
from src.predict_survival import predict_survival, print_applicant_prediction


def run_pipeline() -> dict:
    print("\n" + "=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating loan data (n=5000)...")
    df = generate_loan_data(n=5000, seed=42)
    censor_pct = (df["event_default"] == 0).mean()
    print(f"      Generated {len(df)} rows | Censored: {censor_pct:.1%}")

    # 2. Kaplan-Meier
    print("\n[2/5] Fitting Kaplan-Meier curves by credit band...")
    km_results = fit_km_by_credit_band(df)
    print(print_km_summary(km_results))

    # 3. Cox PH
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph(df)
    print(print_cox_summary(cox_results))

    # 4. Risk Chiizer
    print("\n[4/5] Running Risk Chiizer...")
    chiizer_results = chiize(df)
    print(print_chiizer_summary(chiizer_results))

    # 5. New applicant prediction
    print("\n[5/5] Predicting survival for new applicant...")
    new_applicant = {
        "credit_score": 720,
        "income": 65000,
        "employment_years": 4.5,
        "debt_to_income": 0.28,
        "loan_amount": 25000,
        "interest_rate": 0.11,
        "LTV_ratio": 0.75,
    }
    surv_df = predict_survival(df, new_applicant)
    print(print_applicant_prediction(new_applicant, surv_df))

    # Collect summary outputs for JSON
    summary = {
        "pipeline": "Time-to-Default Survival Analysis",
        "n_samples": len(df),
        "censored_pct": round(censor_pct, 4),
        "km_by_credit_band": {},
        "cox_ph": {
            "concordance_index": round(cox_results["concordance_index"], 4),
            "log_likelihood": round(cox_results["log_likelihood"], 4),
            "hazard_ratios": {
                k: round(v["hazard_ratio"], 4)
                for k, v in cox_results["coefficients"].items()
            },
            "top_risk_factors": [
                {"factor": k, "hr": round(v["hazard_ratio"], 4)}
                for k, v in sorted(
                    cox_results["coefficients"].items(),
                    key=lambda x: x[1]["hazard_ratio"],
                    reverse=True,
                )[:3]
            ],
        },
        "chiizer": chiizer_results,
        "new_applicant_prediction": {
            "applicant": new_applicant,
            "survival_12m": float(surv_df[surv_df["timeline"] == 12]["survival_probability"].values[0]),
            "survival_24m": float(surv_df[surv_df["timeline"] == 24]["survival_probability"].values[0]),
        },
    }

    for band, res in km_results.items():
        if band.startswith("_"):
            continue
        summary["km_by_credit_band"][band] = res

    # Save JSON
    output_path = Path("reports/survival_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OUTPUT] Results saved to {output_path}")
    print(f"[OUTPUT] KM plot saved to reports/km_survival_curves.png")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = run_pipeline()