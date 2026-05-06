import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier
from src.cox_ph import fit_cox_ph
from src.chiizer import build_risk_chiizer
from src.predict_survival import predict_survival, default_applicant

def run_pipeline():
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating loan data (5000 records)...")
    df = generate_loan_data(n=5000, seed=42)
    defaults = df["event_default"].sum()
    censored = len(df) - defaults
    print(f"    Defaults: {defaults} | Censored: {censored} ({censored/len(df)*100:.1f}%)")

    # 2. Kaplan-Meier
    print("\n[2] Fitting Kaplan-Meier curves...")
    km_result = fit_kaplan_meier(df)
    print("\n    Median Survival Time by Credit Band:")
    for band, med in km_result["median_survival"].items():
        print(f"      {band}: {med if med else 'Not reached'} months")

    print("\n    12/24-month Survival Probabilities:")
    for band, probs in km_result["survival_probs_12_24"].items():
        s12 = probs.get(12, "N/A")
        s24 = probs.get(24, "N/A")
        s12_str = f"{s12:.3f}" if s12 is not None else "N/A"
        s24_str = f"{s24:.3f}" if s24 is not None else "N/A"
        print(f"      {band}: 12mo={s12_str}, 24mo={s24_str}")

    # 3. Cox PH
    print("\n[3] Fitting Cox Proportional Hazards model...")
    cox_result = fit_cox_ph(df)
    print(f"\n    Concordance Index: {cox_result['concordance_index']:.4f}")

    print("\n    Top Hazard Ratios (factors most increasing default risk):")
    hr_df = cox_result["summary"][["hazard_ratio", "p"]].sort_values("hazard_ratio", ascending=False)
    for idx, row in hr_df.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"      {idx:20s}: HR={row['hazard_ratio']:.3f}{sig}")

    # 4. Chiizer
    print("\n[4] Building Risk Chiizer (binned survival curves)...")
    chiizer_result = build_risk_chiizer(df)

    # 5. New applicant prediction
    print("\n[5] Predicting survival for new applicant...")
    applicant = default_applicant()
    print(f"    Credit Score: {applicant['credit_score']}")
    print(f"    Income: ${applicant['income']:,.0f}")
    print(f"    DTI: {applicant['debt_to_income']*100:.1f}%")
    print(f"    LTV: {applicant['LTV_ratio']*100:.1f}%")

    pred = predict_survival(applicant, cox_result["cph"])
    s12 = pred["survival_prob"][12]
    s24 = pred["survival_prob"][24]
    print(f"\n    Predicted 12-month survival: {s12:.3f}")
    print(f"    Predicted 24-month survival: {s24:.3f}")

    # 6. Save results
    print("\n[6] Saving results to reports/survival_results.json...")
    results = {
        "n_records": len(df),
        "n_defaults": int(defaults),
        "n_censored": int(censored),
        "median_survival_months": {str(k): float(v) if v is not None else None for k, v in km_result["median_survival"].items()},
        "survival_probabilities": {
            band: {f"{t}mo": float(p) if p is not None else None for t, p in probs.items()}
            for band, probs in km_result["survival_probs_12_24"].items()
        },
        "cox_hazard_ratios": {str(k): float(v) for k, v in cox_result["summary"]["hazard_ratio"].items()},
        "cox_pvalues": {str(k): float(v) for k, v in cox_result["summary"]["p"].items()},
        "concordance_index": float(cox_result["concordance_index"]),
        "new_applicant_prediction": {
            "applicant": applicant,
            "survival_curve": {
                "timeline": pred["timeline"],
                "probabilities": [round(p, 4) for p in pred["survival_prob"]]
            }
        }
    }

    import os
    os.makedirs("/home/workspace/Projects/survival-analysis-time-to-default/reports", exist_ok=True)
    with open("/home/workspace/Projects/survival-analysis-time-to-default/reports/survival_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("    Saved: reports/survival_results.json")

    # 7. Summary print
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Records: {len(df)}")
    print(f"  Defaults: {defaults} | Censored: {censored}")
    print(f"  Cox PH Concordance Index: {cox_result['concordance_index']:.4f}")
    print(f"  Kaplan-Meier plot: reports/km_survival_curves.png")
    print(f"  Chiizer plot: reports/chiizer_curves.png")
    print(f"  Results JSON: reports/survival_results.json")

    return results

if __name__ == "__main__":
    run_pipeline()