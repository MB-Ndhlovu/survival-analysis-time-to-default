"""Execute the full survival analysis pipeline."""

import json
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import generate_loan_data
from kaplan_meier import fit_km
from cox_ph import fit_cox
from chiizer import run_chiizer
from predict_survival import demo_prediction


def main():
    os.makedirs("reports", exist_ok=True)

    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating loan data (n=5000)...")
    data = generate_loan_data(5000)
    print(f"    Defaults: {data['event_default'].sum()} ({data['event_default'].mean():.1%})")
    print(f"    Censored: {(data['event_default']==0).sum()} ({(data['event_default']==0).mean():.1%})")

    # 2. Kaplan-Meier
    print("\n[2/5] Fitting Kaplan-Meier curves...")
    km_results = fit_km(data)
    print("    Median survival by band:")
    for band, res in km_results.items():
        print(f"      {band}: {res['median_survival_months']} months "
              f"(12m survival: {res['survival_12m']}, 24m: {res['survival_24m']})")

    # 3. Cox PH
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cph, cox_results = fit_cox(data)
    print("    Concordance index:", cox_results["concordance_index"])
    print("    Top hazard drivers:")
    sorted_coef = sorted(cox_results["coefficients"].items(),
                         key=lambda x: x[1]["hazard_ratio"], reverse=True)
    for var, info in sorted_coef[:3]:
        print(f"      {var}: HR={info['hazard_ratio']:.4f} (p={info['p_value']:.4f})")

    # 4. Risk Chiizer
    print("\n[4/5] Running risk chiizer...")
    chi_results = run_chiizer(data)

    # 5. Demo predictions
    print("\n[5/5] Running new applicant predictions...")
    pred_results = demo_prediction(cph, data)

    # Compile results
    output = {
        "km_results": km_results,
        "cox_results": {
            "concordance_index": cox_results["concordance_index"],
            "coefficients": {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in cox_results["coefficients"].items()},
        },
        "chiizer_results": chi_results,
        "prediction_results": {k: {kk: float(vv) if isinstance(vv, (float, int)) else vv
                                   for kk, vv in v.items()}
                               for k, v in pred_results.items()},
        "data_summary": {
            "n": int(len(data)),
            "defaults": int(data["event_default"].sum()),
            "censored": int((data["event_default"] == 0).sum()),
        },
    }

    out_path = "reports/survival_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    summary = (
        f"n=5000 | Defaults: {data['event_default'].sum()} | "
        f"Censored: {(data['event_default']==0).sum()}\n"
        f"Concordance: {cox_results['concordance_index']}\n"
        f"KM plot saved to reports/km_survival_curves.png"
    )
    print(summary)
    return output


if __name__ == "__main__":
    main()