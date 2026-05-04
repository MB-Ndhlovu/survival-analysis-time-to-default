import json
import sys
import numpy as np
from pathlib import Path

from src.data_loader import generate_loan_data, get_credit_score_band
from src.kaplan_meier import fit_kaplan_meier, plot_kaplan_meier, compute_band_statistics
from src.cox_ph import fit_cox_ph, print_cox_results
from src.chiizer import compute_risk_bands, plot_risk_bands, get_risk_summary
from src.predict_survival import predict_survival, print_prediction, create_applicant


def main():
    project_root = Path(__file__).parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1/5] Generating loan data...")
    df = generate_loan_data(n_samples=5000)
    df["credit_score_band"] = df["credit_score"].apply(get_credit_score_band)
    print(f"  Generated {len(df)} loans")
    print(f"  Default rate: {df['event_default'].mean():.2%}")
    print(f"  Censored rate: {(df['event_default'] == 0).mean():.2%}")

    print("\n[2/5] Fitting Kaplan-Meier curves...")
    km_results = fit_kaplan_meier(df)
    print(f"  Overall median survival: {km_results['median_survival']:.1f} months")

    band_stats = compute_band_statistics(df, km_results)
    print("  Median survival by credit band:")
    for stat in band_stats:
        median = f"{stat['median_survival_months']:.1f}m" if stat["median_survival_months"] else "Not reached"
        print(f"    {stat['band']}: {median} | 12m: {stat['survival_12m']:.3f} | 24m: {stat['survival_24m']:.3f}")

    km_plot_path = reports_dir / "kaplan_meier_curves.png"
    plot_kaplan_meier(df, km_results, str(km_plot_path))
    print(f"  Kaplan-Meier plot saved to {km_plot_path}")

    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cox_results = fit_cox_ph(df)
    print_cox_results(cox_results)

    print("\n[4/5] Computing risk chiizer bands...")
    risk_bands = compute_risk_bands(df)
    risk_plot_path = reports_dir / "risk_bands.png"
    plot_risk_bands(df, risk_bands, str(risk_plot_path))
    print(f"  Risk bands plot saved to {risk_plot_path}")

    print("\n[5/5] Predicting survival for new applicant...")
    applicant = create_applicant()
    prediction = predict_survival(applicant, cox_results["cph"], df)
    print_prediction(prediction)

    results_json = {
        "data_summary": {
            "n_loans": int(len(df)),
            "default_rate": round(float(df["event_default"].mean()), 4),
            "censored_rate": round(float((df["event_default"] == 0).mean()), 4),
        },
        "km_overall_median_survival": (
            float(km_results["median_survival"]) if not np.isnan(km_results["median_survival"]) else None
        ),
        "km_band_statistics": band_stats,
        "cox_hazard_ratios": {
            k: {
                "hazard_ratio": v["hazard_ratio"],
                "coefficient": v["coefficient"],
                "p_value": v["p_value"],
                "significant": v["significant"],
            }
            for k, v in cox_results["hazard_ratios"].items()
        },
        "cox_concordance_index": cox_results["concordance_index"],
        "risk_band_summaries": get_risk_summary(risk_bands),
        "new_applicant_prediction": {
            "profile": prediction["applicant_data"],
            "survival_probabilities": prediction["survival_at_time"],
        },
    }

    json_path = reports_dir / "survival_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {km_plot_path}")
    print(f"  - {risk_plot_path}")
    print(f"  - {json_path}")

    return results_json


if __name__ == "__main__":
    main()