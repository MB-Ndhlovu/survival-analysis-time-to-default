"""Execute full survival analysis pipeline."""

import json
import sys
from pathlib import Path

from src.data_loader import generate_loan_data
from src.kaplan_meier import run as run_km
from src.cox_ph import fit_cox_ph, interpret_coefficients
from src.chiizer import run as run_chiizer
from src.predict_survival import run as run_predict

# Output directory
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Pipeline output summary
pipeline_summary = {
    "data_generation": {},
    "kaplan_meier": {},
    "cox_ph": {},
    "chiizer": {},
    "predictions": {},
}


def main():
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Load/generate data
    print("\n[1/5] Generating loan data...")
    df = generate_loan_data(n=5000, seed=42)

    n_total = len(df)
    n_default = int(df["event_default"].sum())
    n_censored = n_total - n_default
    cens_rate = n_censored / n_total

    pipeline_summary["data_generation"] = {
        "n_records": n_total,
        "n_defaults": n_default,
        "n_censored": n_censored,
        "censoring_rate": round(cens_rate, 4),
    }

    print(f"  Generated {n_total} records")
    print(f"  Defaults: {n_default} ({n_default/n_total:.1%})")
    print(f"  Censored: {n_censored} ({cens_rate:.1%})")

    # Step 2: Kaplan-Meier
    print("\n[2/5] Fitting Kaplan-Meier curves...")
    km_summary, kmfs = run_km(df)

    pipeline_summary["kaplan_meier"] = {
        "bands": km_summary.to_dict(orient="records"),
    }

    # Step 3: Cox PH
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    coef_df, cph = fit_cox_ph(df)
    interp = interpret_coefficients(coef_df)

    pipeline_summary["cox_ph"] = {
        "hazard_ratios": coef_df[["hazard_ratio"]].round(4).to_dict()["hazard_ratio"],
        "concordance_index": round(float(cph.concordance_index_), 4),
        "interpretation": interp.to_dict(orient="records"),
    }

    print("\nTop Default Risk Factors (by hazard ratio):")
    sorted_hr = coef_df.sort_values("hazard_ratio", ascending=False)
    for var, row in sorted_hr.iterrows():
        hr = row["hazard_ratio"]
        direction = "↑" if hr > 1 else "↓"
        print(f"  {direction} {var}: HR={hr:.3f}")

    # Step 4: Chiizer
    print("\n[4/5] Running Risk Chiizer...")
    chiizer_results = run_chiizer(df)

    pipeline_summary["chiizer"] = {
        var: {
            "summary": results["summary"].to_dict(orient="records"),
        }
        for var, results in chiizer_results.items()
    }

    # Step 5: Predictions
    print("\n[5/5] Predicting survival for new applicants...")
    predictions = run_predict(df, cph)

    pipeline_summary["predictions"] = predictions

    # Save results
    output_path = REPORTS_DIR / "survival_results.json"
    with open(output_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2, default=str)

    print(f"\n[OK] Results saved to {output_path}")
    print(f"[OK] Kaplan-Meier plot: reports/km_plot.png")

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - KEY INSIGHTS")
    print("=" * 60)

    print("\n12-Month Survival by Credit Band:")
    for band in km_summary["band"]:
        row = km_summary[km_summary["band"] == band].iloc[0]
        print(f"  {band}: {row['survival_12m']:.1%}")

    print("\n24-Month Survival by Credit Band:")
    for band in km_summary["band"]:
        row = km_summary[km_summary["band"] == band].iloc[0]
        print(f"  {band}: {row['survival_24m']:.1%}")

    print("\nBusiness Insight:")
    print("  Survival analysis reveals WHEN default is likely,")
    print("  not just IF. Subprime borrowers show steep survival")
    print("  curves early on, while Prime borrowers retain")
    print("  higher survival over the 24-month horizon.")

    print("\n" + "=" * 60)
    print("FILES GENERATED")
    print("=" * 60)
    print("  reports/survival_results.json  - Full results JSON")
    print("  reports/km_plot.png            - Kaplan-Meier curves")
    print("  reports/debt_to_income_chiized.png")
    print("  reports/LTV_ratio_chiized.png")
    print("  reports/loan_amount_chiized.png")
    print("  reports/interest_rate_chiized.png")
    print("=" * 60)

    return pipeline_summary


if __name__ == "__main__":
    results = main()