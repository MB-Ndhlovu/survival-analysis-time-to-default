"""
Time-to-Default Survival Analysis Pipeline
Executes the full pipeline: data generation, KM curves, Cox PH, Chiizer, predictions.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from src.data_loader import generate_loan_data, get_credit_score_band
from src.kaplan_meier import fit_kaplan_meier_by_band, plot_kaplan_meier, summarize_km_results, export_km_json
from src.cox_ph import fit_cox_ph, get_hazard_ratios, top_hazard_factors, print_cox_summary, export_cox_json
from src.chiizer import chiize_all, print_chiizer_summary
from src.predict_survival import (
    prepare_applicant_features, predict_survival_bands,
    plot_applicant_survival
)


def run():
    print("=" * 70)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE".center(70))
    print("=" * 70)

    # ── 1. Data Generation ───────────────────────────────────────────────
    print("\n[1/5] Generating loan data...")
    df = generate_loan_data(n=5000, censor_at_month=24)
    print(f"  Generated {len(df)} loans")
    print(f"  Censored: {(df['event_default']==0).sum()} ({(df['event_default']==0).mean()*100:.1f}%)")
    print(f"  Events:   {df['event_default'].sum()}")

    # ── 2. Kaplan-Meier ──────────────────────────────────────────────────
    print("\n[2/5] Fitting Kaplan-Meier curves...")
    km_results = fit_kaplan_meier_by_band(df)
    summarize_km_results(km_results)

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    plot_kaplan_meier(km_results, save_path=os.path.join(reports_dir, "km_survival_curves.png"))
    export_km_json(km_results, os.path.join(reports_dir, "km_results.json"))

    # ── 3. Cox PH ─────────────────────────────────────────────────────────
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cph, X_train = fit_cox_ph(df)
    print_cox_summary(cph)

    print("\n  Top 5 Default Hazard Drivers (highest HR):")
    top_factors = top_hazard_factors(cph, n=5)
    for _, row in top_factors.iterrows():
        print(f"  {row.name:<20} HR={row['hazard_ratio']:.3f}  p={row['p_value']:.4f}")

    export_cox_json(cph, os.path.join(reports_dir, "cox_ph_results.json"))

    feature_means = X_train.mean().to_dict()
    feature_stds = X_train.std().to_dict()

    # ── 4. Risk Chiizer ───────────────────────────────────────────────────
    print("\n[4/5] Running Risk Chiizer on continuous variables...")
    chi_results = chiize_all(df)
    print_chiizer_summary(chi_results)

    # ── 5. New Applicant Prediction ─────────────────────────────────────
    print("\n[5/5] Predicting survival for new applicant...")
    new_applicant = {
        "income": 180000,
        "credit_score": 690,
        "employment_years": 4,
        "debt_to_income": 0.30,
        "loan_amount": 320000,
        "interest_rate": 0.11,
        "LTV_ratio": 0.78,
    }

    X_new = prepare_applicant_features(new_applicant, 0, 0, feature_means, feature_stds)
    survival_pred = predict_survival_bands(X_new, cph)

    s12 = float(survival_pred[survival_pred["timeline"] == 12]["survival_probability"].values[0])
    s24 = float(survival_pred[survival_pred["timeline"] == 24]["survival_probability"].values[0])

    print(f"\n  New Applicant Profile:")
    for k, v in new_applicant.items():
        print(f"    {k}: {v}")
    print(f"\n  Predicted 12-month survival: {s12:.1%}")
    print(f"  Predicted 24-month survival: {s24:.1%}")

    plot_applicant_survival(
        survival_pred, new_applicant,
        save_path=os.path.join(reports_dir, "applicant_survival_curve.png")
    )

    # ── Summary JSON ───────────────────────────────────────────────────────
    top_factors_df = top_hazard_factors(cph, n=5).reset_index()
    top_factors_df = top_factors_df.rename(columns={"index": "covariate"})
    hazard_ratios_list = get_hazard_ratios(cph).reset_index().rename(columns={"index": "covariate"})
    hazard_ratios_list = hazard_ratios_list.to_dict("records")
    for r in hazard_ratios_list:
        for k, v in r.items():
            if isinstance(v, (float, np.floating, np.integer)) and not isinstance(v, bool):
                r[k] = round(float(v), 6)

    summary = {
        "dataset": {
            "n_loans": len(df),
            "n_defaults": int(df["event_default"].sum()),
            "n_censored": int((df["event_default"] == 0).sum()),
            "censorship_rate_pct": round((df["event_default"] == 0).mean() * 100, 2),
            "observation_months": 24,
        },
        "kaplan_meier": {
            band: {k: v for k, v in data.items() if k != "kmf"}
            for band, data in km_results.items()
        },
        "cox_ph": {
            "concordance_index": round(float(cph.concordance_index_), 4),
            "top_5_hazard_drivers": [
                {k: (round(float(v), 4) if isinstance(v, (float, np.floating, np.integer)) and not isinstance(v, bool) and not isinstance(v, str) else v)
                 for k, v in row.items()}
                for _, row in top_hazard_factors(cph, n=5).reset_index().rename(columns={"index": "covariate"}).iterrows()
            ],
            "all_hazard_ratios": hazard_ratios_list,
        },
        "chiizer": {var: bins for var, bins in chi_results.items()},
        "new_applicant_prediction": {
            "profile": new_applicant,
            "survival_12_month": round(float(s12), 4),
            "survival_24_month": round(float(s24), 4),
        }
    }

    summary_path = os.path.join(reports_dir, "survival_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] Full results → {summary_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE".center(70))
    print("=" * 70)
    print("""
Key Insights:
  - Survival analysis reveals WHEN default occurs, not just IF
  - Cox PH hazard ratios quantify which factors drive default risk
  - Kaplan-Meier shows clear separation by credit score band
  - Chiizer bins continuous vars into interpretable risk segments
  - Individual survival predictions enable risk-based pricing
""")

    return summary


if __name__ == "__main__":
    run()