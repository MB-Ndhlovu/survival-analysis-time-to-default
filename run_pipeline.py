"""
Execute full survival analysis pipeline.
"""
import json
import pandas as pd
from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_kaplan_meier, compute_median_times, survival_at_months, plot_km_curves, CREDIT_BANDS
from src.cox_ph import fit_cox_ph, print_cox_summary, hazard_ratios
from src.chiizer import chiize_all_vars
from src.predict_survival import predict_survival, print_applicant_prediction


def main():
    print("=" * 60)
    print("PROJECT 6: TIME-TO-DEFAULT SURVIVAL ANALYSIS")
    print("=" * 60)

    # 1. Load/generate data
    print("\n[1] Generating loan data (n=5000, 35% censored at 24 months)...")
    df = generate_loan_data(n=5000, censor_at=24)
    print(f"    Shape: {df.shape}")
    print(f"    Censored: {(df['event_default'] == 0).mean():.1%}")
    print(f"    Defaulted: {(df['event_default'] == 1).mean():.1%}")

    # 2. Kaplan-Meier
    print("\n[2] Fitting Kaplan-Meier curves by credit score band...")
    kmf_by_band = fit_kaplan_meier(df)

    print("\n    --- Median Time to Default ---")
    medians = compute_median_times(kmf_by_band)
    for _, row in medians.iterrows():
        mtd = row["median_time_to_default"]
        val = f"{mtd:.0f} months" if mtd is not None else "> 24 months (not reached)"
        print(f"    {row['band']}: {val}")

    print("\n    --- 12/24-month Survival Probabilities ---")
    surv_probs = survival_at_months(kmf_by_band, [12, 24])
    for band in surv_probs["band"].unique():
        band_data = surv_probs[surv_probs["band"] == band]
        s12 = band_data[band_data["month"] == 12]["survival_prob"].values[0]
        s24 = band_data[band_data["month"] == 24]["survival_prob"].values[0]
        print(f"    {band}: 12-mo={s12:.1%}, 24-mo={s24:.1%}")

    # Save KM plot
    plot_km_curves(kmf_by_band, save_path="reports/km_curves.png")

    # 3. Cox PH
    print("\n[3] Fitting Cox Proportional Hazards model...")
    cph = fit_cox_ph(df)
    print_cox_summary(cph)

    # 4. Risk Chiizer
    print("\n[4] Building risk categories via chiizer...")
    chiize_results = chiize_all_vars(df)

    # 5. New applicant prediction
    print("\n[5] Predicting survival for new applicant...")
    new_applicant = {
        "credit_score": 680,
        "income": 450_000,
        "employment_years": 3.5,
        "debt_to_income": 0.28,
        "loan_amount": 250_000,
        "interest_rate": 0.095,
        "LTV_ratio": 0.72,
    }
    surv_df = predict_survival(cph, new_applicant)
    print_applicant_prediction(new_applicant, surv_df)

    # 6. Compile results JSON
    print("\n[6] Saving results to reports/survival_results.json...")
    hr_df = hazard_ratios(cph)

    results = {
        "data_summary": {
            "n_loans": int(len(df)),
            "censored_pct": float(round((df["event_default"] == 0).mean(), 4)),
            "defaulted_pct": float(round((df["event_default"] == 1).mean(), 4)),
            "censor_at_months": 24,
        },
        "median_time_to_default": {
            row["band"]: (float(row["median_time_to_default"]) if pd.notna(row["median_time_to_default"]) and row["median_time_to_default"] is not None else None)
            for _, row in medians.iterrows()
        },
        "survival_probabilities_12_24_months": {
            band: {
                "12_month": float(surv_probs[(surv_probs["band"] == band) & (surv_probs["month"] == 12)]["survival_prob"].values[0]),
                "24_month": float(surv_probs[(surv_probs["band"] == band) & (surv_probs["month"] == 24)]["survival_prob"].values[0]),
            }
            for band in surv_probs["band"].unique()
        },
        "cox_ph_hazard_ratios": {
            idx: {
                "hazard_ratio": float(row["hazard_ratio"]),
                "coefficient": float(row["coefficient"]),
                "p_value": float(row["p_value"]),
            }
            for idx, row in hr_df.iterrows()
        },
        "new_applicant_prediction": {
            "applicant": {k: float(v) for k, v in new_applicant.items()},
            "survival_curve": {
                str(r["month"]): float(r["survival_prob"])
                for _, r in surv_df.iterrows()
            },
            "12_month_default_prob": float(round(1 - surv_df[surv_df["month"] == 12]["survival_prob"].values[0], 4)),
            "24_month_default_prob": float(round(1 - surv_df[surv_df["month"] == 24]["survival_prob"].values[0], 4)),
        },
    }

    with open("reports/survival_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nBusiness Insight:")
    print("  Survival analysis tells you WHEN default is likely,")
    print("  not just IF. A 10% PD in 12 months is very different")
    print("  from 10% over 5 years.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()