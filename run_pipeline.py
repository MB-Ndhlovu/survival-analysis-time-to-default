"""
Full survival analysis pipeline for time-to-default modeling.
"""
import json
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_loan_data
from src.kaplan_meier import fit_km_by_credit_band, get_credit_band
from src.cox_ph import fit_cox_ph
from src.chiizer import run_chiizer
from src.predict_survival import predict_survival


def clean_for_json(obj):
    """Recursively clean objects for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif hasattr(obj, 'item'):  # numpy types
        return clean_for_json(obj.item())
    else:
        return obj


def run_pipeline():
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating loan data (5000 observations)...")
    df = generate_loan_data()
    print(f"  - Shape: {df.shape}")
    print(f"  - Default rate: {df['event_default'].mean():.1%}")
    print(f"  - Censored rate: {(df['event_default'] == 0).mean():.1%}")

    # 2. Kaplan-Meier by credit band
    print("\n[2/5] Fitting Kaplan-Meier curves by credit score band...")
    km_results = fit_km_by_credit_band(df)
    print("  - Saved: reports/km_survival_curves.png")

    # 3. Cox PH model
    print("\n[3/5] Fitting Cox Proportional Hazards model...")
    cph, coefs = fit_cox_ph(df)

    # 4. Risk Chiizer
    print("\n[4/5] Running Risk Chiizer...")
    chiizer_results = run_chiizer(df)

    # 5. Predict for new applicant
    print("\n[5/5] Predicting survival for new applicant...")
    new_applicant = {'credit_score': 680, 'debt_to_income': 0.28, 'LTV_ratio': 0.45}
    predictions = predict_survival(df, new_applicant)

    # Build results dicts with full cleaning
    km_summary = {}
    for band, r in km_results.items():
        ms = r['median_survival_months']
        km_summary[band] = {
            'n_obs': clean_for_json(r['n_obs']),
            'n_events': clean_for_json(r['n_events']),
            'median_survival_months': clean_for_json(ms),
            'survival_12m': clean_for_json(r['survival_12m']),
            'survival_24m': clean_for_json(r['survival_24m']),
        }

    hr_summary = {}
    for feat, row in coefs.iterrows():
        hr_summary[feat] = {
            'coefficient': clean_for_json(row['coefficient']),
            'hazard_ratio': clean_for_json(row['hazard_ratio']),
            'p_value': clean_for_json(row['p_value']),
        }

    chiizer_summary = {}
    for var, res in chiizer_results.items():
        chiizer_summary[var] = {}
        for bin_label, stats in res.items():
            chiizer_summary[var][bin_label] = clean_for_json(stats)

    predictions_clean = clean_for_json(predictions)

    pipeline_results = {
        'km_by_credit_band': km_summary,
        'cox_ph_hazard_ratios': hr_summary,
        'chiizer_results': chiizer_summary,
        'new_applicant_predictions': predictions_clean,
        'new_applicant_features': new_applicant,
        'data_summary': {
            'n_total': clean_for_json(len(df)),
            'default_rate': clean_for_json(df['event_default'].mean()),
            'censored_rate': clean_for_json((df['event_default'] == 0).mean()),
        }
    }

    # Save results
    os.makedirs('reports', exist_ok=True)
    with open('reports/survival_results.json', 'w') as f:
        json.dump(pipeline_results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nKey Results:")
    print("  KM Curves: reports/km_survival_curves.png")
    print("  Full results: reports/survival_results.json")

    # Summary strings
    summary_lines = []
    summary_lines.append("KM Median Survival by Band:")
    for band in ['< 580 (Deep Subprime)', '580-669 (Subprime)', '670-739 (Near Prime)', '740+ (Prime)']:
        m = km_summary[band]['median_survival_months']
        s12 = km_summary[band]['survival_12m']
        s24 = km_summary[band]['survival_24m']
        ms_str = f"{m:.1f}m" if m is not None else "N/A (50% still surviving)"
        summary_lines.append(f"  {band}: median={ms_str}, 12m={s12:.3f}, 24m={s24:.3f}")

    top_risk_factors = sorted(hr_summary.items(), key=lambda x: -x[1]['hazard_ratio'])[:3]
    summary_lines.append("\nTop 3 Risk Factors (by HR):")
    for feat, vals in top_risk_factors:
        summary_lines.append(f"  {feat}: HR={vals['hazard_ratio']:.3f}")

    summary_lines.append(f"\nNew applicant (score=680, DTI=0.28, LTV=0.45):")
    summary_lines.append(f"  12m survival: {predictions['survival_12m']:.3f}")
    summary_lines.append(f"  24m survival: {predictions['survival_24m']:.3f}")
    med = predictions['median_survival']
    if med is None or (isinstance(med, float) and (math.isinf(med) or math.isnan(med))):
        summary_lines.append(f"  Median: >36m (majority haven't defaulted)")
    else:
        summary_lines.append(f"  Median: {med:.1f}m")

    telegram_summary = "\n".join(summary_lines)
    print("\n--- Summary ---")
    print(telegram_summary)

    return pipeline_results, telegram_summary


if __name__ == "__main__":
    results, summary = run_pipeline()