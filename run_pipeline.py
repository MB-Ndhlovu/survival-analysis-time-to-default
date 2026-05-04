import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import generate_loan_data
from kaplan_meier import create_credit_bands, fit_kaplan_meier, compute_survival_probabilities
from cox_ph import fit_cox_ph, get_top_risk_factors
from chiizer import build_risk_chiizer
from predict_survival import predict_survival_for_applicant
from lifelines import KaplanMeierFitter

def main():
    print("=" * 60)
    print("TIME-TO-DEFAULT SURVIVAL ANALYSIS")
    print("=" * 60)
    
    # 1. Load/generate data
    print("\n[1] Generating loan data...")
    df = generate_loan_data(n=5000)
    df = create_credit_bands(df)
    
    n_total = len(df)
    n_defaults = df['event_default'].sum()
    n_censored = n_total - n_defaults
    censoring_rate = n_censored / n_total
    
    print(f"  Total loans: {n_total}")
    print(f"  Defaults: {n_defaults} ({n_defaults/n_total*100:.1f}%)")
    print(f"  Censored: {n_censored} ({censoring_rate*100:.1f}%)")
    
    # 2. Kaplan-Meier Analysis
    print("\n[2] Fitting Kaplan-Meier curves...")
    km_results, kmf = fit_kaplan_meier(df)
    survival_probs = compute_survival_probabilities(df, kmf)
    
    print("\n  Median Survival Time by Credit Band:")
    for band, data in km_results.items():
        median = data['median_survival']
        median_str = f"{median:.1f} months" if isinstance(median, float) else median
        print(f"    {band}: {median_str}")
    
    print("\n  Survival Probabilities by Credit Band:")
    for band, probs in survival_probs.items():
        print(f"    {band}:")
        print(f"      12-month: {probs['survival_12m']*100:.1f}%")
        print(f"      24-month: {probs['survival_24m']*100:.1f}%")
    
    # 3. Cox PH Model
    print("\n[3] Fitting Cox Proportional Hazards model...")
    cph, cox_results = fit_cox_ph(df)
    
    print(f"\n  Model Concordance Index: {cox_results['concordance']:.4f}")
    print("  (Concordance > 0.6 indicates good predictive power)")
    
    print("\n  Hazard Ratios (exp(coef) - >1 means higher risk):")
    for var, hr in cox_results['hazard_ratios'].items():
        direction = "↑" if hr > 1 else "↓"
        print(f"    {var}: {hr:.4f} {direction}")
    
    print("\n  Top Risk Factors:")
    top_factors = get_top_risk_factors(cox_results, n=5)
    for i, (factor, hr) in enumerate(top_factors, 1):
        print(f"    {i}. {factor}: HR={hr:.4f}")
    
    # 4. Risk Chiizer
    print("\n[4] Building risk chiizer...")
    chiizer_results = build_risk_chiizer(df)
    
    print("\n  24-Month Survival by Variable Categories:")
    for var, categories in chiizer_results.items():
        print(f"\n    {var}:")
        for cat, stats in categories.items():
            print(f"      {cat}: {stats['survival_24m']*100:.1f}% (n={stats['n']})")
    
    # 5. New Applicant Prediction
    print("\n[5] Predicting survival for new applicant...")
    
    # Build KMF dict for prediction
    kmf_dict = {}
    for band in ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']:
        band_data = df[df['credit_band'] == band]
        kmf_band = KaplanMeierFitter()
        kmf_band.fit(band_data['time_end'], band_data['event_default'])
        kmf_dict[band] = kmf_band
    
    new_applicant = {
        'income': 85000,
        'credit_score': 720,
        'employment_years': 6,
        'debt_to_income': 0.35,
        'loan_amount': 180000,
        'interest_rate': 8.5,
        'LTV_ratio': 0.75
    }
    
    pred_result = predict_survival_for_applicant(cph, new_applicant, kmf_dict)
    
    # Get band survival for reference
    score = new_applicant['credit_score']
    if score < 580:
        band = 'Poor (<580)'
    elif score < 670:
        band = 'Fair (580-669)'
    elif score < 740:
        band = 'Good (670-739)'
    else:
        band = 'Excellent (740+)'
    
    band_km = kmf_dict[band]
    s12 = band_km.predict(12)
    s24 = band_km.predict(24)
    
    print(f"\n  Applicant credit score {score} ({band})")
    print(f"  Predicted 12-month survival: {s12*100:.1f}%")
    print(f"  Predicted 24-month survival: {s24*100:.1f}%")
    print(f"  Relative hazard vs baseline: {pred_result['partial_hazard']:.4f}")
    
    # 6. Save results
    print("\n[6] Saving results...")
    
    results = {
        'data_summary': {
            'total_loans': int(n_total),
            'defaults': int(n_defaults),
            'censored': int(n_censored),
            'censoring_rate': round(float(censoring_rate), 4)
        },
        'km_results': {
            band: {
                'n': int(data['n']),
                'events': int(data['events']),
                'median_survival': str(data['median_survival']) if not isinstance(data['median_survival'], float) else float(data['median_survival'])
            }
            for band, data in km_results.items()
        },
        'survival_probabilities': survival_probs,
        'cox_ph_results': {
            'concordance': float(cox_results['concordance']),
            'hazard_ratios': {k: float(v) for k, v in cox_results['hazard_ratios'].items()}
        },
        'top_risk_factors': [{'factor': f, 'hr': float(hr)} for f, hr in top_factors],
        'chiizer_results': chiizer_results,
        'applicant_prediction': {
            'credit_score': int(score),
            'band': band,
            'survival_12m': round(float(s12), 4),
            'survival_24m': round(float(s24), 4),
            'relative_hazard': round(float(pred_result['partial_hazard']), 4)
        }
    }
    
    with open('/home/workspace/Projects/survival-analysis-time-to-default/reports/survival_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey Business Insights:")
    print("• Survival analysis reveals WHEN default is likely, not just IF")
    print(f"• Lower credit scores dramatically reduce survival probability")
    print(f"• Concordance index of {cox_results['concordance']:.2f} shows model predictive power")
    print("• Key drivers: credit score, interest rate, and LTV ratio")
    print("\nFiles saved:")
    print("  - reports/km_survival_curves.png")
    print("  - reports/risk_chiizer.png")
    print("  - reports/applicant_survival.png")
    print("  - reports/survival_results.json")
    
    return results

if __name__ == "__main__":
    results = main()