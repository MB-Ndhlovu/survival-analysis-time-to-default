"""
Execute full survival analysis pipeline.
"""

import os
import json
import sys
import numpy as np

# Create reports directory
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/chiizer', exist_ok=True)

# Generate data
from src.data_loader import generate_loan_data
print("\n[1/5] Generating loan data...")
df = generate_loan_data(5000)
print(f"  Generated {len(df)} rows | {df['event_default'].sum()} defaults ({df['event_default'].mean()*100:.1f}%)")

# Kaplan-Meier
from src.kaplan_meier import fit_kaplan_meier, print_km_summary
print("\n[2/5] Fitting Kaplan-Meier curves...")
km_results = fit_kaplan_meier(df, plot_path='reports/km_curves.png')
print_km_summary(km_results)

# Cox PH
from src.cox_ph import fit_cox_ph, print_cox_summary
print("\n[3/5] Fitting Cox Proportional Hazards model...")
_, cox_results = fit_cox_ph(df)
print_cox_summary(cox_results)

# Chiizer
from src.chiizer import chiize_all_variables, print_chiizer_summary
print("\n[4/5] Running Risk Chiizer...")
chi_variables = ['credit_score', 'debt_to_income', 'LTV_ratio']
chi_results = chiize_all_variables(df, chi_variables, 'reports/chiizer')
print_chiizer_summary(chi_results)

# Predict survival for new applicant
from src.predict_survival import train_cox_model, predict_survival, print_applicant_score
print("\n[5/5] Scoring new applicant...")
means, stds = train_cox_model(df)

new_applicant = {
    'income': 85000,
    'credit_score': 720,
    'employment_years': 4,
    'debt_to_income': 0.28,
    'loan_amount': 180000,
    'interest_rate': 0.072,
    'LTV_ratio': 0.55
}
prediction = predict_survival(means, stds, new_applicant, df)
print_applicant_score(prediction, new_applicant)

# Compile results JSON
def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

results = {
    'data_summary': {
        'total_observations': int(len(df)),
        'defaults': int(df['event_default'].sum()),
        'censored': int((df['event_default']==0).sum()),
        'censoring_rate': float(df['event_default'].mean())
    },
    'kaplan_meier': {},
    'cox_ph': {
        'hazard_ratios': {k: float(v) for k, v in cox_results['hazard_ratios'].items()},
        'concordance_index': float(cox_results['concordance_index']) if cox_results.get('concordance_index') else None
    },
    'chiizer': chi_results,
    'new_applicant': {
        'features': new_applicant,
        'prediction': {
            '12m_survival': float(prediction['12m_survival']),
            '24m_survival': float(prediction['24m_survival']),
            'expected_months': float(prediction['expected_months'])
        }
    }
}

for band, r in km_results.items():
    median_val = r['median_survival']
    if median_val == float('inf'):
        median_val = None
    results['kaplan_meier'][band] = {
        'n': int(r['n']),
        'events': int(r['events']),
        'median_survival': float(median_val) if median_val is not None else None,
        '12m_survival': float(r['12m_survival']),
        '24m_survival': float(r['24m_survival'])
    }

results = convert_types(results)

with open('reports/survival_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print("  Reports: reports/survival_results.json")
print("  Plots:   reports/km_curves.png, reports/chiizer/")
print("="*60)

# Print final summary for Telegram
summary = f"""Survival Analysis Pipeline Summary:
- 5,000 loans | {df['event_default'].sum()} defaults | {df['event_default'].mean()*100:.1f}% default rate
- KM Median Survival: <580={km_results['<580']['median_survival']:.0f}mo, 740+={km_results['740+']['median_survival']:.0f}mo
- Top Hazard Factor: debt_to_income (HR={cox_results['hazard_ratios'].get('debt_to_income', 'N/A'):.2f})
- New Applicant 12m Survival: {prediction['12m_survival']*100:.1f}%"""

print(summary)