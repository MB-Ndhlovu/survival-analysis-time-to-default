"""Execute full time-to-default survival analysis pipeline."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.kaplan_meier import fit_kaplan_meier, plot_km_curves, km_summary
from src.cox_ph import fit_cox_ph, print_cox_summary
from src.chiizer import chiize, chiizer_summary
from src.predict_survival import predict_survival, print_applicant_survival

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def build_results_json(km_results, cox_summary_df, chiizer_results, new_applicant_pred):
    results = {
        'generated_at': datetime.now().isoformat(),
        'data_summary': {},
        'kaplan_meier': {},
        'cox_ph': {},
        'chiizer': {},
        'new_applicant_prediction': {},
    }

    for label, km_result in km_results.items():
        results['kaplan_meier'][label] = {
            'n': km_result['n'],
            'median_survival_months': float(km_result['median']) if not np.isinf(km_result['median']) else None,
            'survival_12m': float(km_result['kmf'].survival_function_at_times(12).values[0]),
            'survival_24m': float(km_result['kmf'].survival_function_at_times(24).values[0]),
        }

    for var in cox_summary_df.index:
        row = cox_summary_df.loc[var]
        results['cox_ph'][var] = {
            'coefficient': float(row['coef']),
            'hazard_ratio': float(row['hazard_ratio']),
            'p_value': float(row['p']),
            'ci_lower': float(row.get('hr_lower', 0)),
            'ci_upper': float(row.get('hr_upper', 0)),
        }

    for var, var_results in chiizer_results.items():
        results['chiizer'][var] = {}
        for bin_label, res in var_results.items():
            results['chiizer'][var][bin_label] = {
                'n': res['n'],
                'median_survival': float(res['median']) if not np.isinf(res['median']) else None,
                's12': float(res['s12']),
                's24': float(res['s24']),
            }

    na = new_applicant_pred['new_applicant']
    results['new_applicant_prediction'] = {
        'applicant': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in na.items()},
        'survival_curve': {
            'timeline': new_applicant_pred['timeline'],
            'probabilities': [round(float(p), 4) for p in new_applicant_pred['survival_probability']],
        },
        'linear_predictor': new_applicant_pred.get('linear_predictor'),
    }

    return results

def main():
    print("=" * 60)
    print("  TIME-TO-DEFAULT SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    n = len(df)
    default_rate = df['event_default'].mean()
    censored_rate = 1 - default_rate
    print(f"  n = {n}")
    print(f"  Default rate: {default_rate:.1%}")
    print(f"  Censored rate: {censored_rate:.1%}")

    # 2. Kaplan-Meier
    print("\n[2/6] Fitting Kaplan-Meier curves...")
    km_results = fit_kaplan_meier(df)
    km_summary(km_results)
    km_plot_path = os.path.join(REPORTS_DIR, 'km_survival_curves.png')
    plot_km_curves(km_results, save_path=km_plot_path)

    # 3. Cox PH
    print("\n[3/6] Fitting Cox Proportional Hazards model...")
    cph = fit_cox_ph(df)
    cox_summary_df = print_cox_summary(cph)

    # 4. Chiizer
    print("\n[4/6] Running risk chiizer...")
    chiizer_vars = ['debt_to_income', 'LTV_ratio', 'interest_rate']
    chiizer_results = {}
    for var in chiizer_vars:
        res = chiize(df, var, n_bins=5)
        chiizer_summary(res, var)
        chiizer_results[var] = res

    # 5. Predict for new applicant
    print("\n[5/6] Predicting survival for new applicant...")
    new_applicant = {
        'income': 350_000,
        'credit_score': 610,
        'employment_years': 2.0,
        'debt_to_income': 0.40,
        'loan_amount': 200_000,
        'interest_rate': 0.145,
        'LTV_ratio': 0.85,
    }
    pred = predict_survival(cph, new_applicant, df)
    print_applicant_survival(pred)

    # 6. Save results
    print("\n[6/6] Saving results...")
    results = build_results_json(km_results, cox_summary_df, chiizer_results, pred)
    results['data_summary'] = {
        'n_records': int(n),
        'default_rate': float(default_rate),
        'censored_rate': float(censored_rate),
    }

    results_path = os.path.join(REPORTS_DIR, 'survival_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    # Summary for Telegram
    s12 = round(pred['survival_probability'][11], 3)
    s24 = round(pred['survival_probability'][23], 3)

    # Top risk factors from Cox PH (sorted by |HR - 1| descending)
    hr_col = cox_summary_df['hazard_ratio']
    top_idx = hr_col.sub(1).abs().sort_values(ascending=False).head(3).index

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nData: {n} loans, {default_rate:.1%} default rate, {censored_rate:.1%} censored")
    print(f"\nKey Cox PH hazard ratios (top risk drivers):")
    for var in top_idx:
        row = cox_summary_df.loc[var]
        print(f"  {var:25s} HR = {row['hazard_ratio']:.4f} (p = {row['p']:.4f})")
    print(f"\nNew applicant (score=610, rate=14.5%, DTI=0.40):")
    print(f"  S(12) = {s12:.1%}, S(24) = {s24:.1%}")

    return results

if __name__ == '__main__':
    main()