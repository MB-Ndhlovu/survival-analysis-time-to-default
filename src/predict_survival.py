"""Predict survival function for new loan applicants."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from src.cox_ph import prepare_covariates


def build_survival_predictor(df: pd.DataFrame):
    """Build Cox PH model for survival prediction."""
    X = prepare_covariates(df)
    
    covariates = [
        'credit_score', 'employment_years', 'debt_to_income',
        'loan_amount', 'LTV_ratio'
    ]
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        X[['time_end', 'event_default'] + covariates],
        duration_col='time_end',
        event_col='event_default'
    )
    
    return cph


def create_baseline_km(df: pd.DataFrame) -> KaplanMeierFitter:
    """Create baseline Kaplan-Meier for median applicant."""
    kmf = KaplanMeierFitter()
    kmf.fit(df['time_end'], df['event_default'])
    return kmf


def predict_survival(cph: CoxPHFitter, kmf_baseline: KaplanMeierFitter, 
                     applicant: dict, times: np.ndarray = None) -> dict:
    """Predict survival curve for a new applicant.
    
    Args:
        cph: Fitted Cox PH model.
        kmf_baseline: Baseline Kaplan-Meier fitter.
        applicant: Dictionary with applicant features.
        times: Time points to predict at. Defaults to range(0, 61).
    
    Returns:
        Dictionary with predicted survival probabilities.
    """
    if times is None:
        times = np.arange(0, 61)
    
    log_income = np.log1p(applicant['income'])
    log_loan_amount = np.log1p(applicant['loan_amount'])
    high_dti = 1 if applicant['debt_to_income'] > 0.36 else 0
    high_ltv = 1 if applicant['LTV_ratio'] > 0.8 else 0
    short_employment = 1 if applicant['employment_years'] < 2 else 0
    
    X_new = pd.DataFrame([{
        'credit_score': applicant['credit_score'],
        'employment_years': applicant['employment_years'],
        'debt_to_income': applicant['debt_to_income'],
        'loan_amount': applicant['loan_amount'],
        'LTV_ratio': applicant['LTV_ratio']
    }])
    
    partial_hazard = cph.predict_partial_hazard(X_new).values
    if isinstance(partial_hazard, np.ndarray) and partial_hazard.ndim > 0:
        partial_hazard = partial_hazard[0]
    linear_predictor = np.log(float(partial_hazard)) if float(partial_hazard) > 0 else 0
    
    baseline_hazard = cph.baseline_hazard_
    baseline_cumulative_hazard = cph.baseline_cumulative_hazard_
    
    survival_probs = []
    for t in times:
        try:
            base_surv = kmf_baseline.predict(t)
        except:
            base_surv = np.exp(-cph.predict_partial_hazard(X_new).values[0] * t * 0.05)
        
        hr_adjustment = np.exp(linear_predictor)
        survival_prob = np.exp(-base_surv ** hr_adjustment) if base_surv > 0 else 1.0
        survival_probs.append(float(survival_prob))
    
    return {
        'times': times.tolist(),
        'survival_probability': survival_probs,
        'survival_12m': survival_probs[min(12, len(survival_probs)-1)],
        'survival_24m': survival_probs[min(24, len(survival_probs)-1)],
        'median_time_to_default': find_median_time(times, survival_probs)
    }


def find_median_time(times: np.ndarray, survival_probs: list) -> float:
    """Find median survival time."""
    for i, prob in enumerate(survival_probs):
        if prob <= 0.5:
            return float(times[i])
    return float('inf')


def risk_profile(applicant: dict) -> str:
    """Classify applicant risk profile."""
    score = applicant['credit_score']
    dti = applicant['debt_to_income']
    ltv = applicant['LTV_ratio']
    
    risk_score = 0
    if score < 580: risk_score += 3
    elif score < 670: risk_score += 2
    elif score < 740: risk_score += 1
    
    if dti > 0.43: risk_score += 2
    elif dti > 0.36: risk_score += 1
    
    if ltv > 0.9: risk_score += 2
    elif ltv > 0.8: risk_score += 1
    
    if risk_score <= 2: return "Prime"
    elif risk_score <= 4: return "Low"
    elif risk_score <= 6: return "Medium"
    else: return "High"


def run_prediction_demo(df: pd.DataFrame, output_dir: str = 'reports') -> dict:
    """Run prediction for sample applicants."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cph = build_survival_predictor(df)
    kmf_baseline = create_baseline_km(df)
    
    sample_applicants = [
        {
            'name': 'Prime Borrower',
            'income': 95000,
            'credit_score': 760,
            'employment_years': 8,
            'debt_to_income': 0.28,
            'loan_amount': 220000,
            'interest_rate': 0.065,
            'LTV_ratio': 0.70
        },
        {
            'name': 'Near-Prime Borrower',
            'income': 65000,
            'credit_score': 660,
            'employment_years': 3,
            'debt_to_income': 0.35,
            'loan_amount': 180000,
            'interest_rate': 0.085,
            'LTV_ratio': 0.82
        },
        {
            'name': 'Subprime Borrower',
            'income': 42000,
            'credit_score': 580,
            'employment_years': 1,
            'debt_to_income': 0.44,
            'loan_amount': 140000,
            'interest_rate': 0.115,
            'LTV_ratio': 0.92
        }
    ]
    
    predictions = []
    for applicant in sample_applicants:
        pred = predict_survival(cph, kmf_baseline, applicant)
        profile = risk_profile(applicant)
        
        predictions.append({
            'name': applicant['name'],
            'profile': profile,
            'survival_12m': round(pred['survival_12m'], 3),
            'survival_24m': round(pred['survival_24m'], 3),
            'median_months': pred['median_time_to_default'] if pred['median_time_to_default'] != float('inf') else 'N/A'
        })
    
    return {
        'predictions': predictions,
        'cph_model': cph
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    
    df = load_data()
    result = run_prediction_demo(df)
    
    print("=== Survival Predictions for Sample Applicants ===")
    for pred in result['predictions']:
        print(f"\n{pred['name']} ({pred['profile']} risk):")
        print(f"  12-month survival: {pred['survival_12m']:.1%}")
        print(f"  24-month survival: {pred['survival_24m']:.1%}")
        print(f"  Median time to default: {pred['median_months']}")