"""Predict survival function for new loan applicants."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter


def predict_survival_for_applicant(cph: CoxPHFitter, 
                                   applicant: dict,
                                   baseline_hazard: bool = True) -> np.ndarray:
    """
    Predict survival function for a new applicant.
    
    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox PH model
    applicant : dict
        Applicant features
    baseline_hazard : bool
        If True, use baseline hazard; if False, use median survival
    
    Returns
    -------
    np.ndarray
        Predicted survival probabilities at each time point
    """
    # Prepare applicant data
    applicant_df = pd.DataFrame([applicant])
    
    # Predict hazard (returns array of cumulative hazard)
    cumulative_hazard = cph.predict_cumulative_hazard(applicant_df)
    
    # Survival = exp(-cumulative_hazard)
    survival_function = np.exp(-cumulative_hazard.values.flatten())
    
    return survival_function


def predict_for_new_applicant(df: pd.DataFrame, applicant: dict) -> dict:
    """
    Fit Cox model and predict survival for a new applicant.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    applicant : dict
        New applicant features
    
    Returns
    -------
    dict
        Prediction results
    """
    # Prepare features (same as cox_ph.py)
    model_df = df.copy()
    model_df['income_k'] = model_df['income'] / 1000
    model_df['loan_amount_k'] = model_df['loan_amount'] / 1000
    
    feature_cols = ['credit_score', 'income_k', 'employment_years',
                    'debt_to_income', 'loan_amount_k', 'interest_rate', 'LTV_ratio']
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(model_df[['time_end', 'event_default'] + feature_cols],
           duration_col='time_end',
           event_col='event_default')
    
    # Prepare applicant with same features
    app_df = pd.DataFrame([applicant])
    app_df['income_k'] = app_df['income'] / 1000
    app_df['loan_amount_k'] = app_df['loan_amount'] / 1000
    
    # Predict survival for the applicant
    times = np.linspace(0, 36, 100)  # 0 to 36 months
    
    # Get survival function - returns DataFrame with times as index, columns=individuals
    surv_df = cph.predict_survival_function(app_df[feature_cols], times=times)
    
    # Survival values (same for all times since single applicant)
    survival_at_times = surv_df.iloc[:, 0].values.tolist()
    
    # Get 12m and 24m survival by finding closest times in the index
    idx_12 = np.argmin(np.abs(surv_df.index.values - 12))
    idx_24 = np.argmin(np.abs(surv_df.index.values - 24))
    
    result = {
        'survival_curve': {
            'times': surv_df.index.values.tolist(),
            'probabilities': survival_at_times,
        },
        '12_month_survival': float(surv_df.iloc[idx_12, 0]),
        '24_month_survival': float(surv_df.iloc[idx_24, 0]),
        'expected_months_to_default': float(np.argmax(np.array(survival_at_times) < 0.5) if np.any(np.array(survival_at_times) < 0.5) else 36),
    }
    
    return result


def plot_applicant_survival(applicant_result: dict, 
                            save_path: str = 'reports/applicant_survival.png',
                            applicant_label: str = 'New Applicant') -> None:
    """Plot predicted survival curve for an applicant."""
    times = applicant_result['survival_curve']['times']
    probs = applicant_result['survival_curve']['probabilities']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(times, probs, linewidth=2.5, color='#1f77b4', label=applicant_label)
    ax.fill_between(times, 0, probs, alpha=0.2, color='#1f77b4')
    
    # Mark key points
    for m in [12, 24]:
        idx = np.argmin(np.abs(np.array(times) - m))
        ax.scatter(m, probs[idx], color='#d62728', s=80, zorder=5)
        ax.annotate(f'{m}m: {probs[idx]:.1%}',
                   xy=(m, probs[idx]), xytext=(m+1, probs[idx]+0.05),
                   fontsize=10, color='#d62728')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% survival')
    
    ax.set_xlabel('Months Since Origination', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Prediction: {applicant_label}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved applicant survival plot to {save_path}")


def print_applicant_prediction(result: dict, applicant: dict) -> None:
    """Print prediction results for an applicant."""
    print("\n" + "="*70)
    print("NEW APPLICANT SURVIVAL PREDICTION")
    print("="*70)
    print(f"\nApplicant Profile:")
    print(f"  Credit Score: {applicant['credit_score']}")
    print(f"  Income: ${applicant['income']:,.0f}")
    print(f"  Employment: {applicant['employment_years']:.1f} years")
    print(f"  Debt-to-Income: {applicant['debt_to_income']:.1%}")
    print(f"  Loan Amount: ${applicant['loan_amount']:,.0f}")
    print(f"  Interest Rate: {applicant['interest_rate']:.2f}%")
    print(f"  LTV Ratio: {applicant['LTV_ratio']:.2f}")
    
    print(f"\nSurvival Probabilities:")
    print(f"  12-month survival: {result['12_month_survival']:.1%}")
    print(f"  24-month survival: {result['24_month_survival']:.1%}")
    
    if result['expected_months_to_default'] < 36:
        print(f"  Expected time to 50% default: {result['expected_months_to_default']:.0f} months")
    else:
        print(f"  Expected time to 50% default: Not expected within 36 months")


if __name__ == '__main__':
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    
    # Example applicant
    applicant = {
        'credit_score': 650,
        'income': 55000,
        'employment_years': 3.5,
        'debt_to_income': 0.32,
        'loan_amount': 180000,
        'interest_rate': 7.5,
        'LTV_ratio': 0.82,
    }
    
    result = predict_for_new_applicant(df, applicant)
    print_applicant_prediction(result, applicant)