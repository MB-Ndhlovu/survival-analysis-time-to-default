"""Predict survival function for new loan applicants."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter


def predict_survival_for_applicant(
    cph: CoxPHFitter,
    applicant: dict,
    df: pd.DataFrame,
    output_dir: str = 'reports'
) -> dict:
    """Predict survival curve for a new loan applicant using Cox PH model.

    Args:
        cph: Fitted CoxPHFitter
        applicant: Dict with features for new applicant:
            - credit_score: int (500-850)
            - employment_years: float
            - debt_to_income: float
            - loan_amount: float
            - interest_rate: float
            - LTV_ratio: float
            - income: float
        df: Original training data (for baseline survival)

    Returns:
        dict with survival probabilities at key time horizons
    """
    # Build feature vector matching Cox model (all 9 features used in fit)
    features = {
        'credit_score': float(applicant['credit_score']),
        'employment_years': float(applicant['employment_years']),
        'debt_to_income': float(applicant['debt_to_income']),
        'loan_amount': float(applicant['loan_amount']),
        'interest_rate': float(applicant['interest_rate']),
        'LTV_ratio': float(applicant['LTV_ratio']),
        'income': float(applicant['income']),
        'log_income': float(np.log(applicant['income'])),
        'log_loan_amount': float(np.log(applicant['loan_amount'])),
    }

    # All 9 features must match exactly what Cox was fit with
    fit_features = [
        'credit_score', 'employment_years', 'debt_to_income',
        'loan_amount', 'interest_rate', 'LTV_ratio', 'income',
        'log_income', 'log_loan_amount'
    ]

    X = pd.DataFrame([{k: features[k] for k in fit_features}])

    # Get baseline survival (from KM on original data)
    kmf = KaplanMeierFitter()
    kmf.fit(df['time_end'], df['event_default'])

    # Predict log partial hazard ratio for this applicant
    log_hazard = cph.predict_log_partial_hazard(X).values[0]
    hazard_ratio = np.exp(log_hazard)

    # Compute survival curve: S(t)^exp(beta * X - beta * X_mean)
    # = S(t)^exp(log_hr) where S(t) is baseline survival
    baseline_surv = kmf.survival_function_at_times(range(1, 25)).values.flatten()
    predicted_surv = np.power(baseline_surv, hazard_ratio)

    # Key time horizons
    months = [6, 12, 18, 24]
    survival_probs = {m: round(float(predicted_surv[m-1]), 4) for m in months}

    # Print applicant details
    print("\n" + "=" * 55)
    print("  New Applicant Survival Prediction")
    print("=" * 55)
    print(f"  Credit Score:        {applicant['credit_score']}")
    print(f"  Employment:          {applicant['employment_years']:.1f} years")
    print(f"  DTI Ratio:           {applicant['debt_to_income']:.1f}%")
    print(f"  Loan Amount:         ${applicant['loan_amount']:,.0f}")
    print(f"  Interest Rate:       {applicant['interest_rate']:.2%}")
    print(f"  LTV Ratio:           {applicant['LTV_ratio']:.3f}")
    print(f"  Income:              ${applicant['income']:,.0f}")
    print("-" * 55)
    print(f"  Predicted Hazard Ratio: {hazard_ratio:.3f}x baseline")
    print(f"  (HR > 1 means higher than average default risk)")
    print("-" * 55)
    print(f"  6-month survival:   {survival_probs[6]:.2%}")
    print(f"  12-month survival:  {survival_probs[12]:.2%}")
    print(f"  18-month survival:  {survival_probs[18]:.2%}")
    print(f"  24-month survival:  {survival_probs[24]:.2%}")
    print("=" * 55)

    # Plot survival curve
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(1, 25)
    ax.plot(t, predicted_surv, 'b-', linewidth=2.5, label='Applicant Predicted')
    ax.plot(t, baseline_surv, 'k--', linewidth=1.5, alpha=0.6, label='Portfolio Baseline')

    # Reference lines
    for m in months:
        ax.axhline(y=survival_probs[m], color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=m, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel('Months', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title(f"Predicted Survival Curve (HR={hazard_ratio:.2f}x baseline)", fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 25)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/applicant_survival_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'hazard_ratio': round(float(hazard_ratio), 4),
        'survival_probs': survival_probs,
        'baseline_comparison': {
            '12mo_baseline': round(float(baseline_surv[11]), 4),
            '24mo_baseline': round(float(baseline_surv[23]), 4),
        }
    }


def demonstrate_predictions(df: pd.DataFrame, cph: CoxPHFitter, output_dir: str = 'reports'):
    """Demonstrate predictions for different applicant profiles."""
    print("\n\n" + "=" * 65)
    print("  RISK STRATIFIED APPLICANT PREDICTIONS")
    print("=" * 65)

    applicants = [
        {
            'name': 'Low-Risk Prime Borrower',
            'data': {
                'credit_score': 780,
                'employment_years': 10,
                'debt_to_income': 18,
                'loan_amount': 200000,
                'interest_rate': 0.065,
                'LTV_ratio': 0.65,
                'income': 150000,
            }
        },
        {
            'name': 'Near-Prime Borrower',
            'data': {
                'credit_score': 700,
                'employment_years': 4,
                'debt_to_income': 35,
                'loan_amount': 180000,
                'interest_rate': 0.085,
                'LTV_ratio': 0.80,
                'income': 85000,
            }
        },
        {
            'name': 'High-Risk Subprime Borrower',
            'data': {
                'credit_score': 580,
                'employment_years': 1,
                'debt_to_income': 48,
                'loan_amount': 75000,
                'interest_rate': 0.15,
                'LTV_ratio': 0.95,
                'income': 42000,
            }
        },
    ]

    results = {}
    for app in applicants:
        print(f"\n>>> {app['name']}")
        pred = predict_survival_for_applicant(cph, app['data'], df, output_dir)
        results[app['name']] = pred

    return results


if __name__ == '__main__':
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph
    df = generate_loan_data()
    _, cph = fit_cox_ph(df)
    demo = demonstrate_predictions(df, cph)
    import json; print(json.dumps(demo, indent=2))