"""
Predict survival function for a new loan applicant.
Uses Kaplan-Meier from applicant's credit band for simplified prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def create_applicant_profile(credit_score=650, income=300000, employment_years=3,
                             debt_to_income=0.25, loan_amount=500000,
                             interest_rate=0.14, LTV_ratio=0.75):
    """Create a dictionary representing a new applicant."""
    return {
        'credit_score': credit_score,
        'income': income,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': LTV_ratio,
    }


def get_credit_band(credit_score):
    """Return credit band for a score."""
    if credit_score < 580:
        return 'Very Poor'
    elif credit_score < 670:
        return 'Fair'
    elif credit_score < 740:
        return 'Good'
    else:
        return 'Excellent'


def predict_survival(df, applicant, ax=None):
    """Predict survival curve for a new applicant based on their credit band."""
    credit_score = applicant['credit_score']
    band = get_credit_band(credit_score)
    band_df = df[df['credit_band'] == band]

    kmf = KaplanMeierFitter()
    kmf.fit(band_df['time_end'], band_df['event_default'])

    timeline = np.array([1, 6, 12, 18, 24, 30, 36])
    surv_probs = np.array([kmf.predict(t) for t in timeline])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(timeline, surv_probs, 'b-', linewidth=2.5, label='Predicted Survival', where='post')
    ax.axvline(x=12, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=24, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Months', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Predicted Survival Curve\nCredit Score: {credit_score} ({band})\n'
                 f'Income: R{applicant["income"]:,.0f}, DTI: {applicant["debt_to_income"]:.0%}', fontsize=11)
    ax.legend(loc='lower left')
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    prob_12 = surv_probs[2]
    prob_24 = surv_probs[4]

    ax.plot(12, prob_12, 'ro', markersize=10)
    ax.plot(24, prob_24, 'ro', markersize=10)
    ax.annotate(f'12-mo: {prob_12:.1%}', xy=(12, prob_12), xytext=(14, prob_12-0.08), fontsize=9)
    ax.annotate(f'24-mo: {prob_24:.1%}', xy=(24, prob_24), xytext=(26, prob_24-0.08), fontsize=9)

    # Find median survival
    surv_function = kmf.survival_function_
    median_idx = np.where(surv_function.values <= 0.5)[0]
    median_surv = surv_function.index[median_idx[0]] if len(median_idx) > 0 else 'Not reached'

    predicted = {
        'survival_at_12': float(prob_12),
        'survival_at_24': float(prob_24),
        'median_survival': float(median_surv) if isinstance(median_surv, (int, float)) else median_surv
    }

    return predicted, kmf


def compare_to_all_bands(df, applicant, ax):
    """Compare applicant's band to all other bands."""
    bands = ['Very Poor', 'Fair', 'Good', 'Excellent']
    colors = {'Very Poor': '#d62728', 'Fair': '#ff7f0e', 'Good': '#2ca02c', 'Excellent': '#1f77b4'}

    band = get_credit_band(applicant['credit_score'])

    for b in bands:
        band_df = df[df['credit_band'] == b]
        kmf = KaplanMeierFitter()
        kmf.fit(band_df['time_end'], band_df['event_default'])
        timeline = np.array([12, 18, 24])
        surv = kmf.predict(timeline)

        style = '-' if b == band else '--'
        lw = 2.5 if b == band else 1.5
        ax.step(timeline, surv, style, color=colors[b], linewidth=lw,
                label=f'{b} (n={len(band_df)})', where='post')

    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.05)


def demo_prediction(df):
    """Demonstrate prediction for a sample applicant."""
    print("\n" + "="*70)
    print("NEW APPLICANT SURVIVAL PREDICTION")
    print("="*70)

    applicants = [
        create_applicant_profile(credit_score=520, income=150000, debt_to_income=0.4,
                                  loan_amount=400000, interest_rate=0.18, LTV_ratio=0.9),
        create_applicant_profile(credit_score=680, income=350000, debt_to_income=0.2,
                                  loan_amount=600000, interest_rate=0.11, LTV_ratio=0.65),
        create_applicant_profile(credit_score=780, income=800000, debt_to_income=0.15,
                                  loan_amount=1200000, interest_rate=0.08, LTV_ratio=0.55),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    predictions = []
    for i, (applicant, ax) in enumerate(zip(applicants, axes)):
        pred, _ = predict_survival(df, applicant, ax)
        compare_to_all_bands(df, applicant, ax)
        predictions.append(pred)

        band = get_credit_band(applicant['credit_score'])
        print(f"\nApplicant {i+1} ({band}):")
        print(f"  Credit Score: {applicant['credit_score']}")
        print(f"  Income: R{applicant['income']:,.0f}")
        print(f"  DTI: {applicant['debt_to_income']:.1%}")
        print(f"  12-Month Survival: {pred['survival_at_12']:.1%}")
        print(f"  24-Month Survival: {pred['survival_at_24']:.1%}")
        print(f"  Median Survival: {pred['median_survival']}")

    plt.suptitle('Survival Predictions by Credit Band', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/applicant_predictions.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    return predictions


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    predictions = demo_prediction(df)
    print("\nPlot saved to reports/applicant_predictions.png")