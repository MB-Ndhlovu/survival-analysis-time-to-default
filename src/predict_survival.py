"""
Predict survival function for a new loan applicant.
Uses Kaplan-Meier for predictions after training.
"""

import numpy as np
import pandas as pd


def train_cox_model(df):
    """Get feature means and stds for normalization."""
    feature_cols = [
        'income', 'credit_score', 'employment_years',
        'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio'
    ]

    means = df[feature_cols].mean()
    stds = df[feature_cols].std()

    return means, stds


def predict_survival(means, stds, applicant, df, horizon=24):
    """
    Predict survival using stratified KM based on similar applicants.
    """
    features = ['income', 'credit_score', 'employment_years',
               'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Normalize applicant features
    normalized = {}
    for f in features:
        normalized[f] = (applicant[f] - means[f]) / stds[f]

    # Calculate distance to all applicants (Euclidean on normalized features)
    distances = np.zeros(len(df))
    for f in features:
        distances += (df[f].values - normalized[f]) ** 2
    distances = np.sqrt(distances)

    # Find k nearest neighbors
    k = min(500, len(df))
    nearest_idx = np.argsort(distances)[:k]
    subset = df.iloc[nearest_idx]

    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(durations=subset['time_end'], event_observed=subset['event_default'])

    survival_probs = {}
    for month in range(1, horizon + 1):
        survival_probs[month] = kmf.predict(month)

    timeline = np.arange(1, horizon + 1)
    survival_values = [survival_probs[t] for t in timeline]

    return {
        'survival_probs': survival_probs,
        'timeline': timeline.tolist(),
        '12m_survival': float(survival_probs[12]),
        '24m_survival': float(survival_probs[24]),
        'expected_months': float(sum(survival_values) / len(survival_values))
    }


def print_applicant_score(prediction, applicant):
    print("\n" + "="*50)
    print("NEW APPLICANT SURVIVAL PREDICTION")
    print("="*50)
    print(f"{'Credit Score':<20} {applicant.get('credit_score', 'N/A')}")
    print(f"{'Income':<20} ${applicant.get('income', 0):,.0f}")
    print(f"{'DTI':<20} {applicant.get('debt_to_income', 0):.3f}")
    print(f"{'LTV':<20} {applicant.get('LTV_ratio', 0):.3f}")
    print("-"*50)
    print(f"{'12-Month Survival':<20} {prediction['12m_survival']*100:.1f}%")
    print(f"{'24-Month Survival':<20} {prediction['24m_survival']*100:.1f}%")
    print(f"{'Expected Survival':<20} {prediction['expected_months']:.1f} months")
    print("="*50)


if __name__ == '__main__':
    from data_loader import generate_loan_data

    df = generate_loan_data()
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