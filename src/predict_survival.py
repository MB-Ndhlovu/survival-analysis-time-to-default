"""
Predict survival function for a new loan applicant.
"""
import numpy as np
from lifelines import KaplanMeierFitter


def predict_survival(df, applicant_features):
    """
    Predict survival curve for a new applicant based on similar historical cases.
    
    applicant_features: dict with keys matching df columns
    Returns: predicted survival probabilities at 6, 12, 18, 24, 30, 36 months
    """
    df = df.copy()

    # Simple nearest-neighbor approach: find similar applicants
    credit_score = applicant_features.get('credit_score', 650)
    dti = applicant_features.get('debt_to_income', 0.3)
    ltv = applicant_features.get('LTV_ratio', 0.5)

    # Score distance
    df['dist'] = (
        (df['credit_score'] - credit_score) ** 2 / 10000
        + (df['debt_to_income'] - dti) ** 2 * 10
        + (df['LTV_ratio'] - ltv) ** 2 * 5
    )

    # Use 500 nearest neighbors
    n_neighbors = 500
    similar = df.nsmallest(n_neighbors, 'dist')

    kmf = KaplanMeierFitter()
    kmf.fit(similar['time_end'], event_observed=similar['event_default'])

    # Predictions at key time horizons
    time_horizons = [6, 12, 18, 24, 30, 36]
    predictions = {f'survival_{t}m': kmf.predict(t) for t in time_horizons}
    predictions['median_survival'] = kmf.median_survival_time_

    print("\n=== New Applicant Predicted Survival ===")
    print(f"Applicant: credit_score={credit_score}, DTI={dti:.2f}, LTV={ltv:.2f}")
    for t in time_horizons:
        print(f"  {t}-month survival: {predictions[f'survival_{t}m']:.3f}")
    print(f"  Median survival: {predictions['median_survival']:.1f} months")

    return predictions


if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    applicant = {'credit_score': 700, 'debt_to_income': 0.25, 'LTV_ratio': 0.4}
    preds = predict_survival(df, applicant)