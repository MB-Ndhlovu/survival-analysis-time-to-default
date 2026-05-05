import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

def predict_survival(new_applicant, cph, df_train, duration_col='time_end', event_col='event_default'):
    """Predict survival function for a new loan applicant."""
    features = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'ltv_ratio']

    # Prepare applicant data
    applicant = pd.DataFrame([new_applicant])
    applicant['log_income'] = np.log(applicant['income'])
    applicant['log_loan_amount'] = np.log(applicant['loan_amount'])

    features_model = ['log_income', 'credit_score', 'employment_years', 'debt_to_income',
                      'log_loan_amount', 'interest_rate', 'ltv_ratio']

    # Predict median survival for this applicant profile
    baseline_km = KaplanMeierFitter()
    baseline_km.fit(df_train[duration_col], df_train[event_col])

    # Use Cox PH to get conditional survival
    # We approximate by finding similar profiles in training data
    sim_mask = (
        (df_train['credit_score'] >= new_applicant['credit_score'] - 50) &
        (df_train['credit_score'] <= new_applicant['credit_score'] + 50) &
        (df_train['ltv_ratio'] >= new_applicant['ltv_ratio'] - 0.1) &
        (df_train['ltv_ratio'] <= new_applicant['ltv_ratio'] + 0.1)
    )
    sim_df = df_train[sim_mask]

    if len(sim_df) > 50:
        km_sim = KaplanMeierFitter()
        km_sim.fit(sim_df[duration_col], sim_df[event_col])
        survival_12m = km_sim.predict(12)
        survival_24m = km_sim.predict(24)
        median_survival = km_sim.median_survival_time_ if not np.isinf(km_sim.median_survival_time_) else None
    else:
        survival_12m = baseline_km.predict(12)
        survival_24m = baseline_km.predict(24)
        median_survival = baseline_km.median_survival_time_ if not np.isinf(baseline_km.median_survival_time_) else None

    # Build time points for survival curve
    time_points = np.arange(0.5, 25, 0.5)
    survival_probs = []

    for t in time_points:
        if len(sim_df) > 50:
            survival_probs.append(km_sim.predict(t))
        else:
            survival_probs.append(baseline_km.predict(t))

    return {
        'survival_curve': dict(zip(time_points.round(1), [round(p, 4) for p in survival_probs])),
        'survival_12m': survival_12m,
        'survival_24m': survival_24m,
        'median_survival_months': median_survival,
        'similar_profiles_n': len(sim_df)
    }

def print_prediction(prediction, applicant):
    """Print prediction results."""
    print("\n=== New Applicant Prediction ===")
    print(f"Credit Score: {applicant['credit_score']}")
    print(f"Income: ${applicant['income']:,.0f}")
    print(f"LTV: {applicant['ltv_ratio']:.2%}")
    print(f"DTI: {applicant['debt_to_income']:.2%}")
    print(f"\n12-Month Survival: {prediction['survival_12m']:.1%}")
    print(f"24-Month Survival: {prediction['survival_24m']:.1%}")
    median_str = f"{prediction['median_survival_months']:.1f} months" if prediction['median_survival_months'] else "Not reached"
    print(f"Median Survival: {median_str}")
    print(f"\nBased on {prediction['similar_profiles_n']} similar profiles in training data")

if __name__ == "__main__":
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph

    df = generate_loan_data()
    results, cph = fit_cox_ph(df)

    new_applicant = {
        'income': 65000,
        'credit_score': 720,
        'employment_years': 7,
        'debt_to_income': 0.28,
        'loan_amount': 180000,
        'interest_rate': 7.5,
        'ltv_ratio': 0.75
    }

    prediction = predict_survival(new_applicant, cph, df)
    print_prediction(prediction, new_applicant)