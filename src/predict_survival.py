"""Predict survival function for a new loan applicant."""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

def predict_survival_new_applicant(cph, applicant, time_points=None):
    """Predict survival curve for a new applicant using fitted Cox PH model.

    Args:
        cph: Fitted CoxPHFitter
        applicant: dict with feature values (income, credit_score, etc.)
        time_points: Array of time points to predict at (default: 1-36 months)

    Returns:
        dict with predicted survival probabilities at each time point
    """
    if time_points is None:
        time_points = np.arange(1, 37)

    features = ['income', 'credit_score', 'employment_years',
                'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    # Get training data means and stds from Cox PH (stored in fitted model)
    train_means = {}
    train_stds = {}

    # Reconstruct from model's training data summary
    for f in features:
        train_means[f] = cph.baseline_hazard_.index.get_level_values(f).mean()
        train_stds[f] = cph.baseline_hazard_.index.get_level_values(f).std()

    # Normalize applicant features
    normalized = {}
    for f in features:
        val = applicant.get(f, train_means[f])
        normalized[f] = (val - train_means[f]) / train_stds[f]

    # Create input dataframe
    X = pd.DataFrame([normalized])

    # Get baseline survival
    baseline_survival = cph.baseline_survival_

    # Calculate predicted survival using Cox PH formula
    # S(t|x) = S0(t)^exp(x*beta)
    linear_predictor = sum(X.iloc[0][f] * cph.params_[f] for f in features if f in cph.params_)

    predictions = {}
    for t in time_points:
        baseline_haz = baseline_survival.loc[t, 'baseline_survival'] if t in baseline_survival.index else baseline_survival.iloc[-1, 0]
        predictions[int(t)] = baseline_haz ** np.exp(linear_predictor)

    return predictions

def predict_with_kmf(applicant_closest, kmf_by_credit_band):
    """Predict survival using nearest credit band's Kaplan-Meier curve."""
    import sys
    sys.path.insert(0, '/home/workspace/Projects/survival-analysis-time-to-default/src')
    from kaplan_meier import get_credit_band

    band = get_credit_band(applicant_closest['credit_score'])
    kmf = kmf_by_credit_band[band]['kmf']

    time_points = np.arange(1, 37)
    survival_probs = []

    for t in time_points:
        sf = kmf.survival_function_
        if len(sf) == 0:
            survival_probs.append(0.5)
        else:
            nearest_idx = sf.index.get_indexer([t], method='ffill')[0]
            nearest_idx = max(0, min(nearest_idx, len(sf) - 1))
            prob = float(sf.iloc[nearest_idx].values[0])
            survival_probs.append(prob)

    return {
        'band': band,
        'time_points': list(time_points),
        'survival_probabilities': survival_probs
    }

def new_applicant_example():
    """Example new applicant profile for survival prediction."""
    return {
        'income': 75,  # $75k annual income
        'credit_score': 720,
        'employment_years': 4.5,
        'debt_to_income': 0.28,
        'loan_amount': 120,  # $120k loan
        'interest_rate': 0.065,
        'LTV_ratio': 0.75
    }

if __name__ == "__main__":
    from data_loader import generate_loan_data
    from kaplan_meier import fit_kaplan_meier, get_credit_band
    from cox_ph import fit_cox_ph_model

    df = generate_loan_data()
    km_results = fit_kaplan_meier(df)
    cph = fit_cox_ph_model(df)

    applicant = new_applicant_example()
    print(f"Applicant: Credit Score {applicant['credit_score']} ({get_credit_band(applicant['credit_score'])})")

    # Using Cox PH
    print("\nUsing Cox PH model:")
    predictions = predict_survival_new_applicant(cph, applicant)
    for t in [12, 24, 36]:
        print(f"  {t}-month survival probability: {predictions.get(t, 0):.1%}")

    # Using Kaplan-Meier band
    print("\nUsing Kaplan-Meier band:")
    km_pred = predict_with_kmf(applicant, km_results)
    for t in [12, 24, 36]:
        idx = t - 1
        if idx < len(km_pred['survival_probabilities']):
            print(f"  {t}-month survival probability: {km_pred['survival_probabilities'][idx]:.1%}")