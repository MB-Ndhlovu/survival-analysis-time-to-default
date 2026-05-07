"""
Predict survival function for a new loan applicant.
Uses fitted Cox PH model to generate time-to-default predictions.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def predict_survival_for_applicant(cph, applicant, t_values=None):
    """
    Predict survival curve for a new applicant using fitted Cox model.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox proportional hazards model
    applicant : dict
        Dictionary of applicant features:
        {
            'credit_score': int,
            'employment_years': float,
            'debt_to_income': float,
            'loan_amount': float,
            'interest_rate': float,
            'LTV_ratio': float,
        }
    t_values : array, optional
        Time points at which to predict survival. Defaults to range(0, 37, 1)

    Returns
    -------
    pd.DataFrame
        DataFrame with time and predicted_survival columns
    """
    if t_values is None:
        t_values = list(range(0, 37))

    # Build feature vector in correct order
    feature_cols = [
        'credit_score',
        'employment_years',
        'debt_to_income',
        'interest_rate',
        'LTV_ratio',
        'log_loan_amount',
    ]

    # Compute derived features
    applicant_features = {
        'credit_score': applicant['credit_score'],
        'employment_years': applicant['employment_years'],
        'debt_to_income': applicant['debt_to_income'],
        'interest_rate': applicant['interest_rate'],
        'LTV_ratio': applicant['LTV_ratio'],
        'log_loan_amount': np.log(applicant['loan_amount'] + 1),
    }

    # Create DataFrame for prediction
    X = pd.DataFrame([applicant_features])[feature_cols]

    # Predict median survival and hazard
    # The Cox model gives us log(hazard) = X @ beta
    # Survival at time t = exp(-H(t)) where H(t) = base_hazard_cumulative * exp(X @ beta)

    # For simplicity, use the predict_expectation which gives E[T]
    # and predict_percentile for survival curve

    # Get survival function via predicting conditional median
    # Use the conditional_after parameter for flexible time predictions

    predictions = []
    for t in t_values:
        try:
            # Calculate survival at time t using baseline and applicant hazard
            # S(t) = exp(-H_0(t) * exp(Xβ))
            # We approximate this using the model's baseline cumulative hazard

            # Get baseline cumulative hazard at time t
            baseline_hazard = cph.baseline_cumulative_hazard_
            if len(baseline_hazard) > 0:
                # Interpolate to time t
                times = baseline_hazard.index.values
                if t <= times.max():
                    idx = np.searchsorted(times, t)
                    if idx >= len(baseline_hazard):
                        idx = len(baseline_hazard) - 1
                    H0_t = baseline_hazard.iloc[idx, 0]
                else:
                    H0_t = baseline_hazard.iloc[-1, 0]

                # Calculate linear predictor
                X_beta = sum(
                    applicant_features[col] * cph.params_.get(col, 0)
                    for col in feature_cols if col in cph.params_
                )

                # Survival probability
                survival_prob = np.exp(-H0_t * np.exp(X_beta))
            else:
                survival_prob = np.nan

            predictions.append({
                'time_months': t,
                'predicted_survival': round(survival_prob, 4) if not np.isnan(survival_prob) else None
            })
        except Exception as e:
            predictions.append({
                'time_months': t,
                'predicted_survival': None
            })

    return pd.DataFrame(predictions)


def expected_default_month(cph, applicant, threshold=0.5):
    """
    Predict the expected month of default (when survival drops below threshold).

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox model
    applicant : dict
        Applicant features
    threshold : float
        Survival probability threshold (default 0.5 = median)

    Returns
    -------
    float
        Estimated months until default, or None if survival stays above threshold
    """
    t_values = list(range(0, 37))
    survival_df = predict_survival_for_applicant(cph, applicant, t_values)

    below_threshold = survival_df[survival_df['predicted_survival'] <= threshold]

    if len(below_threshold) > 0:
        return below_threshold['time_months'].iloc[0]
    else:
        return None


def risk_segment_applicant(applicant):
    """
    Classify an applicant into risk segment based on credit score.

    Returns
    -------
    dict with segment name, expected 12m and 24m survival
    """
    score = applicant['credit_score']

    if score < 580:
        segment = 'Deep Subprime'
        base_surv_12 = 0.65
        base_surv_24 = 0.45
    elif score < 670:
        segment = 'Subprime'
        base_surv_12 = 0.78
        base_surv_24 = 0.60
    elif score < 740:
        segment = 'Near Prime'
        base_surv_12 = 0.88
        base_surv_24 = 0.75
    else:
        segment = 'Prime'
        base_surv_12 = 0.93
        base_surv_24 = 0.85

    return {
        'segment': segment,
        'credit_score_band': f'{segment} ({score})',
        'base_survival_12m': base_surv_12,
        'base_survival_24m': base_surv_24,
    }


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_band
    from cox_ph import fit_cox_ph

    df = generate_loan_data(5000)
    df = add_credit_band(df)
    cph = fit_cox_ph(df)

    # Example applicant
    new_applicant = {
        'credit_score': 650,
        'employment_years': 3.5,
        'debt_to_income': 0.32,
        'loan_amount': 350000,
        'interest_rate': 0.18,
        'LTV_ratio': 0.75,
    }

    print("New Applicant Profile:")
    for k, v in new_applicant.items():
        print(f"  {k}: {v}")

    print("\nRisk Segment:", risk_segment_applicant(new_applicant)['segment'])

    survival_curve = predict_survival_for_applicant(cph, new_applicant)
    print("\nPredicted Survival Curve:")
    print(survival_curve[survival_curve['time_months'].isin([6, 12, 18, 24])].to_string(index=False))

    exp_default = expected_default_month(cph, new_applicant)
    print(f"\nExpected Default Month (50% survival): {exp_default}")