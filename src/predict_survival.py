"""Predict survival function for a new loan applicant using Cox PH with correct baseline application."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def predict_survival(cph, new_applicant, df_train, timeline=None):
    """
    Predict survival curve for a new applicant.
    Applies the Cox PH linear predictor to the baseline survival.
    S(t|x) = [S_0(t)] ^ exp(beta'x)
    """
    if timeline is None:
        timeline = np.arange(1, 61)

    features = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'LTV_ratio']

    row = {}
    for feat in features:
        row[feat] = new_applicant.get(feat, 0)

    # Normalize using training statistics (same as in fit_cox_ph)
    X = pd.DataFrame([row])
    for col in features:
        mean_val = df_train[col].mean()
        std_val = df_train[col].std()
        X[col] = (X[col] - mean_val) / std_val if std_val > 0 else 0

    X['log_loan_amount'] = np.log(row['loan_amount'] + 1)

    # Compute linear predictor: sum(beta_i * x_i)
    coefs = cph.params_
    lp = 0.0
    for col in X.columns:
        lp += coefs[col] * X[col].values[0]

    # Get baseline survival function
    baseline_surv = cph.baseline_survival_
    bs_times = baseline_surv.index.values
    bs_surv = baseline_surv.values.flatten()

    # Interpolate baseline to requested timeline
    interp_bs = interp1d(bs_times, bs_surv, kind='linear',
                         bounds_error=False, fill_value=(1.0, bs_surv[-1]))

    # S(t) = [S_0(t)] ^ exp(lp)
    baseline_at_t = interp_bs(timeline)
    surv_prob = np.power(baseline_at_t, np.exp(lp))

    result = {
        'timeline': timeline.tolist(),
        'survival_probability': surv_prob.tolist(),
        'new_applicant': new_applicant,
        'linear_predictor': float(lp),
    }

    return result

def print_applicant_survival(pred, t_values=None):
    """Print survival probabilities at key time points."""
    if t_values is None:
        t_values = [6, 12, 18, 24, 36, 48, 60]

    print("\n=== New Applicant Predicted Survival ===")
    app = pred['new_applicant']
    print(f"Credit Score: {app['credit_score']}, Income: {app['income']:.0f}, "
          f"DTI: {app['debt_to_income']:.2f}, LTV: {app['LTV_ratio']:.2f}, "
          f"Interest Rate: {app['interest_rate']:.1%}")
    print(f"Linear predictor (log HR): {pred['linear_predictor']:.4f}")

    print(f"\n{'Month':<8} {'Survival Prob':>14}")
    print("-" * 25)
    for t in t_values:
        t_idx = t - 1
        if t_idx < len(pred['survival_probability']):
            prob = pred['survival_probability'][t_idx]
            print(f"{t:<8} {prob:>14.1%}")

    return pred