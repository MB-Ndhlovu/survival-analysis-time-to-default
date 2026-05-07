"""Predict survival function for a new loan applicant."""

import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from src.data_loader import generate_loan_data


def train_survival_model(df, time_col='time_end', event_col='event_default'):
    """Train Cox PH model for prediction."""
    df_model = df.copy()

    features = ['credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'LTV_ratio']

    # Standardize features
    for col in features:
        df_model[f'{col}_scaled'] = (df_model[col] - df_model[col].mean()) / df_model[col].std()

    df_model = df_model.dropna(subset=[time_col, event_col])

    scaled_features = [f'{col}_scaled' for col in features]

    cph = CoxPHFitter()
    cph.fit(df_model[scaled_features + [time_col, event_col]],
            duration_col=time_col, event_col=event_col)

    # Also fit a baseline Kaplan-Meier for the overall population
    kmf = KaplanMeierFitter()
    kmf.fit(df_model[time_col], df_model[event_col])

    # Store means/stds for inverse scaling
    normalization = {col: {'mean': df[col].mean(), 'std': df[col].std()} for col in features}

    return {'cph': cph, 'kmf': kmf, 'normalization': normalization, 'features': features}


def predict_survival_for_applicant(model_data, applicant, ax=None):
    """Predict survival curve for a new applicant.

    applicant: dict with keys for credit_score, employment_years, debt_to_income,
               loan_amount, interest_rate, LTV_ratio
    """
    cph = model_data['cph']
    normalization = model_data['normalization']
    features = model_data['features']

    # Scale the applicant data
    scaled_applicant = {}
    for col in features:
        scaled_applicant[f'{col}_scaled'] = (
            (applicant[col] - normalization[col]['mean']) / normalization[col]['std']
        )

    # Create dataframe for prediction
    pred_df = {f'{col}_scaled': [scaled_applicant[f'{col}_scaled']] for col in features}
    pred_df[features[0]] = [applicant[features[0]]]  # placeholder

    # Predict median survival time and hazard
    # For Cox PH, we can compute the predicted hazard/risk score
    X = np.array([[scaled_applicant[f'{col}_scaled'] for col in features]])
    linear_predictor = cph.predict_partial_hazard(X)[0]

    # Predict survival times (median)
    median_pred = cph.predict_median(X)
    if isinstance(median_pred, (float, np.floating)) or (hasattr(median_pred, '__iter__') and not hasattr(median_pred, '__len__')):
        median_pred = float(median_pred) if not np.isinf(median_pred) else None
    elif hasattr(median_pred, '__iter__'):
        median_pred = float(median_pred[0]) if len(median_pred) > 0 and not np.isinf(median_pred[0]) else None
    else:
        median_pred = None

    # Get survival function
    timeline = np.arange(1, 61)
    surv_func = cph.predict_survival_function(X)
    if hasattr(surv_func, 'flatten'):
        baseline_survival = surv_func.flatten()
    else:
        baseline_survival = surv_func.values.flatten()

    result = {
        'linear_predictor': linear_predictor,
        'median_survival_months': median_pred,
        'survival_function': dict(zip(timeline, baseline_survival)),
        'risk_score': linear_predictor,  # Higher = more risk
    }

    return result


def plot_applicant_survival(result, applicant_id='New Applicant', ax=None):
    """Plot survival curve for an applicant."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    timeline = list(result['survival_function'].keys())
    survival = list(result['survival_function'].values())

    ax.plot(timeline, survival, 'b-', linewidth=2, label=f'{applicant_id}')

    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% survival')
    ax.axvline(x=12, color='green', linestyle='--', alpha=0.5, label='12 months')
    ax.axvline(x=24, color='orange', linestyle='--', alpha=0.5, label='24 months')

    # Mark median survival
    median = result['median_survival_months']
    if median:
        median_surv = result['survival_function'].get(int(median), 0.5)
        ax.plot(median, median_surv, 'ro', markersize=10)
        ax.annotate(f'Median: {median:.0f} mo', xy=(median, median_surv),
                   xytext=(median + 5, median_surv + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')

    ax.set_xlabel('Months')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Survival Curve Prediction for {applicant_id}')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 60)

    return ax


if __name__ == '__main__':
    df = generate_loan_data()
    model = train_survival_model(df)

    # Example applicant
    new_applicant = {
        'credit_score': 720,
        'employment_years': 5.0,
        'debt_to_income': 0.25,
        'loan_amount': 250000,
        'interest_rate': 0.105,
        'LTV_ratio': 0.70,
    }

    result = predict_survival_for_applicant(model, new_applicant)

    print("Applicant Prediction:")
    print(f"  Risk Score (linear predictor): {result['risk_score']:.4f}")
    print(f"  Median Survival: {result['median_survival_months']:.1f} months" if result['median_survival_months'] else "  Median Survival: Not reached")
    print(f"  12-month survival: {result['survival_function'].get(12, 'N/A'):.1%}")
    print(f"  24-month survival: {result['survival_function'].get(24, 'N/A'):.1%}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_applicant_survival(result, 'Good Credit Applicant', ax)
    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/applicant_survival.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: reports/applicant_survival.png")