"""
Predict survival function for a new loan applicant.
Uses the fitted Cox PH model to predict individualized survival curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter


def prepare_applicant_features(applicant, income_scaler, loan_scaler, feature_means, feature_stds):
    """
    Prepare a new applicant's features for Cox PH prediction.
    Applies log transforms and standardization matching training data.

    Parameters
    ----------
    applicant : dict
        Raw applicant features with keys:
        - income, credit_score, employment_years,
        - debt_to_income, loan_amount, interest_rate, LTV_ratio
    income_scaler, loan_scaler : float
        Scalers used during training (means of log-transformed vars)
    feature_means, feature_stds : dict
        Means and stds of each standardized feature

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with processed features in correct format.
    """
    features = {
        "credit_score": applicant["credit_score"],
        "employment_years": applicant["employment_years"],
        "debt_to_income": applicant["debt_to_income"],
        "interest_rate": applicant["interest_rate"],
        "LTV_ratio": applicant["LTV_ratio"],
        "log_income": np.log1p(applicant["income"]),
        "log_loan_amount": np.log1p(applicant["loan_amount"]),
    }

    standardize_cols = ["credit_score", "employment_years", "debt_to_income",
                         "interest_rate", "LTV_ratio", "log_income", "log_loan_amount"]

    X = pd.DataFrame([features])
    for col in standardize_cols:
        X[col] = (X[col] - feature_means[col]) / feature_stds[col]

    return X


def predict_survival_curve(cph, X_new, timeline=None):
    """
    Predict conditional survival function for a new applicant.
    Returns survival probability at each time point.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox PH model
    X_new : pd.DataFrame
        Single-row processed applicant features
    timeline : array, optional
        Time points at which to evaluate survival. Defaults to 0-60 months.

    Returns
    -------
    pd.DataFrame
        DataFrame with timeline and survival probability columns.
    """
    if timeline is None:
        timeline = np.arange(0, 61)

    # Predict hazard ratios for this individual
    log_hazard_ratio = cph.predict_log_partial_hazard(X_new).values[0]
    baseline_hazard = cph.baseline_hazard_

    # Get baseline survival from the model's baseline survival
    # We'll compute survival using the cumulative baseline hazard
    baseline_cumulative_hazard = cph.baseline_cumulative_hazard_
    baseline_survival = np.exp(-baseline_cumulative_hazard["baseline_cumulative_hazard"].values)

    # Expand baseline to match timeline
    times = baseline_cumulative_hazard.index.values
    survival_at_times = np.exp(-baseline_cumulative_hazard["baseline_cumulative_hazard"].values)

    # Interpolate to requested timeline
    from scipy import interpolate
    interp_func = interpolate.interp1d(times, survival_at_times, kind="linear",
                                        fill_value="extrapolate", bounds_error=False)
    survival_curve = np.exp(log_hazard_ratio * interp_func(timeline))

    result = pd.DataFrame({
        "timeline": timeline,
        "survival_probability": np.clip(survival_curve, 0, 1)
    })
    return result


def predict_survival_bands(X_new, cph, timeline=None):
    """
    Predict survival with confidence bands using individual prediction.
    Simpler version that directly computes from baseline survival.

    Parameters
----------
    X_new : pd.DataFrame
        Processed applicant features
    cph : CoxPHFitter
        Fitted model
    timeline : array, optional
        Time points

    Returns
    -------
    pd.DataFrame
        Timeline + survival + lower/upper confidence bounds
    """
    if timeline is None:
        timeline = np.arange(0, 61)

    # Get the predicted survival function
    surv_func = cph.predict_survival_function(X_new, times=timeline)

    # For a single individual, use the point estimate
    result = pd.DataFrame({
        "timeline": timeline,
        "survival_probability": surv_func.values.ravel()
    })
    return result


def plot_applicant_survival(survival_df, applicant_info=None, save_path=None):
    """
    Plot a new applicant's predicted survival curve.

    Parameters
    ----------
    survival_df : pd.DataFrame
        Output from predict_survival_bands
    applicant_info : dict, optional
        Dict of applicant characteristics to display in title
    save_path : str, optional
        Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.step(survival_df["timeline"], survival_df["survival_probability"],
            where="post", linewidth=2.5, color="#2ecc71")

    # Reference lines at 12 and 24 months
    s12 = survival_df[survival_df["timeline"] == 12]["survival_probability"].values[0]
    s24 = survival_df[survival_df["timeline"] == 24]["survival_probability"].values[0]

    ax.axhline(y=s12, color="orange", linestyle="--", alpha=0.7, linewidth=1.2)
    ax.axhline(y=s24, color="red", linestyle="--", alpha=0.7, linewidth=1.2)
    ax.text(50, s12 + 0.02, f"12m survival: {s12:.1%}", color="orange", fontsize=10)
    ax.text(50, s24 - 0.04, f"24m survival: {s24:.1%}", color="red", fontsize=10)

    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)

    if applicant_info:
        info_str = " | ".join([f"{k}: {v}" for k, v in applicant_info.items()])
        ax.set_title(f"Predicted Survival Curve for New Applicant\n{info_str}",
                     fontsize=12, fontweight="bold")
    else:
        ax.set_title("Predicted Survival Curve for New Applicant", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved applicant survival plot to {save_path}")
    plt.close()


def new_applicant_example(cph, feature_means, feature_stds):
    """
    Create a new applicant example and predict their survival.

    Returns
    -------
    dict
        Applicant info and predicted survival DataFrame
    """
    applicant = {
        "income": 180000,
        "credit_score": 690,
        "employment_years": 4,
        "debt_to_income": 0.30,
        "loan_amount": 320000,
        "interest_rate": 0.11,
        "LTV_ratio": 0.78,
    }

    X_new = prepare_applicant_features(
        applicant, 0, 0, feature_means, feature_stds
    )
    # Use feature means/stds from training (we'll compute from a reference df)
    survival = predict_survival_bands(X_new, cph)

    return applicant, survival


if __name__ == "__main__":
    from .data_loader import generate_loan_data
    from .cox_ph import fit_cox_ph

    df = generate_loan_data()
    cph, X = fit_cox_ph(df)

    feature_means = X.mean().to_dict()
    feature_stds = X.std().to_dict()

    applicant = {
        "income": 180000,
        "credit_score": 690,
        "employment_years": 4,
        "debt_to_income": 0.30,
        "loan_amount": 320000,
        "interest_rate": 0.11,
        "LTV_ratio": 0.78,
    }

    X_new = prepare_applicant_features(applicant, 0, 0, feature_means, feature_stds)
    survival = predict_survival_bands(X_new, cph)

    print("New Applicant:")
    for k, v in applicant.items():
        print(f"  {k}: {v}")
    print(f"\n  Predicted 12m survival: {survival[survival['timeline']==12]['survival_probability'].values[0]:.1%}")
    print(f"  Predicted 24m survival: {survival[survival['timeline']==24]['survival_probability'].values[0]:.1%}")

    plot_applicant_survival(survival, applicant)