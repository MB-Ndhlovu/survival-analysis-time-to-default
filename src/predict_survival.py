"""Predict survival function for a new loan applicant."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from .cox_ph import fit_cox_ph

def predict_new_applicant(cph_model, applicant_features, timeline=None):
    """
    Predict survival curve for a new applicant using the fitted Cox PH model.
    Args:
        cph_model: fitted CoxPHFitter
        applicant_features: dict of {feature_name: value}
        timeline: array of time points (default 0-36 months)
    """
    if timeline is None:
        timeline = np.arange(0, 37)

    # Standardize features using approximate means/stds from training data
    # These would ideally come from training data; using rough approximations
    feature_means = {
        "credit_score": 680, "employment_years": 4, "debt_to_income": 22,
        "loan_amount": 400000, "interest_rate": 11, "LTV_ratio": 0.55
    }
    feature_stds = {
        "credit_score": 80, "employment_years": 4, "debt_to_income": 15,
        "loan_amount": 350000, "interest_rate": 3, "LTV_ratio": 0.25
    }

    X = {}
    for k, v in applicant_features.items():
        if k in feature_means:
            X[k] = (v - feature_means[k]) / feature_stds[k]
    X["log_loan_amount"] = np.log(applicant_features.get("loan_amount", 400000))

    # Build dataframe for prediction
    pred_df = pd.DataFrame([X])
    pred_df = pred_df.rename(columns={
        "debt_to_income": "debt_to_income",
        "loan_amount": "loan_amount",
    })

    # Get baseline hazard and predicted survival
    survival_probs = cph_model.predict_survival_function(pred_df, times=timeline)

    return timeline, survival_probs.values.flatten()

def demo_predictions(cph_model, df):
    """Show predicted survival for three example applicants."""
    timeline = np.arange(0, 37)

    applicants = {
        "High-Risk (score=540)": {
            "credit_score": 540, "employment_years": 1, "debt_to_income": 40,
            "loan_amount": 800000, "interest_rate": 18, "LTV_ratio": 0.95
        },
        "Medium-Risk (score=680)": {
            "credit_score": 680, "employment_years": 5, "debt_to_income": 25,
            "loan_amount": 500000, "interest_rate": 11, "LTV_ratio": 0.70
        },
        "Low-Risk (score=780)": {
            "credit_score": 780, "employment_years": 10, "debt_to_income": 15,
            "loan_amount": 300000, "interest_rate": 8, "LTV_ratio": 0.45
        },
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"High-Risk (score=540)": "red",
              "Medium-Risk (score=680)": "orange",
              "Low-Risk (score=780)": "green"}

    results = {}
    for name, features in applicants.items():
        t, surv = predict_new_applicant(cph_model, features, timeline)
        ax.plot(t, surv, label=name, color=colors[name], linewidth=2)
        results[name] = {
            "surv_12": np.interp(12, t, surv),
            "surv_24": np.interp(24, t, surv),
        }

    ax.set_title("Predicted Survival Curve by Applicant Profile", fontsize=12)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = "/home/workspace/Projects/survival-analysis-time-to-default/reports/applicant_survival_prediction.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved prediction plot: {out_path}")

    print("\n=== Predicted Survival Probabilities ===")
    for name, vals in results.items():
        print(f"  {name}: 12-mo={vals['surv_12']:.1%}, 24-mo={vals['surv_24']:.1%}")

    return results

if __name__ == "__main__":
    from .data_loader import generate_loan_data
    df = generate_loan_data()
    cph, _ = fit_cox_ph(df)
    demo_predictions(cph, df)