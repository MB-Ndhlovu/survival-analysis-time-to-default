"""Predict survival function for a new loan applicant."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def predict_survival_for_applicant(
    cph: CoxPHFitter,
    applicant: dict,
    times: list = None,
) -> dict:
    """
    Predict survival function for a new applicant.

    Args:
        cph: Fitted CoxPHFitter
        applicant: dict with feature values
        times: list of time points to predict at

    Returns dict with times and survival probabilities
    """
    if times is None:
        times = list(range(1, 61))  # 1 to 60 months

    features = [
        "credit_score",
        "income",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    # Build DataFrame for prediction
    row = {f: applicant.get(f, 0) for f in features}
    row["credit_score"] = row["credit_score"] / 100  # Scale like training data
    pred_df = pd.DataFrame([row])

    # Predict survival function
    surv_func = cph.predict_survival_function(pred_df, times)

    return {
        "times": times,
        "survival_probabilities": surv_func.values.flatten().tolist(),
    }


def print_prediction_results(prediction: dict, applicant: dict) -> str:
    """Format prediction results for display."""
    times = prediction["times"]
    surv_probs = prediction["survival_probabilities"]

    lines = ["\n=== Survival Prediction for New Applicant ===\n"]
    lines.append("Applicant profile:")
    for key, val in applicant.items():
        lines.append(f"  {key}: {val}")
    lines.append("\nPredicted survival probabilities:")
    lines.append(f"{'Month':>8} {'Survival Prob':>14}\n" + "-" * 24)

    # Key time points
    key_times = [6, 12, 18, 24, 36, 48, 60]
    for t in key_times:
        if t in times:
            idx = times.index(t)
            lines.append(f"{t:>8} {surv_probs[idx]:>14.1%}")

    # Find median survival time
    median_idx = next((i for i, p in enumerate(surv_probs) if p <= 0.5), None)
    if median_idx:
        lines.append(f"\nEstimated median survival time: {times[median_idx]} months")
    else:
        lines.append("\nEstimated median survival time: Not reached within 60 months")

    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph_model

    df = generate_loan_data()
    results = fit_cox_ph_model(df)
    cph = results["cph"]

    new_applicant = {
        "credit_score": 650,
        "income": 55.0,
        "employment_years": 3.5,
        "debt_to_income": 0.40,
        "loan_amount": 75.0,
        "interest_rate": 8.5,
        "LTV_ratio": 0.85,
    }

    prediction = predict_survival_for_applicant(cph, new_applicant)
    print(print_prediction_results(prediction, new_applicant))