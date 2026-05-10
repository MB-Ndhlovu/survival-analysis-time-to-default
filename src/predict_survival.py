"""Predict survival function for a new loan applicant."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


def predict_survival_for_applicant(
    cox_model: CoxPHFitter,
    applicant: dict,
    baseline_data: pd.DataFrame,
    times: list = None,
) -> dict:
    """
    Predict survival curve for a new applicant using the fitted Cox PH model.
    """
    if times is None:
        times = [6, 12, 18, 24, 36, 48, 60]

    features = [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    # Scale applicant features using baseline stats
    scaled_applicant = {}
    for feat in features:
        mean = baseline_data[feat].mean()
        std = baseline_data[feat].std()
        scaled_applicant[feat] = (applicant[feat] - mean) / std

    # Get Cox model coefficients
    coef_names = list(cox_model.params_.index)
    coef_vals = np.array(list(cox_model.params_.values))
    x_cov = np.array([scaled_applicant[c] for c in coef_names])
    linear_predictor = float(np.dot(x_cov, coef_vals))

    # Use Cox model's built-in predict_survival_function
    X_pred = pd.DataFrame([scaled_applicant])
    try:
        surv_func = cox_model.predict_survival_function(X_pred)
        survival_predictions = {}
        for t in times:
            if t in surv_func.index:
                survival_predictions[f"month_{t}"] = round(float(surv_func.loc[t].values[0]), 4)
            else:
                survival_predictions[f"month_{t}"] = None
    except Exception:
        survival_predictions = {f"month_{t}": None for t in times}

    # Build KM curve for similar applicants
    similar_mask = (
        (baseline_data["credit_score"] >= applicant["credit_score"] - 20) &
        (baseline_data["credit_score"] <= applicant["credit_score"] + 20) &
        (baseline_data["debt_to_income"] >= applicant["debt_to_income"] - 0.05) &
        (baseline_data["debt_to_income"] <= applicant["debt_to_income"] + 0.05)
    )
    similar = baseline_data[similar_mask]

    km_predictions = None
    if len(similar) > 30:
        kmf = KaplanMeierFitter()
        kmf.fit(similar["time_end"], event_observed=similar["event_default"])
        km_predictions = {}
        for t in times:
            s = kmf.survival_function_at_times(t).values
            km_predictions[f"month_{t}"] = round(float(s[0]) if len(s) > 0 else np.nan, 4)

    return {
        "cox_ph_predictions": survival_predictions,
        "km_similar_applicants": km_predictions,
        "linear_predictor": round(linear_predictor, 4),
        "n_similar_applicants": len(similar),
    }


def demo_prediction():
    """Demonstrate survival prediction for a sample applicant."""
    from src.data_loader import load_data

    df = load_data()

    cph = CoxPHFitter()
    features = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_amount", "interest_rate", "LTV_ratio",
    ]
    X = df[features].copy()
    X = (X - X.mean()) / X.std()
    cph.fit(
        pd.concat([df["time_end"], df["event_default"]], axis=1).rename(
            columns={"time_end": "duration", "event_default": "event"}
        ),
        duration_col="duration",
        event_col="event",
    )

    new_applicant = {
        "income": 85_000,
        "credit_score": 670,
        "employment_years": 3.5,
        "debt_to_income": 0.28,
        "loan_amount": 450_000,
        "interest_rate": 0.105,
        "LTV_ratio": 0.72,
    }

    result = predict_survival_for_applicant(cph, new_applicant, df)
    return result


if __name__ == "__main__":
    result = demo_prediction()
    print("\n=== New Applicant Survival Predictions ===")
    print(f"Similar applicants in database: {result['n_similar_applicants']}")
    print("\nCox PH predictions:")
    for k, v in result["cox_ph_predictions"].items():
        print(f"  {k}: {v}")
    if result["km_similar_applicants"]:
        print("\nKM (similar applicants) predictions:")
        for k, v in result["km_similar_applicants"].items():
            print(f"  {k}: {v}")