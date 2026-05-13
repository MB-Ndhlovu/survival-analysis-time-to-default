"""Predict survival function for new loan applicants."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .data_loader import get_credit_score_band


def create_applicant(
    credit_score: int = 680,
    employment_years: float = 3.0,
    debt_to_income: float = 0.25,
    loan_amount: float = 150000,
    interest_rate: float = 0.075,
    LTV_ratio: float = 0.75,
) -> pd.DataFrame:
    """Create a new applicant record."""
    return pd.DataFrame([{
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "LTV_ratio": LTV_ratio,
    }])


def prepare_applicant_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare applicant DataFrame for Cox PH prediction."""
    df = df.copy()

    # Add credit band indicators (must match training model)
    df["credit_band"] = df["credit_score"].apply(get_credit_score_band)
    df["is_subprime"] = (df["credit_band"] == "Deep Subprime (< 580)").astype(int)
    df["is_near_prime"] = (df["credit_band"] == "Near Prime (670-739)").astype(int)
    df["is_prime"] = (df["credit_band"] == "Prime (740+)").astype(int)

    # Drop credit_score to avoid multicollinearity
    feature_cols = [
        "is_subprime",
        "is_near_prime",
        "is_prime",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "LTV_ratio",
    ]

    return df[feature_cols]


def predict_survival(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    times: np.ndarray = None,
) -> pd.DataFrame:
    """Predict survival function for applicants.

    Args:
        cph: Fitted Cox PH model
        df: Applicant DataFrame (single or multiple rows)
        times: Time points to predict at

    Returns:
        DataFrame with survival probabilities at each time point
    """
    if times is None:
        times = np.arange(1, 25)

    # Get baseline survival
    baseline_survival = cph.baseline_survival_

    # Get predicted cumulative hazard
    partial_hazards = cph.predict_partial_hazard(df)
    cumulative_hazard = np.outer(partial_hazards.values.flatten(), baseline_survival.index) ** 1

    # Survival = exp(-cumulative_hazard)
    survival_matrix = np.exp(-cumulative_hazard)

    # Build result DataFrame
    result = pd.DataFrame(survival_matrix.T, index=baseline_survival.index)
    result.index.name = "time"
    result.columns = [f"applicant_{i}" for i in range(len(df))]

    return result


def predict_for_new_applicant(
    cph: CoxPHFitter,
    credit_score: int = 680,
    employment_years: float = 3.0,
    debt_to_income: float = 0.25,
    loan_amount: float = 150000,
    interest_rate: float = 0.075,
    LTV_ratio: float = 0.75,
    max_time: int = 24,
) -> dict:
    """Predict survival for a single new applicant.

    Returns dict with survival probabilities and risk assessment.
    """
    applicant = create_applicant(
        credit_score, employment_years, debt_to_income,
        loan_amount, interest_rate, LTV_ratio
    )
    features = prepare_applicant_features(applicant)

    # Use lifelines built-in predict_survival_function
    survival_df = cph.predict_survival_function(features, times=np.arange(1, max_time + 1))

    # Key metrics
    survival_12m = float(survival_df.iloc[11, 0]) if len(survival_df) >= 12 else 0.5
    survival_24m = float(survival_df.iloc[23, 0]) if len(survival_df) >= 24 else 0.5

    # Risk score from Cox PH
    risk_score = cph.predict_partial_hazard(features).values[0]

    return {
        "credit_score": credit_score,
        "band": get_credit_score_band(credit_score),
        "risk_score": round(float(risk_score), 4),
        "survival_12m": round(survival_12m, 4),
        "survival_24m": round(survival_24m, 4),
        "monthly_survival": {
            str(t): round(float(survival_df.iloc[t-1, 0]), 4)
            for t in [6, 12, 18, 24]
        },
    }


def run(df: pd.DataFrame, cph: CoxPHFitter) -> dict:
    """Run prediction examples."""
    print("\n=== Survival Prediction for New Applicants ===")

    examples = [
        {
            "name": "High-Risk Applicant",
            "credit_score": 540,
            "employment_years": 1.0,
            "debt_to_income": 0.42,
            "loan_amount": 200000,
            "interest_rate": 0.14,
            "LTV_ratio": 0.95,
        },
        {
            "name": "Medium-Risk Applicant",
            "credit_score": 670,
            "employment_years": 4.0,
            "debt_to_income": 0.28,
            "loan_amount": 180000,
            "interest_rate": 0.085,
            "LTV_ratio": 0.80,
        },
        {
            "name": "Low-Risk Applicant",
            "credit_score": 780,
            "employment_years": 8.0,
            "debt_to_income": 0.18,
            "loan_amount": 250000,
            "interest_rate": 0.055,
            "LTV_ratio": 0.65,
        },
    ]

    results = {}

    for ex in examples:
        print(f"\n--- {ex['name']} ---")
        result = predict_for_new_applicant(
            cph,
            credit_score=ex["credit_score"],
            employment_years=ex["employment_years"],
            debt_to_income=ex["debt_to_income"],
            loan_amount=ex["loan_amount"],
            interest_rate=ex["interest_rate"],
            LTV_ratio=ex["LTV_ratio"],
        )

        print(f"Credit Score: {result['credit_score']} ({result['band']})")
        print(f"Risk Score: {result['risk_score']}")
        print(f"12-Month Survival: {result['survival_12m']:.1%}")
        print(f"24-Month Survival: {result['survival_24m']:.1%}")

        results[ex["name"]] = result

    return results


if __name__ == "__main__":
    from .data_loader import generate_loan_data
    from .cox_ph import fit_cox_ph

    df = generate_loan_data()
    _, cph = fit_cox_ph(df)

    results = run(df, cph)