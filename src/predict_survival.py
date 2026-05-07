"""
Predict survival function for a new loan applicant using Cox PH model.
"""

import pandas as pd
from lifelines import CoxPHFitter


def predict_survival(df: pd.DataFrame, applicant: dict, timelines=None) -> pd.DataFrame:
    """
    Fit Cox PH on the dataset, then predict survival curve for a new applicant.

    Parameters
    ----------
    df : pd.DataFrame — full loan dataset
    applicant : dict — feature values for new applicant
    timelines : array-like — time points at which to predict S(t)

    Returns
    -------
    pd.DataFrame with timelines and survival probabilities
    """
    features = [
        "credit_score",
        "income",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    if timelines is None:
        timelines = list(range(1, 61))

    df_model = df[features + ["time_end", "event_default"]].copy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_model, duration_col="time_end", event_col="event_default")

    appl_df = pd.DataFrame([applicant])
    for col in features:
        appl_df[col] = float(applicant.get(col, df_model[col].median()))

    surv = cph.predict_survival_function(appl_df, times=timelines)
    surv_df = pd.DataFrame({"timeline": timelines, "survival_probability": surv.values.flatten()})

    return surv_df


def print_applicant_prediction(applicant: dict, surv_df: pd.DataFrame) -> str:
    lines = ["\n" + "=" * 60]
    lines.append("NEW APPLICANT — PREDICTED SURVIVAL CURVE")
    lines.append("=" * 60)
    lines.append("Applicant Features:")
    for k, v in applicant.items():
        lines.append(f"  {k:<22} {v}")

    lines.append("\nSurvival Probability at Key Milestones:")
    for t in [6, 12, 18, 24, 36, 48, 60]:
        row = surv_df[surv_df["timeline"] == t]
        if not row.empty:
            lines.append(f"  S({t:>2}) = {row['survival_probability'].values[0]:.2%}")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    new_applicant = {
        "credit_score": 720,
        "income": 65000,
        "employment_years": 4.5,
        "debt_to_income": 0.28,
        "loan_amount": 25000,
        "interest_rate": 0.11,
        "LTV_ratio": 0.75,
    }
    surv_df = predict_survival(df, new_applicant)
    print(print_applicant_prediction(new_applicant, surv_df))