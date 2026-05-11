"""Predict survival function for a new loan applicant."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter


def predict_survival_for_applicant(
    df: pd.DataFrame,
    applicant: dict,
    output_path: str = "reports/applicant_survival.png"
) -> dict:
    """
    Predict the survival curve for a new loan applicant using the Cox PH model.

    applicant dict should contain:
    - income, credit_score, employment_years, debt_to_income,
      loan_amount, interest_rate, LTV_ratio
    """
    # Ensure column name is correct
    df_model = df.copy()

    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(
        df_model[["time_end", "event_default", "income", "credit_score", "employment_years",
                  "debt_to_income", "loan_amount", "interest_rate", "LTV_ratio"]],
        duration_col="time_end",
        event_col="event_default"
    )

    # Build applicant dataframe
    applicant_df = pd.DataFrame([{
        "income": applicant.get("income", 50000),
        "credit_score": applicant.get("credit_score", 680),
        "employment_years": applicant.get("employment_years", 3),
        "debt_to_income": applicant.get("debt_to_income", 0.25),
        "loan_amount": applicant.get("loan_amount", 50000),
        "interest_rate": applicant.get("interest_rate", 7.5),
        "LTV_ratio": applicant.get("LTV_ratio", 0.7),
    }])

    # Predict median survival
    median_survival = cph.predict_median(applicant_df)
    median_survival_val = float(median_survival)
    med_months = None
    if not np.isnan(median_survival_val) and not np.isinf(median_survival_val):
        med_months = int(median_survival_val)

    # Predict survival function
    survival_times = np.arange(0, 37, 1)
    survival_df = cph.predict_survival_function(applicant_df, times=survival_times)
    survival_probs = survival_df.values.flatten()

    # Compare to population average
    kmf = KaplanMeierFitter()
    kmf.fit(df["time_end"], df["event_default"])
    population_survival = kmf.survival_function_at_times(survival_times).values

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(survival_times, survival_probs, "b-", linewidth=2, label="Applicant")
    ax.plot(survival_times, population_survival, "k--", linewidth=1.5, alpha=0.7, label="Population Avg")

    # Reference lines at 50%, 75%, 90% survival
    for threshold in [0.5, 0.75, 0.9]:
        idx = np.where(survival_probs <= threshold)[0]
        if len(idx) > 0:
            t_cross = survival_times[idx[0]]
            ax.axhline(y=threshold, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(x=t_cross, color="gray", linestyle=":", alpha=0.5)
            ax.annotate(f"{threshold:.0%} survival at {t_cross}m",
                       xy=(t_cross, threshold), fontsize=8, color="gray")

    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Predicted Survival Curve for New Applicant (Score: {applicant.get('credit_score', 680)})")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "median_survival_months": med_months,
        "survival_12m": round(float(survival_probs[np.where(survival_times == 12)[0][0]]), 4),
        "survival_24m": round(float(survival_probs[np.where(survival_times == 24)[0][0]]), 4),
        "survival_36m": round(float(survival_probs[np.where(survival_times == 36)[0][0]]), 4),
        "applicant_profile": applicant,
    }

    return results


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data()
    applicant = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.22,
        "loan_amount": 80000,
        "interest_rate": 6.5,
        "LTV_ratio": 0.75,
    }
    results = predict_survival_for_applicant(df, applicant)
    print("\nApplicant Survival Prediction:")
    print(f"  Median survival: {results['median_survival_months']} months")
    print(f"  12m survival: {results['survival_12m']:.1%}")
    print(f"  24m survival: {results['survival_24m']:.1%}")
    print(f"  36m survival: {results['survival_36m']:.1%}")