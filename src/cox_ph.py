"""
Cox Proportional Hazards model for time-to-default.
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


COVARIATES = [
    "credit_score",
    "income",
    "employment_years",
    "debt_to_income",
    "loan_amount",
    "interest_rate",
    "LTV_ratio",
]


def fit_cox_ph(df: pd.DataFrame) -> CoxPHFitter:
    """
    Fit Cox Proportional Hazards model.
    Returns fitted cph object.
    """
    cph = CoxPHFitter()
    cph.fit(
        df[COVARIATES + ["time_end", "event_default"]],
        duration_col="time_end",
        event_col="event_default",
    )
    return cph


def hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    """
    Extract hazard ratios and confidence intervals.
    HR > 1 = increases default risk.
    HR < 1 = decreases default risk (protective).
    """
    summary = cph.summary.copy()
    summary = summary[["coef", "exp(coef)", "se(coef)", "p"]]
    summary.columns = ["coefficient", "hazard_ratio", "std_error", "p_value"]
    summary["hazard_ratio"] = np.exp(summary["coefficient"])
    summary = summary.round(4)
    return summary


def print_cox_summary(cph: CoxPHFitter) -> None:
    """Print formatted Cox PH results."""
    print("\n=== Cox Proportional Hazards Results ===")
    print(f"Concordance Index: {cph.concordance_index_:.4f}")
    print(f"Log-likelihood: {cph.log_likelihood_:.2f}")
    print("\nHazard Ratios (HR > 1 increases default risk):")
    print(hazard_ratios(cph))

    # Interpretation
    hr_df = hazard_ratios(cph)
    print("\n=== Key Risk Factors ===")
    top_risk = hr_df[hr_df["hazard_ratio"] > 1].sort_values("hazard_ratio", ascending=False)
    protective = hr_df[hr_df["hazard_ratio"] < 1].sort_values("hazard_ratio")

    print("\nTop factors that INCREASE default risk:")
    for idx, row in top_risk.iterrows():
        pct = (row["hazard_ratio"] - 1) * 100
        print(f"  {idx}: HR={row['hazard_ratio']:.4f} (+{pct:.1f}% risk per unit increase)")

    print("\nFactors that DECREASE default risk (protective):")
    for idx, row in protective.iterrows():
        pct = (1 - row["hazard_ratio"]) * 100
        print(f"  {idx}: HR={row['hazard_ratio']:.4f} (-{pct:.1f}% risk per unit increase)")


if __name__ == "__main__":
    from data_loader import generate_loan_data

    df = generate_loan_data(5000)
    cph = fit_cox_ph(df)
    print_cox_summary(cph)