"""Cox Proportional Hazards model for default prediction."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .data_loader import get_credit_score_band


def fit_cox_ph(
    df: pd.DataFrame,
    time_col: str = "time_end",
    event_col: str = "event_default",
) -> tuple:
    """Fit Cox PH model and return coefficients with hazard ratios.

    Returns (coefficients DataFrame, fitted model).
    """
    # Prepare features
    features = [
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]

    df_model = df[features + [time_col, event_col]].copy()

    # Add credit band as categorical
    df_model["credit_band"] = df_model["credit_score"].apply(get_credit_score_band)
    # Create binary indicators for bands (drop one for reference)
    df_model["is_subprime"] = (df_model["credit_band"] == "Deep Subprime (< 580)").astype(int)
    df_model["is_near_prime"] = (df_model["credit_band"] == "Near Prime (670-739)").astype(int)
    df_model["is_prime"] = (df_model["credit_band"] == "Prime (740+)").astype(int)

    # Drop credit_score in favor of bands to avoid multicollinearity
    features_for_model = [
        "is_subprime",
        "is_near_prime",
        "is_prime",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "LTV_ratio",
    ]

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_model[features_for_model + [time_col, event_col]], duration_col=time_col, event_col=event_col)

    # Extract coefficients
    coef_df = pd.DataFrame({
        "coefficient": cph.params_,
        "hazard_ratio": np.exp(cph.params_),
        "se": cph.standard_errors_,
        "z": cph.summary["z"].values if "z" in cph.summary else cph.coef_z,
        "p": cph.summary["p"].values if "p" in cph.summary else cph.coef_p,
    })

    return coef_df, cph


def interpret_coefficients(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Interpret Cox PH coefficients into actionable insights."""
    interpretations = []

    for var, row in coef_df.iterrows():
        hr = row["hazard_ratio"]
        p = row["p"]

        # Interpret hazard ratio
        if hr > 1:
            direction = "increases"
            effect = f"{((hr-1)*100):.1f}% higher"
        else:
            direction = "decreases"
            effect = f"{((1-hr)*100):.1f}% lower"

        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        interpretations.append({
            "variable": var,
            "hazard_ratio": round(hr, 4),
            "effect": f"{effect} default risk",
            "p_value": round(p, 4),
            "significance": significance,
        })

    return pd.DataFrame(interpretations)


def run(df: pd.DataFrame) -> tuple:
    """Run Cox PH analysis."""
    print("\n=== Cox Proportional Hazards Model ===")

    coef_df, cph = fit_cox_ph(df)

    print("\nHazard Ratios:")
    print(coef_df[["hazard_ratio"]].round(4).to_string())

    print("\n--- Full Coefficient Summary ---")
    print(coef_df.round(4).to_string())

    # Concordance index (model fit)
    print(f"\nConcordance Index: {round(cph.concordance_index_, 4)}")

    return coef_df, cph


if __name__ == "__main__":
    from .data_loader import generate_loan_data

    df = generate_loan_data()
    coef_df, cph = run(df)
    interp = interpret_coefficients(coef_df)
    print("\nInterpretation:")
    print(interp.to_string(index=False))