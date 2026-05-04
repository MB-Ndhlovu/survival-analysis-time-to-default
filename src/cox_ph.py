import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_ph(df: pd.DataFrame) -> dict:
    features = [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "LTV_ratio",
    ]
    X = df[features].copy()
    X["credit_score"] = (X["credit_score"] - X["credit_score"].mean()) / X["credit_score"].std()
    X["income"] = np.log1p(X["income"])
    X["loan_amount"] = np.log1p(X["loan_amount"])

    cph = CoxPHFitter()
    cph.fit(
        df[["time_end", "event_default"] + features],
        duration_col="time_end",
        event_col="event_default",
    )

    summary = cph.summary
    summary = summary.reset_index()
    summary.columns = ["covariate"] + list(summary.columns[1:])

    hazard_ratios = {}
    for _, row in summary.iterrows():
        hazard_ratios[row["covariate"]] = {
            "hazard_ratio": round(float(row["exp(coef)"]), 4),
            "coefficient": round(float(row["coef"]), 4),
            "p_value": round(float(row["p"]), 4),
            "significant": float(row["p"]) < 0.05,
        }

    return {"cph": cph, "hazard_ratios": hazard_ratios, "concordance_index": round(cph.concordance_index_, 4)}


def print_cox_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print("COX PROPORTIONAL HAZARDS MODEL RESULTS")
    print("=" * 60)
    print(f"\nConcordance Index: {results['concordance_index']}")
    print("\nHazard Ratios (sorted by impact):")

    sorted_hr = sorted(results["hazard_ratios"].items(), key=lambda x: x[1]["hazard_ratio"], reverse=True)
    for var, vals in sorted_hr:
        sig = "***" if vals["p_value"] < 0.001 else ("**" if vals["p_value"] < 0.01 else ("*" if vals["p_value"] < 0.05 else ""))
        print(f"  {var:<20} HR={vals['hazard_ratio']:>7.4f}  p={vals['p_value']:.4f} {sig}")

    print("\nInterpretation:")
    top_risk = [v for v, d in sorted_hr if d["hazard_ratio"] > 1 and d["significant"]][:3]
    if top_risk:
        print(f"  Top risk increasers: {', '.join(top_risk)}")
    low_risk = [v for v, d in sorted_hr if d["hazard_ratio"] < 1 and d["significant"]][:3]
    if low_risk:
        print(f"  Risk reducers: {', '.join(low_risk)}")


if __name__ == "__main__":
    from src.data_loader import generate_loan_data

    df = generate_loan_data()
    results = fit_cox_ph(df)
    print_cox_results(results)