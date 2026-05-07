"""
Predict survival function for a new loan applicant.
"""
import pandas as pd
from lifelines import CoxPHFitter


def predict_survival(cph: CoxPHFitter, applicant: dict, months: list = None) -> pd.DataFrame:
    """
    Predict survival curve for a new applicant using the fitted Cox PH model.

    applicant: dict with keys matching COVARIATES from cox_ph.py
    months: list of time points to predict (default: 0-36 months)
    """
    if months is None:
        months = list(range(0, 37))

    # Build DataFrame for prediction
    row = {k: [applicant.get(k, 0)] for k in cph.params_.index}
    X = pd.DataFrame(row)

    # Conditional survival function
    surv_func = cph.predict_survival_function(X).T
    surv_func.index = surv_func.index.astype(int)

    # Interpolate to requested months
    result_rows = []
    for m in months:
        if m in surv_func.columns:
            prob = surv_func[m].values[0]
        else:
            # Find closest column
            closest = min(surv_func.columns, key=lambda x: abs(x - m))
            prob = surv_func[closest].values[0]
        result_rows.append({"month": m, "survival_prob": round(float(prob), 4)})

    return pd.DataFrame(result_rows)


def print_applicant_prediction(applicant: dict, surv_df: pd.DataFrame) -> None:
    """Pretty-print a new applicant's survival prediction."""
    print(f"\n=== New Applicant Prediction ===")
    print(f"Credit Score: {applicant.get('credit_score', 'N/A')}")
    print(f"Income: R{applicant.get('income', 0):,.0f}")
    print(f"Employment Years: {applicant.get('employment_years', 0)}")
    print(f"Debt-to-Income: {applicant.get('debt_to_income', 0):.2%}")
    print(f"Loan Amount: R{applicant.get('loan_amount', 0):,.0f}")
    print(f"Interest Rate: {applicant.get('interest_rate', 0):.2%}")
    print(f"LTV Ratio: {applicant.get('LTV_ratio', 0):.2%}")

    print("\nSurvival Probabilities:")
    key_months = [6, 12, 18, 24, 30, 36]
    for _, row in surv_df.iterrows():
        if int(row["month"]) in key_months:
            prob = float(row["survival_prob"])
            print(f"  Month {int(row['month']):2d}: {prob:.1%}")

    # Default probability at key horizons
    print("\nDefault Probabilities:")
    for _, row in surv_df.iterrows():
        if int(row["month"]) in key_months:
            prob = float(row["survival_prob"])
            print(f"  Month {int(row['month']):2d}: {1-prob:.1%}")


if __name__ == "__main__":
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph

    df = generate_loan_data(5000)
    cph = fit_cox_ph(df)

    new_applicant = {
        "credit_score": 680,
        "income": 450_000,
        "employment_years": 3.5,
        "debt_to_income": 0.28,
        "loan_amount": 250_000,
        "interest_rate": 0.095,
        "LTV_ratio": 0.72,
    }

    surv_df = predict_survival(cph, new_applicant)
    print_applicant_prediction(new_applicant, surv_df)