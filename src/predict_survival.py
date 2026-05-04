import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def create_applicant(
    income: float = 55000,
    credit_score: int = 680,
    employment_years: float = 3.0,
    debt_to_income: float = 0.25,
    loan_amount: float = 15000,
    interest_rate: float = 7.5,
    LTV_ratio: float = 0.75,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [income],
            "credit_score": [credit_score],
            "employment_years": [employment_years],
            "debt_to_income": [debt_to_income],
            "loan_amount": [loan_amount],
            "interest_rate": [interest_rate],
            "LTV_ratio": [LTV_ratio],
        }
    )


def predict_survival(applicant: pd.DataFrame, cph: CoxPHFitter, df_ref: pd.DataFrame) -> dict:
    X = applicant.copy()
    X["credit_score"] = (X["credit_score"] - df_ref["credit_score"].mean()) / df_ref["credit_score"].std()
    X["income"] = np.log1p(X["income"])
    X["loan_amount"] = np.log1p(X["loan_amount"])

    times = [6, 12, 18, 24, 36, 48, 60]
    survival_probs = {}

    for t in times:
        try:
            survival_probs[f"{t}m"] = round(cph.predict_survival_function(X, times=[t]).iloc[0, 0], 4)
        except Exception:
            survival_probs[f"{t}m"] = None

    return {"survival_at_time": survival_probs, "applicant_data": applicant.to_dict(orient="records")[0]}


def print_prediction(prediction: dict) -> None:
    print("\n" + "=" * 50)
    print("NEW APPLICANT PREDICTION")
    print("=" * 50)
    app = prediction["applicant_data"]
    print(f"\nApplicant Profile:")
    print(f"  Income: ${app['income']:,.0f}")
    print(f"  Credit Score: {app['credit_score']}")
    print(f"  Employment: {app['employment_years']:.1f} years")
    print(f"  DTI: {app['debt_to_income']:.2%}")
    print(f"  Loan Amount: ${app['loan_amount']:,.0f}")
    print(f"  Interest Rate: {app['interest_rate']:.2f}%")
    print(f"  LTV: {app['LTV_ratio']:.2f}")
    print("\nPredicted Survival Probabilities:")
    for period, prob in prediction["survival_at_time"].items():
        if prob is not None:
            print(f"  {period}: {prob:.2%}")


if __name__ == "__main__":
    from src.data_loader import generate_loan_data
    from src.cox_ph import fit_cox_ph

    df = generate_loan_data()
    results = fit_cox_ph(df)

    applicant = create_applicant()
    prediction = predict_survival(applicant, results["cph"], df)
    print_prediction(prediction)