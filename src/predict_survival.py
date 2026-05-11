"""
Predict survival function for a new loan applicant.

Uses the fitted Cox PH model to predict the survival curve
for a new applicant's risk profile.

Shows cumulative survival probability at 12 and 24 months,
and plots the predicted survival curve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def predict_survival(cph, new_applicant: dict) -> pd.DataFrame:
    """
    Predict survival curve for a new applicant using Cox PH model.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox PH model.
    new_applicant : dict
        Applicant features:
        - income: annual income in ZAR
        - credit_score: credit score (300-850)
        - employment_years: years employed
        - debt_to_income: monthly debt / monthly income ratio
        - loan_amount: requested loan amount in ZAR
        - interest_rate: annual interest rate (decimal)
        - LTV_ratio: loan-to-value ratio

    Returns
    -------
    pd.DataFrame
        DataFrame with timeline and survival probability.
    """
    # Normalize features to match model training (use standardized credit score)
    cs_mean, cs_std = 650, 85  # Approximate from data generation
    normalized = {
        'income_100k': new_applicant['income'] / 100000,
        'credit_score_z': (new_applicant['credit_score'] - cs_mean) / cs_std,
        'employment_years': new_applicant['employment_years'],
        'debt_to_income': new_applicant['debt_to_income'],
        'loan_amount_m': new_applicant['loan_amount'] / 1000000,
        'interest_rate': new_applicant['interest_rate'],
        'LTV_ratio': new_applicant['LTV_ratio']
    }

    X = pd.DataFrame([normalized])

    # Predict median survival time and survival probabilities
    surv_func = cph.predict_survival_function(X)
    timeline = surv_func.index

    result = pd.DataFrame({
        'month': timeline.astype(int),
        'survival_probability': surv_func.iloc[:, 0].values
    })

    return result


def plot_applicant_survival(survival_df: pd.DataFrame, applicant_label: str,
                            save_path: str = "reports/applicant_survival.png") -> None:
    """
    Plot the predicted survival curve for an applicant.

    Parameters
    ----------
    survival_df : pd.DataFrame
        Output from predict_survival().
    applicant_label : str
        Label for the applicant's risk profile.
    save_path : str
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.step(survival_df['month'], survival_df['survival_probability'],
            where='post', color='#1f77b4', linewidth=2)

    # Mark 12mo and 24mo survival
    surv_12 = survival_df[survival_df['month'] == 12]['survival_probability'].values[0]
    surv_24 = survival_df[survival_df['month'] == 24]['survival_probability'].values[0]

    ax.axhline(y=surv_12, color='orange', linestyle='--', alpha=0.7, label=f'12-mo: {surv_12:.1%}')
    ax.axhline(y=surv_24, color='red', linestyle='--', alpha=0.7, label=f'24-mo: {surv_24:.1%}')

    ax.set_xlabel("Months since origination", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_title(f"Predicted Survival Curve: {applicant_label}", fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def simulate_applicant_survival(df: pd.DataFrame, cph) -> dict:
    """
    Simulate survival predictions for example applicant profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (used for feature statistics).
    cph : CoxPHFitter
        Fitted Cox PH model.

    Returns
    -------
    dict
        Dictionary of example profiles and their predictions.
    """
    # Example: high-risk applicant (low credit score, high DTI, high LTV)
    high_risk = {
        'income': 250000,
        'credit_score': 560,
        'employment_years': 2,
        'debt_to_income': 0.45,
        'loan_amount': 1500000,
        'interest_rate': 0.18,
        'LTV_ratio': 0.90
    }

    # Example: medium-risk applicant
    medium_risk = {
        'income': 600000,
        'credit_score': 680,
        'employment_years': 5,
        'debt_to_income': 0.30,
        'loan_amount': 2000000,
        'interest_rate': 0.13,
        'LTV_ratio': 0.75
    }

    # Example: low-risk applicant
    low_risk = {
        'income': 1200000,
        'credit_score': 780,
        'employment_years': 10,
        'debt_to_income': 0.18,
        'loan_amount': 1800000,
        'interest_rate': 0.095,
        'LTV_ratio': 0.60
    }

    profiles = {
        'High Risk (score=560, high DTI/LTV)': high_risk,
        'Medium Risk (score=680)': medium_risk,
        'Low Risk (score=780, low DTI/LTV)': low_risk
    }

    results = {}
    for label, profile in profiles.items():
        surv_df = predict_survival(cph, profile)
        surv_12 = surv_df[surv_df['month'] == 12]['survival_probability'].values[0]
        surv_24 = surv_df[surv_df['month'] == 24]['survival_probability'].values[0]
        results[label] = {
            'survival_12mo': surv_12,
            'survival_24mo': surv_24,
            'profile': profile,
            'survival_df': surv_df
        }
        plot_applicant_survival(surv_df, label, f"reports/survival_{label.split()[0].lower()}_risk.png")

    return results


def print_applicant_summary(results: dict) -> None:
    """Print summary of applicant survival predictions."""
    print("\n" + "=" * 80)
    print("NEW APPLICANT SURVIVAL PREDICTIONS")
    print("=" * 80)
    print(f"{'Profile':<35} {'12mo Survival':>12} {'24mo Survival':>12} {'Risk Tier'}")
    print("-" * 80)

    for label, data in results.items():
        # Determine risk tier based on 24mo survival
        surv_24 = data['survival_24mo']
        if surv_24 >= 0.90:
            tier = "Low Risk"
        elif surv_24 >= 0.75:
            tier = "Medium Risk"
        else:
            tier = "High Risk"

        print(f"{label:<35} {data['survival_12mo']:>12.1%} {surv_24:>12.1%} {tier}")

    print("=" * 80)
    print("\nNote: Survival = probability of NOT defaulting by that month.")
    print("Higher survival = lower default risk.")


if __name__ == "__main__":
    from data_loader import generate_loan_data
    from cox_ph import fit_cox_ph

    df = generate_loan_data()
    cox_results = fit_cox_ph(df)
    cph = cox_results['cph']

    results = simulate_applicant_survival(df, cph)
    print_applicant_summary(results)