"""Generate synthetic loan data for survival analysis."""

import numpy as np
import pandas as pd


def generate_loan_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan data with survival outcomes.

    Generates n_samples loan records with:
    - time_start: start time (always 0)
    - time_end: time of default or censoring
    - event_default: 1 if default occurred, 0 if censored
    - Various loan and borrower features
    """
    np.random.seed(seed)

    # Credit score distribution (skewed toward higher scores)
    credit_score = np.random.normal(680, 100, n_samples).clip(300, 850).astype(int)

    # Income ($K/year) - correlated with credit score
    income = (
        30 + 0.05 * (credit_score - 680)
        + np.random.normal(0, 15, n_samples)
    ).clip(15, 500).round(1)

    # Employment years
    employment_years = np.random.exponential(5, n_samples).clip(0, 40).round(1)

    # Debt-to-income ratio
    debt_to_income = np.random.normal(0.35, 0.15, n_samples).clip(0.05, 1.5)

    # Loan amount ($K)
    loan_amount = np.random.normal(50, 30, n_samples).clip(5, 500).round(1)

    # Interest rate (risk-based pricing)
    base_rate = 5.0
    interest_rate = (
        base_rate
        + 0.05 * (750 - credit_score).clip(0, 300) / 100
        + 0.5 * debt_to_income.clip(0, 0.5)
        + np.random.normal(0, 0.5, n_samples)
    ).clip(3, 20).round(2)

    # LTV ratio
    LTV_ratio = np.random.normal(0.75, 0.2, n_samples).clip(0.1, 1.2)

    # Base hazard function: lower credit score = higher hazard
    def base_hazard(cs):
        if cs < 580:
            return 0.08
        elif cs < 670:
            return 0.04
        elif cs < 740:
            return 0.02
        else:
            return 0.01

    # Adjust hazard by other risk factors
    def adjusted_hazard(cs, dti, ltv):
        hazard = base_hazard(cs)
        hazard *= 1 + 0.5 * max(0, dti - 0.4)
        hazard *= 1 + 0.3 * max(0, ltv - 0.8)
        return hazard

    # Generate survival times
    time_start = np.zeros(n_samples, dtype=int)
    time_end = np.zeros(n_samples, dtype=int)
    event_default = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        h = adjusted_hazard(credit_score[i], debt_to_income[i], LTV_ratio[i])

        # Survival time follows exponential distribution
        survival_time = np.random.exponential(1 / h)

        # Censor at 24 months with ~35% probability
        if np.random.random() < 0.35:
            time_end[i] = 24
            event_default[i] = 0
        else:
            # Cap at 60 months
            survival_time = min(survival_time, 60)
            time_end[i] = int(round(survival_time))
            event_default[i] = 1

    df = pd.DataFrame({
        "time_start": time_start,
        "time_end": time_end,
        "event_default": event_default,
        "income": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income.round(4),
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "LTV_ratio": LTV_ratio.round(4),
    })

    return df


def get_credit_score_band(score: int) -> str:
    """Map credit score to risk band."""
    if score < 580:
        return "Poor (<580)"
    elif score < 670:
        return "Fair (580-669)"
    elif score < 740:
        return "Good (670-739)"
    else:
        return "Excellent (740+)"


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loan records")
    print(f"Default rate: {df['event_default'].mean():.2%}")
    print(f"Censored rate: {(df['event_default'] == 0).mean():.2%}")
    print("\nSample data:")
    print(df.head(10))