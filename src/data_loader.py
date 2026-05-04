import numpy as np
import pandas as pd


def generate_loan_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan data with survival characteristics."""
    np.random.seed(seed)

    data = {}

    data["income"] = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)
    data["credit_score"] = np.random.normal(loc=680, scale=80, size=n_samples)
    data["credit_score"] = np.clip(data["credit_score"], 300, 850).astype(int)
    data["employment_years"] = np.random.exponential(scale=5, size=n_samples)
    data["employment_years"] = np.clip(data["employment_years"], 0, 40)
    data["debt_to_income"] = np.random.beta(a=2, b=5, size=n_samples) * 0.6
    data["loan_amount"] = np.random.lognormal(mean=9.5, sigma=0.7, size=n_samples)
    data["loan_amount"] = np.clip(data["loan_amount"], 1000, 500000)
    data["interest_rate"] = np.random.normal(loc=8.5, scale=3, size=n_samples)
    data["interest_rate"] = np.clip(data["interest_rate"], 2, 25)
    data["LTV_ratio"] = np.random.beta(a=5, b=5, size=n_samples) * 1.2

    df = pd.DataFrame(data)

    default_prob = (
        0.4 * (df["credit_score"] < 580).astype(float)
        + 0.25 * ((df["credit_score"] >= 580) & (df["credit_score"] < 670)).astype(float)
        + 0.12 * ((df["credit_score"] >= 670) & (df["credit_score"] < 740)).astype(float)
        + 0.05 * (df["credit_score"] >= 740).astype(float)
        + 0.3 * df["debt_to_income"]
        + 0.2 * (df["LTV_ratio"] > 0.8).astype(float)
        + 0.1 * (df["interest_rate"] > 15).astype(float)
    )
    default_prob = np.clip(default_prob, 0.01, 0.85)

    time_to_default = np.random.exponential(
        scale=30 / (default_prob * 10 + 0.5), size=n_samples
    )
    time_to_default = np.clip(time_to_default, 1, 120)

    censor_time = 24.0
    time_end = np.minimum(time_to_default, censor_time)
    event_default = (time_to_default <= censor_time).astype(int)

    df["time_start"] = 0.0
    df["time_end"] = np.round(time_end, 2)
    df["event_default"] = event_default

    return df


def get_credit_score_band(score: int) -> str:
    if score < 580:
        return "< 580"
    elif score < 670:
        return "580-669"
    elif score < 740:
        return "670-739"
    else:
        return "740+"


if __name__ == "__main__":
    df = generate_loan_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Default rate: {df['event_default'].mean():.2%}")
    print(f"Censored rate: {(df['event_default'] == 0).mean():.2%}")