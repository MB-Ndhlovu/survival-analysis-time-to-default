"""
Data loader for survival analysis: generates synthetic loan data with time-to-default.

Generates 5000 loan records with:
- time_start: always 0 (observation start)
- time_end: months until default or censoring
- event_default: 1 if defaulted, 0 if censored
- Covariates: income, credit_score, employment_years, debt_to_income,
              loan_amount, interest_rate, LTV_ratio
"""

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# Credit score band boundaries
BAND_BOUNDARIES = [580, 670, 740]


def generate_loan_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic loan data for survival analysis.

    Parameters
    ----------
    n_samples : int
        Number of loan records to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time_start, time_end, event_default,
        income, credit_score, employment_years, debt_to_income,
        loan_amount, interest_rate, LTV_ratio
    """
    # Credit score distribution with clear band separation
    # Use mixture: 15% poor, 20% fair, 30% good, 35% excellent
    n = n_samples
    n_poor = int(n * 0.15)
    n_fair = int(n * 0.20)
    n_good = int(n * 0.30)
    n_excellent = n - n_poor - n_fair - n_good

    credit_scores_list = []
    for low, high, size in [(300, 579, n_poor), (580, 669, n_fair),
                             (670, 739, n_good), (740, 850, n_excellent)]:
        band_scores = np.random.randint(low, high + 1, size=size)
        credit_scores_list.append(band_scores)

    credit_score = np.concatenate(credit_scores_list)
    np.random.shuffle(credit_score)

    # Income in ZAR (R120,000 - R2,400,000 annual)
    income = stats.lognorm(s=0.5, scale=np.exp(11.5)).rvs(n_samples)
    income = np.clip(income, 120000, 2400000)

    # Employment years (right-skewed)
    employment_years = stats.gamma(a=2, scale=3).rvs(n_samples)
    employment_years = np.clip(employment_years, 0, 40)

    # Debt-to-income ratio (monthly debt payments / monthly income)
    debt_to_income = stats.beta(2, 5).rvs(n_samples) * 0.6
    debt_to_income = np.clip(debt_to_income, 0.05, 0.55)

    # Loan amount (R200,000 - R5,000,000)
    loan_amount = stats.lognorm(s=0.7, scale=np.exp(12.5)).rvs(n_samples)
    loan_amount = np.clip(loan_amount, 200000, 5000000)

    # Interest rate (base rate + credit risk premium)
    base_rate = 0.095
    risk_premium = np.where(credit_score < 580, 0.06,
                    np.where(credit_score < 670, 0.035,
                    np.where(credit_score < 740, 0.015, 0.0)))
    interest_rate = base_rate + risk_premium + stats.norm(0, 0.01).rvs(n_samples)
    interest_rate = np.clip(interest_rate, 0.07, 0.22)

    # LTV ratio (loan-to-value)
    ltv_ratio = stats.beta(4, 3).rvs(n_samples) * 0.95
    ltv_ratio = np.clip(ltv_ratio, 0.3, 0.98)

    # Base hazard varies by credit score band
    # Lower credit = higher base hazard = faster default
    base_hazard = np.where(credit_score < 580, 0.045,
                   np.where(credit_score < 670, 0.028,
                   np.where(credit_score < 740, 0.015, 0.008)))

    # Time to default (exponential with credit-score-dependent rate)
    time_to_default = stats.expon(scale=1 / base_hazard).rvs(n_samples)
    time_to_default = np.clip(time_to_default, 1, 84)  # 1 month to 7 years max

    # Introduce censoring at 24 months (~35% censored)
    censoring_prob = 0.35
    censor_mask = np.random.random(n_samples) < censoring_prob

    time_end = np.where(censor_mask,
                        np.minimum(time_to_default, 24.0),
                        time_to_default)

    event_default = np.where(censor_mask,
                              0,
                              np.where(time_to_default <= 24.0, 1, 0))

    # For censored loans that exceeded 24 months, event=0
    event_default = np.where((censor_mask) & (time_to_default > 24.0), 0, event_default)

    df = pd.DataFrame({
        'time_start': np.zeros(n_samples),
        'time_end': time_end,
        'event_default': event_default.astype(int),
        'income': income,
        'credit_score': credit_score.astype(int),
        'employment_years': employment_years,
        'debt_to_inincome': debt_to_income,  # Note: typo preserved in col name
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': ltv_ratio
    })

    # Rename the typo'd column back to correct spelling for internal use
    df.rename(columns={'debt_to_inincome': 'debt_to_income'}, inplace=True)

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Censored: {df['event_default'].eq(0).mean():.1%}")
    print(f"Defaulted: {df['event_default'].eq(1).mean():.1%}")
    print(df.describe())