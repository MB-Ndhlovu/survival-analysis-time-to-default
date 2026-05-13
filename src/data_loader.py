"""
Generate synthetic loan data with time-to-default survival structure.
Roughly 35% of observations are censored at 24 months.
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(seed=42)


def generate_loan_data(n=5000, censor_at=24):
    """
    Generate loan data with survival characteristics.

    Features influence hazard rate, creating realistic time-to-default patterns.
    Censoring at 24 months for ~35% of observations.
    """
    # Credit score bands with different base hazard rates
    # Lower scores = higher hazard
    credit_scores = rng.normal(680, 120, n).clip(300, 850).astype(int)

    # Income (R160k - R1.5M annual)
    income = rng.lognormal(12.5, 0.5, n)  # median ~270k

    # Employment years
    employment_years = rng.exponential(5, n).clip(0, 40)

    # Debt-to-income ratio
    debt_to_income = rng.beta(2, 8, n) * 15  # skewed toward low values

    # Loan amount (R50k - R2M)
    loan_amount = rng.lognormal(12, 0.8, n).clip(50000, 2000000)

    # Interest rate (coupled to credit score)
    base_rate = 0.12
    credit_effect = (700 - credit_scores) / 400 * 0.08
    interest_rate = base_rate + credit_effect.clip(0, 0.15) + rng.normal(0, 0.01, n)

    # LTV ratio
    property_value = loan_amount / rng.uniform(0.5, 0.95, n)
    LTV_ratio = loan_amount / property_value

    # Convert credit score to band
    def credit_band(score):
        if score < 580:
            return 'Very Poor'
        elif score < 670:
            return 'Fair'
        elif score < 740:
            return 'Good'
        else:
            return 'Excellent'

    credit_band_arr = np.array([credit_band(s) for s in credit_scores])

    # Base hazard multiplier by credit band
    hazard_multiplier = {
        'Very Poor': 3.5,
        'Fair': 2.0,
        'Good': 1.0,
        'Excellent': 0.4
    }

    # Time-to-default using accelerated failure time model
    # log(T) = mu + sigma * Z, where Z ~ standard normal
    # Hazard increases with: lower credit, higher DTI, higher LTV, higher rate

    base_mu = 18  # median ~18 months without risk factors
    sigma = 0.8

    # Risk factors increase hazard (reduce time to default)
    credit_risk = (700 - credit_scores) / 100 * 3
    dti_risk = debt_to_income * 0.15
    ltv_risk = (LTV_ratio - 0.6) * 2
    rate_risk = (interest_rate - 0.12) * 8

    total_risk = credit_risk + dti_risk + ltv_risk + rate_risk

    # Generate time to default
    z = rng.standard_normal(n)
    time_to_default = np.exp(base_mu - total_risk + sigma * z)

    # Ensure positive
    time_to_default = time_to_default.clip(0.5, None)

    # Determine event (1=defaulted, 0=censored)
    event_default = (time_to_default <= censor_at).astype(int)

    # For censored: time_end = censor_at. For defaulted: time_end = actual default time
    time_end = np.where(time_to_default <= censor_at, time_to_default, censor_at)
    time_start = np.zeros(n)  # all loans start at time 0

    # Create DataFrame
    df = pd.DataFrame({
        'time_start': time_start,
        'time_end': time_end.round(2),
        'event_default': event_default,
        'income': income.round(0).astype(int),
        'credit_score': credit_scores,
        'employment_years': employment_years.round(1),
        'debt_to_income': debt_to_income.round(3),
        'loan_amount': loan_amount.round(0).astype(int),
        'interest_rate': (interest_rate * 100).round(2),
        'LTV_ratio': LTV_ratio.round(3),
        'credit_band': credit_band_arr
    })

    # Apply censorship: ~35% censored
    # Currently roughly 65% defaulted (within 24 months)
    # We want exactly 35% censored, so adjust
    n_censored_target = int(n * 0.35)
    censored_mask = df['event_default'] == 0
    n_censored_current = censored_mask.sum()

    if n_censored_current < n_censored_target:
        # Convert some events to censored (extend their time past censor point)
        additional_censor = n_censored_target - n_censored_current
        event_indices = df[df['event_default'] == 1].index
        convert_indices = rng.choice(event_indices, additional_censor, replace=False)
        df.loc[convert_indices, 'event_default'] = 0
        df.loc[convert_indices, 'time_end'] = censor_at
    elif n_censored_current > n_censored_target:
        # Convert some censored to events (shrink their time)
        additional_event = n_censored_current - n_censored_target
        censor_indices = df[df['event_default'] == 0].index
        convert_indices = rng.choice(censor_indices, additional_event, replace=False)
        df.loc[convert_indices, 'event_default'] = 1
        df.loc[convert_indices, 'time_end'] = rng.uniform(1, censor_at - 0.1, additional_event)

    return df


if __name__ == "__main__":
    df = generate_loan_data(5000)
    print(f"Generated {len(df)} loans")
    print(f"Defaulted: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(f"Censored: {(~df['event_default'].astype(bool)).sum()} ({(1-df['event_default'].mean())*100:.1f}%)")
    print("\nCredit band distribution:")
    print(df['credit_band'].value_counts())
    print("\nSample data:")
    print(df.head())