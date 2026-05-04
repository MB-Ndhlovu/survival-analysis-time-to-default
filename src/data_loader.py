import numpy as np
import pandas as pd

def generate_loan_data(n=5000, seed=42):
    """Generate synthetic loan data for survival analysis."""
    np.random.seed(seed)
    
    # Credit score distribution (realistic US distribution)
    credit_score = np.random.normal(700, 100, n).clip(300, 850).astype(int)
    
    # Income correlated with credit score
    income = (credit_score * 80 + np.random.normal(0, 15000, n) + 10000).clip(15000, 500000)
    
    # Employment years
    employment_years = np.random.exponential(5, n).clip(0, 40)
    
    # Debt-to-income ratio
    debt_to_income = np.random.lognormal(0.8, 0.5, n).clip(0.1, 1.5)
    
    # Loan amount correlated with income
    loan_amount = (income * debt_to_income * 24 + np.random.normal(0, 5000, n)).clip(5000, 500000)
    
    # Interest rate inversely related to credit score
    base_rate = 12 - (credit_score - 300) / 550 * 9
    interest_rate = base_rate + np.random.normal(0, 1.5, n).clip(2, 20)
    
    # LTV ratio
    LTV_ratio = np.random.lognormal(0.5, 0.3, n).clip(0.2, 1.5)
    
    # Generate time to default (faster for lower credit scores)
    default_hazard_base = np.exp(-(credit_score - 300) / 200)
    time_to_default = np.random.exponential(20 / (default_hazard_base + 0.3), n).clip(1, 60)
    
    # Generate time to censor (observation period ends at 24 months)
    # ~35% censoring rate
    time_to_censor = np.random.exponential(15, n)
    
    # Assign event vs censor
    event_default = (time_to_default < time_to_censor).astype(int)
    
    # Observed duration
    duration = np.minimum(time_to_default, time_to_censor)
    
    # Cap observation at 24 months max
    duration = duration.clip(1, 24)
    
    # Re-censor: if true default time > 24 but observed < 24, it's censored
    # This achieves ~35% censoring around 24 months
    censor_mask = (time_to_default > 24) & (duration <= 24)
    event_default = np.where(censor_mask, 0, event_default)
    
    # Ensure exactly ~35% censoring
    actual_censored = (event_default == 0).sum()
    target_censored = int(n * 0.35)
    
    if actual_censored > target_censored:
        # Convert some events to censored near end
        excess = actual_censored - target_censored
        near_end = (duration > 18) & (event_default == 1)
        indices = np.where(near_end)[0][:excess]
        event_default[indices] = 0
    elif actual_censored < target_censored:
        # Convert some censored to events early
        excess = target_censored - actual_censored
        early_censored = (duration < 8) & (event_default == 0)
        indices = np.where(early_censored)[0][:excess]
        event_default[indices] = 1
    
    df = pd.DataFrame({
        'time_start': 0,
        'time_end': duration.round(2),
        'event_default': event_default,
        'income': income.round(0).astype(int),
        'credit_score': credit_score,
        'employment_years': employment_years.round(1),
        'debt_to_income': debt_to_income.round(3),
        'loan_amount': loan_amount.round(0).astype(int),
        'interest_rate': interest_rate.round(2),
        'LTV_ratio': LTV_ratio.round(3)
    })
    
    return df

if __name__ == "__main__":
    df = generate_loan_data()
    print(f"Generated {len(df)} loans")
    print(f"Defaults: {df['event_default'].sum()} ({df['event_default'].mean()*100:.1f}%)")
    print(df.head())