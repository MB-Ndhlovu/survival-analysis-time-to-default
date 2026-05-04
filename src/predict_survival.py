"""
Predict survival function for new loan applicants.
Uses fitted Cox PH model to predict individual survival curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from typing import Dict, Optional


def create_applicant_profile(
    credit_score: int,
    income: float,
    employment_years: float,
    debt_to_income: float,
    loan_amount: float,
    interest_rate: float,
    LTV_ratio: float
) -> pd.DataFrame:
    """
    Create a feature vector for a new applicant.

    Args:
        credit_score: FICO credit score
        income: Annual income
        employment_years: Years employed
        debt_to_income: Debt-to-income ratio
        loan_amount: Loan amount requested
        interest_rate: Interest rate
        LTV_ratio: Loan-to-value ratio

    Returns:
        DataFrame with applicant features
    """
    profile = pd.DataFrame([{
        'credit_score': credit_score,
        'income': income,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'LTV_ratio': LTV_ratio
    }])

    return profile


def predict_conditional_survival(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    t: int
) -> float:
    """
    Predict conditional survival probability for an applicant.

    Args:
        cph: Fitted CoxPHFitter model
        df: Training data (needed for baseline)
        profile: Applicant profile
        t: Time horizon in months

    Returns:
        Survival probability at time t
    """
    # Get baseline survival
    baseline_survival = cph.predict_survival_function(df.iloc[:1]) if len(df) > 0 else None

    # Get predicted hazard ratio for this profile
    log_hazard_ratio = cph.predict_log_partial_hazard(profile).iloc[0]

    # Simple approach: scale baseline survival
    # In practice, use the built-in predict methods
    try:
        survival_prob = cph.predict_survival_function(profile, times=[t]).iloc[0, 0]
    except:
        survival_prob = np.nan

    return survival_prob


def predict_full_survival_curve(
    cph: CoxPHFitter,
    profile: pd.DataFrame,
    times: list = None
) -> pd.Series:
    """
    Predict full survival curve for an applicant.

    Args:
        cph: Fitted CoxPHFitter model
        profile: Applicant profile
        times: Time points to evaluate (default: 0-60 months)

    Returns:
        Series with survival probabilities over time
    """
    if times is None:
        times = list(range(0, 61))

    try:
        sf = cph.predict_survival_function(profile)
        # Interpolate or sample at requested times
        if hasattr(sf, 'index') and hasattr(sf, 'columns'):
            # lifelines returns DataFrame with timeline as index
            timeline = sf.index.values
            surv_vals = sf.iloc[:, 0].values
            # Simple linear interpolation
            result = np.interp(times, timeline, surv_vals)
            return pd.Series(result, index=times)
        else:
            return pd.Series([np.nan] * len(times), index=times)
    except Exception as e:
        print(f"Prediction error: {e}")
        return pd.Series([np.nan] * len(times), index=times)


def predict_survival_for_new_applicant(
    cph: CoxPHFitter,
    profile: pd.DataFrame
) -> Dict[str, float]:
    """
    Generate comprehensive survival predictions for an applicant.

    Args:
        cph: Fitted CoxPHFitter model
        profile: Applicant profile

    Returns:
        Dict with survival probabilities at key time horizons
    """
    times = [6, 12, 18, 24, 36, 48, 60]
    curve = predict_full_survival_curve(cph, profile, times)

    # Hazard ratio (relative risk)
    log_hr = cph.predict_log_partial_hazard(profile).iloc[0]
    hazard_ratio = np.exp(log_hr)

    results = {
        'hazard_ratio': hazard_ratio,
        '6_month_survival': curve.get(6, np.nan),
        '12_month_survival': curve.get(12, np.nan),
        '18_month_survival': curve.get(18, np.nan),
        '24_month_survival': curve.get(24, np.nan),
        '36_month_survival': curve.get(36, np.nan),
        '48_month_survival': curve.get(48, np.nan),
        '60_month_survival': curve.get(60, np.nan)
    }

    return results


def plot_applicant_survival(
    cph: CoxPHFitter,
    profile: pd.DataFrame,
    df_train: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot applicant's predicted survival curve with portfolio comparison.

    Args:
        cph: Fitted CoxPHFitter
        profile: Applicant profile
        df_train: Training data for comparison
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = list(range(0, 61))

    # Get applicant's predicted curve
    applicant_curve = predict_full_survival_curve(cph, profile, times)

    # Plot applicant curve
    ax.plot(applicant_curve.index, applicant_curve.values,
            'b-', linewidth=3, label='Applicant Predicted')

    # Plot confidence band (using percentiles of population)
    kmf = KaplanMeierFitter()
    kmf.fit(df_train['time_end'], df_train['event_default'])

    # Overall population curve
    pop_sf = kmf.survival_function_at_times(times)
    population_survival = pop_sf.values.flatten()

    ax.fill_between(times, [p * 0.9 for p in population_survival],
                    [p * 1.1 for p in population_survival],
                    alpha=0.2, color='gray', label='Population Range (±10%)')

    ax.set_title('Applicant Predicted Survival Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def print_applicant_assessment(profile: pd.DataFrame, results: Dict[str, float]):
    """Print formatted assessment for a new applicant."""
    print("\n" + "="*70)
    print("NEW APPLICANT SURVIVAL PREDICTION")
    print("="*70)

    print("\n👤 Applicant Profile:")
    print("-" * 50)
    for col, val in profile.iloc[0].items():
        if 'rate' in col or 'ratio' in col:
            print(f"   {col:<20} {val:.4f}")
        elif 'income' in col or 'loan' in col:
            print(f"   {col:<20} ${val:,.0f}")
        else:
            print(f"   {col:<20} {val}")

    print("\n📊 Survival Probabilities by Time Horizon:")
    print("-" * 50)
    milestones = [12, 24, 36, 48, 60]
    print(f"{'Time':<15} {'Survival':>15} {'Risk':>15}")
    print("-" * 50)

    for t in milestones:
        key = f'{t}_month_survival'
        surv = results.get(key, np.nan)
        risk = 1 - surv
        bar_len = int(risk * 20) if not np.isnan(risk) else 0
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"{t}-month{' ' * 7} {surv:>14.1%} {bar}")

    print("\n🎯 Key Metrics:")
    print("-" * 50)
    hr = results['hazard_ratio']
    if hr > 1.5:
        risk_level = "HIGH"
    elif hr > 1.0:
        risk_level = "ELEVATED"
    elif hr > 0.5:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    print(f"   Hazard Ratio (vs avg): {hr:.3f}")
    print(f"   Risk Classification:   {risk_level}")
    print(f"   24-month Default Risk: {(1 - results.get('24_month_survival', 0))*100:.1f}%")


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands
    from cox_ph import fit_cox_model

    # Train model
    df = add_credit_bands(generate_loan_data())
    cph = fit_cox_model(df)

    # New applicant
    profile = create_applicant_profile(
        credit_score=720,
        income=85000,
        employment_years=7.5,
        debt_to_income=0.22,
        loan_amount=35000,
        interest_rate=0.085,
        LTV_ratio=0.75
    )

    results = predict_survival_for_new_applicant(cph, profile)
    print_applicant_assessment(profile, results)
    plot_applicant_survival(cph, profile, df, 'applicant_survival.png')

    print("\n✅ Prediction complete")