"""
Predict survival function for a new loan applicant.
Uses the fitted Cox PH model to predict conditional survival curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predict_survival_for_applicant(cph, applicant_data):
    """
    Predict survival curve for a new applicant given the fitted Cox model.
    
    Parameters:
    - cph: fitted CoxPHFitter
    - applicant_data: dict with keys matching the model features
    
    Returns: DataFrame with times and survival probabilities
    """
    # Build feature vector matching training order
    feature_order = [
        'credit_score', 'employment_years', 'debt_to_income',
        'interest_rate', 'LTV_ratio', 'income', 'loan_amount'
    ]
    # Note: income and loan_amount were log-transformed in Cox fitting
    # So we pass log-transformed values
    
    features = []
    for f in feature_order:
        if f in ['income', 'loan_amount']:
            features.append(np.log(applicant_data[f]))
        else:
            features.append(applicant_data[f])
    
    # Create DataFrame for prediction
    X = pd.DataFrame([features], columns=feature_order)
    
    # Get survival function predictions
    surv_func = cph.predict_survival_function(X)
    
    # Define time points for prediction
    survival_times = np.linspace(1, 60, 100)
    
    # Get actual times and survival values from the model prediction
    actual_times = surv_func.index.values.astype(float)
    actual_survival = surv_func.iloc[:, 0].values  # first column = first applicant
    
    # Interpolate to desired time points
    result = pd.DataFrame({
        'time': survival_times,
        'survival_probability': np.interp(
            survival_times,
            actual_times,
            actual_survival
        )
    })
    
    return result


def compare_to_bands(applicant_data, km_results_by_band):
    """
    Compare new applicant's survival to the credit band benchmarks.
    
    Returns dict: band -> (survival_12m, survival_24m, median_survival)
    """
    return km_results_by_band


def plot_applicant_survival(survival_df, km_results_by_band, applicant_data, save_path=None):
    """
    Plot new applicant's predicted survival curve vs credit band benchmarks.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Band benchmarks with colors
    band_colors = {
        'Deep Subprime (<580)': '#d62728',
        'Subprime (580-669)': '#ff7f0e',
        'Near Prime (670-739)': '#2ca02c',
        'Prime (740+)': '#1f77b4',
    }
    
    # Plot benchmark curves
    for band, data in km_results_by_band.items():
        kmf = data['kmf']
        ax.plot(kmf.survival_function_.index,
                kmf.survival_function_.iloc[:, 0],
                color=band_colors.get(band),
                linewidth=2, alpha=0.5,
                label=f'{band} (benchmark)')
    
    # Plot applicant curve
    ax.plot(survival_df['time'], survival_df['survival_probability'],
            color='black', linewidth=3, linestyle='--',
            label='New Applicant')
    
    # Confidence band
    ax.fill_between(survival_df['time'],
                    survival_df['survival_probability'] * 0.95,
                    survival_df['survival_probability'] * 1.05,
                    color='gray', alpha=0.2, label='±5% band')
    
    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title('Predicted Survival Curve: New Applicant vs Credit Band Benchmarks',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.text(60, 0.52, '50% survival line', fontsize=9, color='gray')
    
    # Annotate applicant key points
    s12 = np.interp(12, survival_df['time'], survival_df['survival_probability'])
    s24 = np.interp(24, survival_df['time'], survival_df['survival_probability'])
    
    ax.annotate(f'12m: {s12:.1%}', xy=(12, s12), xytext=(15, s12 - 0.08),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, color='black')
    ax.annotate(f'24m: {s24:.1%}', xy=(24, s24), xytext=(27, s24 - 0.08),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, color='black')
    
    # Add applicant details as text box
    details = (
        f"Applicant Profile:\n"
        f"Credit Score: {applicant_data['credit_score']:.0f}\n"
        f"Income: R{applicant_data['income']:,.0f}\n"
        f"DTI: {applicant_data['debt_to_income']:.1%}\n"
        f"Interest Rate: {applicant_data['interest_rate']:.1%}\n"
        f"LTV: {applicant_data['LTV_ratio']:.2f}"
    )
    ax.text(0.98, 0.98, details, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved applicant prediction plot to {save_path}")
    
    return fig


def get_applicant_risk_profile(applicant_data, km_results_by_band, cph):
    """
    Generate a risk profile summary for the applicant.
    """
    profile = {}
    
    # Determine which credit band they fall into
    score = applicant_data['credit_score']
    if score < 580:
        band = 'Deep Subprime (<580)'
    elif score < 670:
        band = 'Subprime (580-669)'
    elif score < 740:
        band = 'Near Prime (670-739)'
    else:
        band = 'Prime (740+)'
    
    profile['assigned_band'] = band
    profile['band_data'] = km_results_by_band.get(band, {})
    
    # Compute hazard relative to reference (mean applicant)
    # For Cox PH, the linear predictor gives log-hazard ratio
    feature_order = [
        'credit_score', 'employment_years', 'debt_to_income',
        'interest_rate', 'LTV_ratio', 'income', 'loan_amount'
    ]
    features = []
    for f in feature_order:
        if f in ['income', 'loan_amount']:
            features.append(np.log(applicant_data[f]))
        else:
            features.append(applicant_data[f])
    
    X = pd.DataFrame([features], columns=feature_order)
    log_hazard = cph.predict_log_partial_hazard(X)[0]
    profile['log_hazard_relative'] = log_hazard
    
    # Risk score: higher = more risky
    risk_score = np.exp(log_hazard)
    profile['relative_risk_score'] = risk_score
    
    # 12m and 24m survival probabilities
    # Using the Cox model survival prediction
    surv_func = cph.predict_survival_function(X)
    profile['survival_12m'] = np.interp(12, surv_func.index, surv_func.iloc[:, 0].values)
    profile['survival_24m'] = np.interp(24, surv_func.index, surv_func.iloc[:, 0].values)
    
    return profile


def print_applicant_profile(profile, applicant_data):
    """Print formatted applicant risk profile."""
    print("\n" + "="*70)
    print("NEW APPLICANT RISK PROFILE")
    print("="*70)
    
    print(f"\nApplicant Data:")
    for k, v in applicant_data.items():
        if k in ['income', 'loan_amount']:
            print(f"  {k}: R{v:,.0f}")
        elif k == 'credit_score':
            print(f"  {k}: {v:.0f}")
        elif k in ['debt_to_income', 'interest_rate', 'LTV_ratio']:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nAssigned Credit Band: {profile['assigned_band']}")
    print(f"Relative Risk Score: {profile['relative_risk_score']:.3f}")
    print("  (1.0 = average risk, >1.0 = higher than average, <1.0 = lower)")
    
    print(f"\nPredicted Survival Probabilities:")
    print(f"  12-month survival: {profile['survival_12m']:.1%}")
    print(f"  24-month survival: {profile['survival_24m']:.1%}")
    
    if profile['band_data']:
        band_data = profile['band_data']
        print(f"\nComparison to {profile['assigned_band']} benchmark:")
        print(f"  Band 12m survival: {band_data.get('survival_12m', 0):.1%}")
        print(f"  Band 24m survival: {band_data.get('survival_24m', 0):.1%}")
        print(f"  Band median survival: {band_data.get('median_survival', 'N/A'):.1f} months")
    
    print("="*70)


if __name__ == '__main__':
    from data_loader import generate_loan_data, add_credit_bands
    from kaplan_meier import fit_by_credit_band
    from cox_ph import fit_cox_ph
    
    # Load and fit
    df = generate_loan_data()
    df = add_credit_bands(df)
    
    cph = fit_cox_ph(df)
    km_results = fit_by_credit_band(df)
    
    # New applicant
    applicant = {
        'credit_score': 710,
        'income': 650000,
        'employment_years': 4.5,
        'debt_to_income': 0.28,
        'loan_amount': 800000,
        'interest_rate': 0.095,
        'LTV_ratio': 0.75,
    }
    
    # Predict
    survival_df = predict_survival_for_applicant(cph, applicant)
    profile = get_applicant_risk_profile(applicant, km_results, cph)
    
    print_applicant_profile(profile, applicant)
    
    # Plot
    fig = plot_applicant_survival(survival_df, km_results, applicant)
    plt.show()