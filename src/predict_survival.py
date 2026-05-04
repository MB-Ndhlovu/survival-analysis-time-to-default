import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def create_credit_bands(df):
    """Create credit score bands."""
    bins = [0, 580, 670, 740, 900]
    labels = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']
    df['credit_band'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    return df

def predict_survival_for_applicant(cph, applicant, kmf_by_band):
    """Predict survival curve for a new loan applicant."""
    # Prepare applicant data
    app_df = pd.DataFrame([applicant])
    
    # Log transforms
    app_df['log_income'] = np.log(applicant['income'])
    app_df['log_loan'] = np.log(applicant['loan_amount'])
    
    # Predict hazard using Cox model
    partial_hazard = cph.predict_partial_hazard(app_df)
    
    print(f"\n=== New Applicant Prediction ===")
    print(f"Credit Score: {applicant['credit_score']}")
    print(f"Income: ${applicant['income']:,.0f}")
    print(f"Loan Amount: ${applicant['loan_amount']:,.0f}")
    print(f"Interest Rate: {applicant['interest_rate']:.2f}%")
    print(f"Debt-to-Income: {applicant['debt_to_income']:.3f}")
    print(f"LTV Ratio: {applicant['LTV_ratio']:.3f}")
    print(f"Employment Years: {applicant['employment_years']}")
    
    # Determine credit band
    score = applicant['credit_score']
    if score < 580:
        band = 'Poor (<580)'
    elif score < 670:
        band = 'Fair (580-669)'
    elif score < 740:
        band = 'Good (670-739)'
    else:
        band = 'Excellent (740+)'
    
    print(f"Credit Band: {band}")
    print(f"Relative Hazard vs Baseline: {partial_hazard.values[0]:.4f}")
    
    # Plot comparison with band
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot band survival function
    band_data = kmf_by_band.get(band)
    if band_data:
        band_km = KaplanMeierFitter()
        # Get the underlying survival timeline from the fitted kmf
        times = band_data.survival_function_.index
        surv_probs = band_data.survival_function_.iloc[:, 0]
        ax.plot(times, surv_probs, label=f'Credit Band: {band}', linewidth=2)
        ax.fill_between(times, surv_probs * 0.95, surv_probs * 1.02, alpha=0.2)
    
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Predicted Survival Curve for New Applicant (Credit Score: {score})')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 24)
    
    # Add annotation for key time points
    if band_data:
        s12 = band_data.predict(12)
        s24 = band_data.predict(24)
        ax.axhline(y=s12, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=12, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'12m: {s12:.1%}', xy=(12, s12), xytext=(14, s12 + 0.05),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
        ax.annotate(f'24m: {s24:.1%}', xy=(24, s24), xytext=(20, s24 - 0.08),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/survival-analysis-time-to-default/reports/applicant_survival.png', dpi=150)
    plt.close()
    
    return {'credit_band': band, 'partial_hazard': partial_hazard.values[0]}

if __name__ == "__main__":
    from data_loader import generate_loan_data
    from kaplan_meier import create_credit_bands
    from cox_ph import fit_cox_ph
    
    df = generate_loan_data()
    df = create_credit_bands(df)
    
    cph, cox_results = fit_cox_ph(df)
    
    kmf_dict = {}
    for band in ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']:
        band_data = df[df['credit_band'] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(band_data['time_end'], band_data['event_default'])
        kmf_dict[band] = kmf
    
    new_applicant = {
        'income': 85000,
        'credit_score': 720,
        'employment_years': 6,
        'debt_to_income': 0.35,
        'loan_amount': 180000,
        'interest_rate': 8.5,
        'LTV_ratio': 0.75
    }
    
    result = predict_survival_for_applicant(cph, new_applicant, kmf_dict)
    print(f"\nPrediction: {result}")