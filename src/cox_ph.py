import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def fit_cox_ph(df):
    """Fit Cox Proportional Hazards model."""
    cph = CoxPHFitter()
    
    # Prepare features
    features = ['income', 'credit_score', 'employment_years', 'debt_to_income', 
                'loan_amount', 'interest_rate', 'LTV_ratio']
    
    # Standardize for better convergence
    df_model = df[features + ['time_end', 'event_default']].copy()
    
    # Log transform skewed features
    df_model['log_income'] = np.log(df_model['income'])
    df_model['log_loan'] = np.log(df_model['loan_amount'])
    
    # Final feature set
    final_features = ['log_income', 'credit_score', 'employment_years', 'debt_to_income',
                      'log_loan', 'interest_rate', 'LTV_ratio']
    
    df_model = df_model[final_features + ['time_end', 'event_default']]
    
    # Fit model
    cph.fit(df_model, duration_col='time_end', event_col='event_default')
    
    # Print summary
    print("=== Cox Proportional Hazards Model ===")
    cph.print_summary()
    
    # Extract coefficients and hazard ratios
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    
    results = {
        'concordance': round(cph.concordance_index_, 4),
        'coefficients': {},
        'hazard_ratios': {}
    }
    
    print("\n=== Key Findings ===")
    for idx in summary.index:
        coef = summary.loc[idx, 'coef']
        hr = summary.loc[idx, 'hazard_ratio']
        p = summary.loc[idx, 'p']
        
        results['coefficients'][idx] = round(coef, 4)
        results['hazard_ratios'][idx] = round(hr, 4)
        
        direction = "increases" if coef > 0 else "decreases"
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        print(f"\n{idx}:")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  Hazard Ratio: {hr:.4f} ({direction} default risk){significance}")
    
    return cph, results

def get_top_risk_factors(results, n=5):
    """Get the top n risk factors by hazard ratio magnitude."""
    hr = results['hazard_ratios']
    sorted_hr = sorted(hr.items(), key=lambda x: abs(x[1] - 1), reverse=True)
    return sorted_hr[:n]

if __name__ == "__main__":
    from data_loader import generate_loan_data
    
    df = generate_loan_data()
    cph, results = fit_cox_ph(df)
    
    print("\n=== Top Risk Factors ===")
    top = get_top_risk_factors(results)
    for factor, hr in top:
        print(f"  {factor}: HR={hr}")