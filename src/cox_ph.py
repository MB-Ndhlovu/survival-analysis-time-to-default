"""Cox Proportional Hazards model for default prediction."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare covariates for Cox PH model."""
    X = df.copy()
    
    X['log_income'] = np.log1p(X['income'])
    X['log_loan_amount'] = np.log1p(X['loan_amount'])
    X['high_dti'] = (X['debt_to_income'] > 0.36).astype(int)
    X['high_ltv'] = (X['LTV_ratio'] > 0.8).astype(int)
    X['short_employment'] = (X['employment_years'] < 2).astype(int)
    
    return X


def fit_cox_ph(df: pd.DataFrame) -> tuple:
    """Fit Cox PH model.
    
    Returns:
        Tuple of (cph fitted model, summary DataFrame).
    """
    X = prepare_covariates(df)
    
    covariates = [
        'credit_score', 'employment_years', 'debt_to_income',
        'loan_amount', 'LTV_ratio'
    ]
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        X[['time_end', 'event_default'] + covariates],
        duration_col='time_end',
        event_col='event_default'
    )
    
    return cph, cph.summary


def get_hazard_ratios(cph) -> pd.DataFrame:
    """Extract hazard ratios with confidence intervals."""
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    
    if 'coef lower 95%' in summary.columns:
        summary['hr_lower'] = np.exp(summary['coef lower 95%'])
        summary['hr_upper'] = np.exp(summary['coef upper 95%'])
    elif '95% lower' in summary.columns:
        summary['hr_lower'] = np.exp(summary['95% lower'])
        summary['hr_upper'] = np.exp(summary['95% upper'])
    else:
        summary['hr_lower'] = None
        summary['hr_upper'] = None
    
    return summary[['coef', 'hazard_ratio', 'hr_lower', 'hr_upper', 'p']]


def interpret_coefficients(hr_df: pd.DataFrame) -> list:
    """Interpret Cox PH coefficients."""
    interpretations = []
    
    for _, row in hr_df.iterrows():
        var = row.get('variable', row.name)
        hr = row['hazard_ratio']
        p = row['p']
        
        direction = "increases" if row['coef'] > 0 else "decreases"
        
        if p < 0.05:
            sig = "significantly"
        else:
            sig = "non-significantly"
        
        interpretations.append({
            'variable': str(var),
            'hazard_ratio': round(hr, 3),
            'direction': direction,
            'significant': sig,
            'p_value': round(p, 4)
        })
    
    return interpretations


def run_cox_analysis(df: pd.DataFrame) -> dict:
    """Run full Cox PH analysis."""
    cph, summary = fit_cox_ph(df)
    hr_df = get_hazard_ratios(cph)
    interpretations = interpret_coefficients(hr_df)
    
    return {
        'cph': cph,
        'hazard_ratios': hr_df.reset_index().to_dict('records'),
        'interpretations': interpretations,
        'concordance_index': round(cph.concordance_index_, 4)
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    
    df = load_data()
    result = run_cox_analysis(df)
    
    print("=== Cox Proportional Hazards Model ===")
    print(f"Concordance Index: {result['concordance_index']}")
    print("\n=== Hazard Ratios ===")
    for hr in result['hazard_ratios']:
        sig = "*" if hr['p_value'] < 0.05 else ""
        print(f"  {hr['variable']}: HR={hr['hazard_ratio']:.3f} {sig}")
    
    print("\n=== Key Risk Factors ===")
    for interp in sorted(result['interpretations'], key=lambda x: x['hazard_ratio'], reverse=True)[:5]:
        print(f"  {interp['variable']}: HR={interp['hazard_ratio']}, {interp['direction']} risk ({interp['significant']})")