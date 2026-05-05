import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def fit_cox_ph(df, duration_col='time_end', event_col='event_default'):
    """Fit Cox Proportional Hazards model."""
    features = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                'loan_amount', 'interest_rate', 'ltv_ratio']

    df_model = df[features + [duration_col, event_col]].copy()

    # Log-transform income and loan_amount to handle skewness
    df_model['log_income'] = np.log(df_model['income'])
    df_model['log_loan_amount'] = np.log(df_model['loan_amount'])

    features_model = ['log_income', 'credit_score', 'employment_years', 'debt_to_income',
                      'log_loan_amount', 'interest_rate', 'ltv_ratio']

    cph = CoxPHFitter()
    cph.fit(df_model[features_model + [duration_col, event_col]],
            duration_col=duration_col, event_col=event_col)

    # Extract coefficients and compute hazard ratios
    summary = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].copy()
    summary.columns = ['coefficient', 'hazard_ratio', 'std_error', 'p_value']
    summary['significant'] = summary['p_value'] < 0.05

    results = {
        'concordance_index': cph.concordance_index_,
        'log_likelihood': cph.log_likelihood_,
        'coefficients': summary.to_dict('index')
    }

    return results, cph

def print_cox_results(results):
    """Print Cox PH results in readable format."""
    print("\n=== Cox Proportional Hazards Model ===")
    print(f"Concordance Index: {results['concordance_index']:.4f}")
    print("\nHazard Ratios (HR > 1 = higher default risk):")
    print("-" * 70)

    coef_df = pd.DataFrame(results['coefficients']).T
    coef_df = coef_df.sort_values('hazard_ratio', ascending=False)

    for var in coef_df.index:
        row = coef_df.loc[var]
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {var:20s}: HR={row['hazard_ratio']:.4f}  (coef={row['coefficient']:.4f})  p={row['p_value']:.4f} {sig}")

    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")
    return coef_df

if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    results, cph = fit_cox_ph(df)
    print_cox_results(results)