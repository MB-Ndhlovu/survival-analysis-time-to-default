"""
Cox Proportional Hazards model for default risk.
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def fit_cox_ph(df):
    """
    Fit Cox PH model using standard duration data.
    """
    df = df.copy()

    features = ['credit_score', 'income', 'employment_years',
                'debt_to_income', 'loan_amount', 'interest_rate', 'LTV_ratio']

    cph = CoxPHFitter()
    cph.fit(df[['time_end', 'event_default'] + features],
            duration_col='time_end',
            event_col='event_default')

    print("\n=== Cox PH Model Summary ===")
    cph.print_summary()

    # Extract coefficients and hazard ratios from summary
    summary = cph.summary.copy()
    coefs = summary['coef'].values
    hazard_ratios = np.exp(coefs)
    pvals = summary['p'].values

    coef_df = pd.DataFrame({
        'coefficient': coefs,
        'hazard_ratio': hazard_ratios,
        'p_value': pvals,
    }, index=features)

    print("\n=== Hazard Ratios ===")
    for feat in features:
        hr = coef_df.loc[feat, 'hazard_ratio']
        p = coef_df.loc[feat, 'p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{feat:20s}: HR={hr:.4f} {sig}")

    top_risk = coef_df.sort_values('hazard_ratio', ascending=False)
    print("\n=== Top 3 Risk Factors (Highest HR) ===")
    for i, (feat, row) in enumerate(top_risk.head(3).iterrows()):
        print(f"{i+1}. {feat}: HR={row['hazard_ratio']:.4f}")

    return cph, coef_df


if __name__ == "__main__":
    from data_loader import generate_loan_data
    df = generate_loan_data()
    cph, coefs = fit_cox_ph(df)