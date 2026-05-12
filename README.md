# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand *when* default is likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a family of statistical methods for analyzing time-to-event data. Unlike standard classification, it handles:

- **Censoring**: Observations that haven't experienced the event by the end of the study period
- **Time-dependent risk**: The hazard of default changes over the loan's lifetime
- **Non-normal distributions**: Default times rarely follow Gaussian distributions

### Key Terminology

- **Survival Function S(t)**: Probability that no default has occurred by time t
- **Hazard Function h(t)**: Instantaneous rate of default at time t, given survival to that point
- **Median Survival Time**: Time at which 50% of loans have defaulted (or 50% survival)
- **Censoring**: When a loan is paid off, refinanced, or hasn't defaulted by study end

### Kaplan-Meier Estimator

Non-parametric estimator of the survival function. Produces step functions that step down at each observed default time. Allows comparison of survival curves across groups using the log-rank test.

### Cox Proportional Hazards Model

Semi-parametric regression model: 

```
h(t) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)
```

Assumes covariates have multiplicative effect on baseline hazard. Provides hazard ratios for interpretable risk quantification.

## Business Application

### Why Survival Analysis for Credit Risk?

Traditional default models answer: **Will this loan default?** (Yes/No)

Survival analysis answers: **When will this loan default, and how does the risk evolve over time?**

This enables:
- Better loss reserves and capital allocation
- Early warning systems for portfolio deterioration  
- Risk-based pricing that accounts for duration
- Understanding which borrower characteristics drive timing, not just occurrence

### Credit Score Bands

| Band | Score Range | Risk Profile |
|------|-------------|--------------|
| Subprime | < 580 | High risk, early defaults |
| Near-Prime | 580-669 | Elevated risk |
| Prime | 670-739 | Moderate risk |
| Super-Prime | 740+ | Low risk, late defaults |

### Key Metrics

- **12-month survival probability**: P(no default in first year)
- **24-month survival probability**: P(no default in first two years)
- **Median time to default**: 50% point of cumulative default curve
- **Hazard ratios**: Relative risk from Cox model coefficients

## Project Structure

```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Generate synthetic loan data
│   ├── kaplan_meier.py      # Kaplan-Meier curves by credit band
│   ├── cox_ph.py            # Cox Proportional Hazards model
│   ├── chiizer.py           # Risk chiizer - bin analysis
│   └── predict_survival.py  # Predict survival for new applicant
└── reports/
    └── survival_results.json
```

## Files

### `src/data_loader.py`
Generates 5000 synthetic loan records with:
- `time_start`: Origin (0 months)
- `time_end`: Default or censorship time
- `event_default`: 1 if defaulted, 0 if censored
- `income`: Annual income ($)
- `credit_score`: FICO-equivalent score (300-850)
- `employment_years`: Years employed
- `debt_to_income`: Monthly debt payments / gross income
- `loan_amount`: Loan principal ($)
- `interest_rate`: Annual rate (%)
- `LTV_ratio`: Loan-to-value ratio

~35% of observations are right-censored at 24 months.

### `src/kaplan_meier.py`
- Fits Kaplan-Meier curves for each credit score band
- Computes median survival times
- Performs log-rank tests for group comparisons
- Plots survival functions with confidence bands

### `src/cox_ph.py`
- Fits Cox Proportional Hazards model
- Reports coefficients, standard errors, z-scores, p-values
- Computes hazard ratios with 95% CI
- Identifies top risk factors

### `src/chiizer.py`
- Bins continuous variables into risk categories
- Computes survival curves for each bin
- Validates monotonicity of risk ordering

### `src/predict_survival.py`
- Predicts survival function for a new applicant
- Provides 12-month and 24-month survival probabilities
- Plots individual predicted survival curve

## Running the Project

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Output:
- Kaplan-Meier plot saved to `reports/km_survival_curves.png`
- Results JSON saved to `reports/survival_results.json`