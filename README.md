# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modeling, specifically predicting **time-to-default** for loan portfolios. Unlike binary default classifiers, survival analysis estimates the probability of default over time, enabling better risk pricing and reserve planning.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis studies the time until an event occurs (time-to-event). In credit risk:
- **Event**: Default (loan charged off)
- **Time**: Months since origination
- **Censoring**: Accounts that paid off, were current at observation end, or were sold

### Key Concepts
- **Survival Function S(t)**: Probability that an account survives beyond time t (hasn't defaulted)
- **Hazard Function h(t)**: Instantaneous failure rate at time t, given survival to t
- **Kaplan-Meier Estimator**: Non-parametric survival curve estimation
- **Cox Proportional Hazards**: Semi-parametric model for covariates' effect on hazard

### Censoring Mechanism
~35% of observations are right-censored at 24 months — accounts that were still performing (paid off, refinanced, or still active) without experiencing default.

## Business Application

### Why Survival Analysis for Credit Risk?
Traditional default models answer: "Will this loan default?" (binary)
Survival analysis answers: "When is this loan likely to default?" (temporal)

This enables:
1. **Better risk pricing**: Higher expected loss for accounts defaulting earlier
2. **Reserve planning**: Expected loss curves by month for reserve allocation
3. **Portfolio segmentation**: Identify high-risk cohorts by survival profile
4. **Life-of-loan profitability**: NPV calculation accounting for timing uncertainty

### Credit Score Bands Analyzed
| Band | Score Range | Risk Profile |
|------|-------------|--------------|
| Deep Subprime | < 580 | Highest risk |
| Subprime | 580-669 | Elevated risk |
| Near Prime | 670-739 | Moderate risk |
| Prime | 740+ | Lowest risk |

## Files

- `src/data_loader.py` — Synthetic loan data generation (5000 accounts)
- `src/kaplan_meier.py` — Kaplan-Meier survival curves by credit band
- `src/cox_ph.py` — Cox Proportional Hazards regression model
- `src/chiizer.py` — Risk chiizer: bin variables into risk categories
- `src/predict_survival.py` — Predict survival for new applicants
- `run_pipeline.py` — Execute full analysis pipeline

## Requirements
```
lifelines>=0.27.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Usage
```bash
pip install -r requirements.txt
python run_pipeline.py
```