# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, specifically to predict **when** a borrower is likely to default on a loan — not just the probability of default at a single point in time.

## Survival Analysis Concepts

- **Survival Function S(t)**: The probability that a loan has NOT defaulted by time `t`
- **Hazard Function h(t)**: The instantaneous default rate at time `t`, given survival up to `t`
- **Censoring**: Loans that haven't defaulted by the observation end date (24 months) are "censored" — they contribute information about survival up to that point but we don't observe their ultimate outcome
- **Kaplan-Meier Estimator**: Non-parametric estimator of the survival function
- **Cox Proportional Hazards Model**: Semi-parametric model for the hazard function, allowing us to quantify how different risk factors affect the hazard (default risk)

## Business Application in Credit Risk

Traditional credit scoring models answer: *"Will this borrower default?"*

Survival analysis answers: *"When is this borrower most likely to default, and which factors accelerate or reduce that risk?"*

This is more informative for:
- **Pricing**: Adjust interest rates based on expected time-to-default
- **Provisioning**: Reserve capital proportional to expected loss timing
- **Early warning systems**: Identify borrowers at risk before default occurs
- **Portfolio management**: Understand the temporal distribution of default risk across the portfolio

## Files

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Generate synthetic loan data with time-to-default observations |
| `src/kaplan_meier.py` | Kaplan-Meier survival curves by credit score band |
| `src/cox_ph.py` | Cox Proportional Hazards model — hazard ratios per risk factor |
| `src/chiizer.py` | Risk chiizer — bin variables into risk categories and compare curves |
| `src/predict_survival.py` | Predict survival curve for a new loan applicant |
| `run_pipeline.py` | Execute full analysis pipeline |

## Credit Score Bands

| Band | Score Range | Risk Level |
|------|-------------|------------|
| Poor | < 580 | High risk |
| Fair | 580–669 | Moderate risk |
| Good | 670–739 | Lower risk |
| Excellent | 740+ | Lowest risk |

## Key Metrics

- **12-month survival probability**: % of loans not defaulted within 12 months
- **24-month survival probability**: % of loans not defaulted within 24 months
- **Median survival time**: Time at which 50% of loans have defaulted
- **Hazard ratios**: Multiplicative effect of each factor on the instantaneous default rate