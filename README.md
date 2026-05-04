# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand *when* default is likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis is a branch of statistics that models time-to-event data. In credit risk, the "event" is loan default, and we analyze how long it takes for defaults to occur.

### Key Concepts
- **Survival Function S(t)**: The probability that no default has occurred by time t
- **Hazard Function h(t)**: The instantaneous rate of default at time t, given survival to that point
- **Censoring**: When a loan is still active at the study end (not yet defaulted) — we know it survived at least that long
- **Median Survival Time**: Time at which 50% of loans have defaulted

### Methods Used
1. **Kaplan-Meier Estimator**: Non-parametric estimate of the survival function
2. **Cox Proportional Hazards Model**: Semi-parametric model for covariates' effect on hazard

## Business Application

Traditional credit scoring answers: *Will this borrower default?*

Survival analysis answers: *When will this borrower default, and how does that vary by risk profile?*

This allows lenders to:
- Optimize pricing by risk segment
- Set dynamic monitoring triggers based on time-varying risk
- Better estimate loss reserves and expected exposure at default

## Project Structure

```
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── kaplan_meier.py
│   ├── cox_ph.py
│   ├── chiizer.py
│   └── predict_survival.py
└── reports/
    └── survival_results.json
```

## Key Outputs

- Kaplan-Meier survival curves by credit score band
- Median time to default per segment
- Cox PH hazard ratios identifying key default drivers
- 12 and 24-month survival probabilities
- Predicted survival curve for new applicants