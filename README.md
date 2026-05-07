# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modeling, specifically to predict **when** a loan borrower is likely to default — not just whether they will default.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis is a branch of statistics that models the time until an event of interest occurs. In credit risk, the "event" is loan default, and the "time" is months or years until default occurs.

### Key Concepts
- **Survival Function S(t)**: The probability that the event (default) has not occurred by time t
- **Hazard Function h(t)**: The instantaneous rate of default at time t, given survival up to t
- **Censoring**: When a loan is still active at the end of the observation period (no default observed yet)
- **Kaplan-Meier Estimator**: Non-parametric estimate of the survival function from censored data
- **Cox Proportional Hazards Model**: Semi-parametric model for the relationship between covariates and survival time

### Why Survival Analysis for Credit Risk?
Traditional default models give binary outcomes (default / no default). Survival analysis adds the **time dimension**:

1. **Risk-adjusted pricing**: Price loans based on when default is likely
2. **Loss given default (LGD) timing**: Earlier defaults may have different recovery rates
3. **Portfolio duration risk**: Expected lifetime of portfolio exposures
4. **Early warning systems**: Identify borrowers trending toward default before it happens

## Business Application
This analysis answers questions like:
- What is the probability a borrower survives 12 months without defaulting?
- Which risk factors have the biggest impact on time-to-default?
- How does credit score affect the shape of the default hazard over time?
- For a new applicant, when is default most likely to occur?

## Project Structure
```
.
├── README.md
├── requirements.txt
├── run_pipeline.py
├── reports/
│   └── survival_results.json
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── kaplan_meier.py
    ├── cox_ph.py
    ├── chiizer.py
    └── predict_survival.py
```

## Files
- `src/data_loader.py` — Generates synthetic loan data (5,000 records)
- `src/kaplan_meier.py` — Fits KM curves by credit score band, computes median survival
- `src/cox_ph.py` — Fits Cox PH model, interprets coefficients and hazard ratios
- `src/chiizer.py` — Bins variables into risk categories, computes segment survival curves
- `src/predict_survival.py` — Predicts survival curve for a new loan applicant
- `run_pipeline.py` — Executes full analysis pipeline