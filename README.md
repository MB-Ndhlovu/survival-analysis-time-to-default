# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand **when** default is likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis is a branch of statistics that models time-to-event data. In credit risk, the "event" is loan default, and the "time" is how long until default occurs.

### Key Concepts
- **Survival Function S(t)**: The probability that default has not occurred by time t
- **Hazard Function h(t)**: The instantaneous rate of default at time t, given survival until t
- **Censoring**: When we don't observe the event (e.g., loan paid off, still active at end of observation period)
- **Median Survival Time**: The time at which 50% of loans have defaulted

### Methods Used
1. **Kaplan-Meier Estimator**: Non-parametric estimate of survival function
2. **Cox Proportional Hazards Model**: Semi-parametric model for covariates' effect on hazard

## Business Application

Traditional credit scoring answers: *"Will this borrower default?"*

Survival analysis answers: *"When is this borrower most likely to default, and what factors drive the timing?"*

This enables:
- Better loan pricing by time-of-default risk
- Early warning systems for portfolio management
- Targeted intervention strategies

## Project Structure
```
.
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

## Credit Score Bands
- **< 580**: Deep subprime
- **580-669**: Subprime
- **670-739**: Near-prime
- **740+**: Prime