# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand **when** default is likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis is a branch of statistics that models time-to-event data. In credit risk, the "event" is loan default, and "survival" means the loan remaining in good standing.

Key advantages over binary classification:
- **Handles censored data**: Loans that haven't defaulted yet (censored at study end)
- **Time-varying risk**: Hazard rates that change over the loan lifecycle
- **Full distribution**: Not just probability of default, but the entire default timing curve

### Key Terms
- **Survival function S(t)**: Probability that default has not occurred by time t
- **Hazard function h(t)**: Instantaneous rate of default at time t, given survival to time t
- **Censoring**: Observation ends before default occurs (paid off, still active, lost to follow-up)
- **Median survival time**: Time at which S(t) = 0.5 (50% default rate)

### Methods Implemented
1. **Kaplan-Meier Estimator**: Non-parametric survival curves by segment
2. **Cox Proportional Hazards Model**: Semi-parametric regression for hazard ratios
3. **Risk Chiizer**: Binning continuous variables to compute segment-specific survival

## Business Application

Traditional default models answer: *"Will this loan default?"*

Survival analysis answers: *"When will this loan default, and what factors accelerate or delay default?"*

This enables:
- Better pricing by accounting for time-varying risk
- Early warning systems with predicted default timing
- Risk-adjusted return calculations over specific horizons
- Segment-specific loss given default (LGD) estimates

## Project Structure
```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Generate synthetic loan data
│   ├── kaplan_meier.py      # Kaplan-Meier survival curves
│   ├── cox_ph.py            # Cox PH hazard ratio estimation
│   ├── chiizer.py           # Risk chiizer binning
│   └── predict_survival.py  # Predict for new applicants
├── run_pipeline.py          # Execute full analysis
└── reports/
    └── survival_results.json
```