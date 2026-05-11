# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, predicting **when** a loan is likely to default — not just whether it will default. This temporal dimension provides significantly more actionable intelligence for credit risk management.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a branch of statistics that models time-to-event data. In credit risk, the "event" is loan default, and the "time" is months until default.

Key advantages over binary classification:
- **Handles censored data**: Loans that haven't defaulted yet (paid off, still active) provide partial information
- **Temporal predictions**: Forecast default probability at specific time horizons (12mo, 24mo)
- **Hazard modeling**: Quantify the instantaneous risk of default at each point in time

### Key Terms

| Term | Definition |
|------|------------|
| **Survival Function S(t)** | Probability that a loan survives beyond time t |
| **Hazard Function h(t)** | Instantaneous default rate at time t, given survival to t |
| **Censoring** | Observation ends before default occurs (right-censored) |
| **Median Survival Time** | Time at which S(t) = 0.50 (50% survival) |

### Methods Used

1. **Kaplan-Meier Estimator**: Non-parametric survival curves by cohort
2. **Cox Proportional Hazards**: Semi-parametric regression for hazard ratios
3. **Risk Chiizer**: Discretized risk stratification from continuous variables

## Business Application

Traditional credit models output a binary probability of default (PD) over some arbitrary timeframe. Survival analysis provides:

- **Rolling default curves** by credit score band
- **Time-to-default distributions** for loss forecasting
- **Hazard ratios** quantifying the impact of each risk factor
- **Applicant-level survival curves** for personalized risk pricing

This allows lenders to:
- Set risk-based pricing by time horizon
- Forecast loss reserves more accurately
- Identify early warning signals before default occurs

## Project Structure

```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Synthetic loan data generation
│   ├── kaplan_meier.py     # KM curves by credit band
│   ├── cox_ph.py           # Cox PH regression
│   ├── chiizer.py          # Risk chiizer binning
│   └── predict_survival.py # New applicant predictions
└── reports/
    └── survival_results.json
```

## Results Summary

Key outputs from the pipeline:
- Survival curves stratified by credit score band
- Median time to default per segment
- Cox PH hazard ratios (which factors accelerate default risk)
- 12-month and 24-month survival probabilities by segment
- Predicted survival curve for a new loan applicant

## Installation

```bash
pip install -r requirements.txt
python run_pipeline.py
```