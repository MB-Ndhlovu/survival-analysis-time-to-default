# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand *when* default is likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a branch of statistics that models time-to-event data. Unlike classification models that predict *if* an event occurs, survival models predict *when* it occurs — while properly handling censored observations (loans that haven't defaulted yet but also haven't reached the end of the observation window).

### Key Concepts

- **Survival Function S(t)**: The probability that the event (default) has not occurred by time t
- **Hazard Function h(t)**: The instantaneous rate of default at time t, given survival up to t
- **Censoring**: When a loan exits the observation window without experiencing the event (paid off, still active at cutoff)
- **Median Survival Time**: The time at which 50% of loans have defaulted (or S(t) = 0.5)

### Methods Used

1. **Kaplan-Meier Estimator**: Non-parametric estimator of the survival function, allows comparison across groups
2. **Cox Proportional Hazards Model**: Semi-parametric model estimating the effect of covariates on hazard
3. **Risk Chiizer**: Binning continuous variables into risk categories for segment-level survival curves

## Business Application in Credit Risk

Traditional default models output a probability of default (PD) over a fixed horizon. Survival analysis enriches this by:

- Providing time-varying default probabilities
- Enabling precise 12-month and 24-month survival rates by segment
- Identifying which risk factors accelerate or delay default timing
- Supporting better pricing and provisioning decisions

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

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Key Outputs

- Kaplan-Meier survival curves by credit score band
- Median time-to-default per segment
- Cox PH hazard ratios identifying default drivers
- 12-month and 24-month survival probabilities
- Predicted survival curve for new loan applicants