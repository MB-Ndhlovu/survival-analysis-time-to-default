# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk, specifically modeling **time-to-default** for loans. Unlike binary default models that only predict IF default occurs, survival analysis predicts WHEN default is likely — providing richer information for pricing, provisioning, and credit decisioning.

## Survival Analysis Concepts

### Key Idea
Survival analysis estimates the probability that an event (default) has NOT occurred by time `t`. The **survival function** S(t) = P(T > t) where T is the time to default.

### Why Survival Analysis?
- Binary classifiers miss timing: a 10% probability of default in 12 months is very different from 10% over 5 years
- Survival analysis naturally handles **censoring**: loans that haven't defaulted yet (still alive) but we're still tracking
- Enables **time-varying** predictions: 12-month, 24-month survival curves

### Censoring
~35% of observations are **right-censored** at 24 months — the loan is still performing, but we stop observing it. This is realistic: many loans in a portfolio haven't defaulted yet but we're analyzing at a point in time.

### Methods Used

1. **Kaplan-Meier Estimator**: Non-parametric survival curve estimation
   - Produces step functions showing survival probability over time
   - Allows comparison across subgroups (e.g., credit score bands)
   - Computes median survival time (time at which 50% have defaulted)

2. **Cox Proportional Hazards Model**: Semi-parametric regression
   - Models hazard (instantaneous default rate) as a function of covariates
   - Coefficients tell us which factors increase/decrease default risk
   - Hazard ratios > 1 = increased risk, < 1 = protective

3. **Risk Chiizer**: Binning continuous variables into risk categories
   - Creates interpretable risk segments
   - Computes survival curves per segment for easy comparison

## Business Application

### Credit Risk Use Cases
- **Pricing**: Risk-based pricing using expected loss = PD × LGD × EAD, where PD is derived from survival curves
- **Provisioning**: Expected loss at each time horizon for IFRS 9/CECL compliance
- **Credit Decisioning**: 12/24-month default probabilities for loan approval
- **Portfolio Monitoring**: Segment-level survival curves for early warning systems

### Key Outputs
- Survival curves by credit score band
- Median time to default by segment
- Cox PH hazard ratios showing which factors matter most
- 12-month and 24-month survival probabilities by segment
- Predicted survival curve for new applicants

## Files

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Synthetic loan data generation
│   ├── kaplan_meier.py      # KM curves by credit band
│   ├── cox_ph.py            # Cox PH regression
│   ├── chiizer.py           # Risk categorization
│   └── predict_survival.py  # New applicant predictions
├── run_pipeline.py         # Execute full pipeline
└── reports/
    └── survival_results.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py
```

## Key Insight

> Survival analysis gives more information than binary default models — it tells you **WHEN** default is likely, not just **IF**.