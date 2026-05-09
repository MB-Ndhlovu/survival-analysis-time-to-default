# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modeling, specifically analyzing **time-to-default** for loans. Unlike binary default classifiers that only predict *if* default occurs, survival analysis reveals **when** default is likely — enabling better risk pricing and capital reserving.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis models the time until an event occurs (e.g., loan default, equipment failure, customer churn). It handles two key complexities that traditional regression cannot:

1. **Censoring**: Some observations haven't experienced the event by the end of the observation period. Simply dropping them loses information. Survival models incorporate censored data correctly.

2. **Non-Normal Time Distribution**: Time-to-event data is often skewed and bounded at zero. Survival models don't assume normality.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Survival Function S(t)** | Probability that the event has NOT occurred by time t |
| **Hazard Function h(t)** | Instantaneous rate of event occurrence at time t |
| **Kaplan-Meier Estimator** | Non-parametric estimate of S(t) from censored data |
| **Cox Proportional Hazards** | Semi-parametric model relating covariates to hazard |
| **Hazard Ratio** | Relative hazard comparing two groups (HR>1 = higher risk) |
| **Median Survival Time** | Time at which S(t) = 0.5 |

### Censoring in This Dataset
~35% of loans are **right-censored** at 24 months — they haven't defaulted by observation end but may default later. Treating them as "no default" would understate risk; survival analysis accounts for this.

## Business Application: Credit Risk

### Why Survival Analysis for Lending?
- **Risk Pricing**: A 12-month survival curve tells you default probability by month, enabling risk-based pricing
- **Expected Loss (EL)**: EL = PD × LGD × EAD, but PD varies over loan life — survival curves give month-by-month PD
- **Capital Reserve**: Regulatory capital (Basel III) benefits from precise loss timing
- **Early Warning**: Identify borrowers trending toward default before it happens

### Credit Score Bands Analyzed
- **< 580**: Subprime — high risk
- **580-669**: Near-prime — elevated risk
- **670-739**: Prime — moderate risk
- **740+**: Super-prime — low risk

## Files

```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Generate synthetic loan data
│   ├── kaplan_meier.py     # Kaplan-Meier curves by segment
│   ├── cox_ph.py           # Cox Proportional Hazards model
│   ├── chiizer.py          # Risk chiizer — binning + curves
│   └── predict_survival.py # Predict survival for new applicant
├── run_pipeline.py         # Execute full analysis pipeline
└── reports/
    └── survival_results.json
```

## Key Outputs

1. **Kaplan-Meier Survival Curves** by credit score band
2. **Median Time-to-Default** — time at which 50% of loans in each band have defaulted
3. **Cox PH Coefficients** — which factors (income, DTI, LTV) drive default risk most
4. **12/24-Month Survival Probabilities** by segment
5. **Predicted Survival Curve** for a new loan applicant

## The Core Insight

> Binary default models answer: *"Will this loan default?"*
> Survival analysis answers: *"When will this loan default, and what's the probability over time?"*

This temporal dimension is critical for pricing, reserves, and early intervention.