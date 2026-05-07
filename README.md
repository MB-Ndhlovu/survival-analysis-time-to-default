# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, answering the question every lender wants to solve: **not just if a borrower will default, but when**.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a branch of statistics that models time-to-event data. Unlike classical regression, it gracefully handles **censoring** — observations where the event hasn't occurred yet (e.g., a loan still performing, customer still active).

Key functions:
- **Survival function S(t)**: Probability that the event (default) has NOT occurred by time t
- **Hazard function h(t)**: Instantaneous rate of event occurrence at time t, given survival up to t
- **Cumulative hazard H(t)**: Total hazard accumulated up to time t

### Why Survival Analysis for Credit Risk?

Traditional default models output binary predictions (will default / won't default). Survival analysis provides:

1. **Time-varying predictions**: Probability of default in month 6 vs month 24
2. **Censored data handling**: Loans still performing get proper treatment, not dropped
3. **Hazard ratios**: Quantify how much each factor increases/decreases default risk
4. **Customer lifetime value**: Better LTV estimates when you know default timing

### Key Methods Implemented

1. **Kaplan-Meier Estimator**: Non-parametric survival curve estimation. Produces step functions showing survival probability over time. Compare curves across credit score bands.

2. **Cox Proportional Hazards Model**: Semi-parametric model where hazard = base_hazard × exp(β₁x₁ + β₂x₂ + ...). Coefficients give hazard ratios — interpretable as "risk multipliers."

3. **Risk Chiizer**: Bin continuous variables into risk categories and compare survival curves. Finds optimal cut points that separate risk levels.

## Business Application

### Credit Risk Pipeline

For a portfolio of 5,000 loans, survival analysis answers:
- What percentage of subprime loans survive beyond 24 months?
- Which factors most accelerate default timing?
- What's the predicted survival curve for a new applicant?

### Model Outputs

| Output | Description |
|--------|-------------|
| Kaplan-Meier curves | Survival probability by credit score band |
| Median survival time | Time at which 50% of loans have defaulted |
| Hazard ratios | Risk multipliers per factor (Cox PH) |
| 12/24-month survival | Portfolio retention rates by segment |
| Predicted survival curve | Individual applicant time-to-default estimate |

### Business Insight

> A 650 credit score borrower has 3× higher hazard than a 750 scorer. But the Kaplan-Meier curve shows the critical difference: subprime loans start defaulting heavily at month 8, while prime loans don't accelerate until month 18. This timing information enables proactive intervention and better pricing.

## Project Structure

```
survival-analysis-time-to-default/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Synthetic loan data generation
│   ├── kaplan_meier.py         # KM curves by credit band
│   ├── cox_ph.py               # Cox PH hazard model
│   ├── chiizer.py              # Risk binning and chiizer
│   └── predict_survival.py     # New applicant prediction
├── run_pipeline.py             # Execute full analysis
└── reports/
    └── survival_results.json   # Output metrics
```