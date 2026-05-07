# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk, specifically modeling the **time until a loan defaults** rather than just predicting whether a default occurs. This gives lenders more actionable intelligence: not just *if* a borrower will default, but *when* default is most likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a family of statistical methods for analyzing time-to-event data. Unlike regular regression, it handles:

- **Censoring**: Some observations haven't experienced the event by the end of the study period. We know the event *hasn't* happened yet, but not *when* it will.
- **Truncated data**: Subjects who enter the study at different times.
- **Non-normal time distributions**: Event times are often heavily skewed.

### Key Functions

**Survival Function S(t)**: The probability that the event (default) has not occurred by time t.
$$S(t) = P(T > t)$$

**Hazard Function h(t)**: The instantaneous rate of event occurrence at time t, given survival until t.
$$h(t) = \frac{f(t)}{S(t)}$$

**Cumulative Hazard H(t)**: Total hazard accumulated up to time t.
$$H(t) = -\ln(S(t))$$

### Kaplan-Meier Estimator

A non-parametric estimator of the survival function that handles right-censored data:

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

Where:
- $t_i$ are observed event times
- $d_i$ is the number of events at $t_i$
- $n_i$ is the number at-risk subjects just before $t_i$

### Cox Proportional Hazards Model

A semi-parametric regression model for survival data:

$$h(t | X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \ldots)$$

- $h_0(t)$ is the baseline hazard (unspecified)
- $\exp(\beta_i)$ is the **hazard ratio** — a multiplier on hazard for a 1-unit increase in $X_i$
- HR > 1: factor increases default risk
- HR < 1: factor decreases default risk

## Business Application in Credit Risk

### Why Survival Analysis for Credit?

Traditional credit scoring models output a binary probability of default (PD). Survival analysis extends this by providing:

1. **Time-varying PD**: Default risk changes over the loan's lifetime
2. **Censoring handles ongoing loans**: Active loans that haven't defaulted are "censored" — they contribute information without diluting the analysis
3. **Segment-specific curves**: Different credit tiers have different survival trajectories
4. **Lifetime value modeling**: Better expected loss and NPL calculations

### Data Generating Process

Loan-level data with features including:
- `time_start`: Origin time (months since origination)
- `time_end`: Time of default or censoring
- `event_default`: 1 if default occurred, 0 if censored
- `income`, `credit_score`, `employment_years`, `debt_to_income`, `loan_amount`, `interest_rate`, `LTV_ratio`

### Key Outputs

- Kaplan-Meier survival curves by credit score band
- Median survival time (time to 50% default rate)
- Cox PH hazard ratios — which factors push default risk most
- 12-month and 24-month survival probabilities by segment
- Predicted survival curve for a new applicant

## Files

```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Generate 5000-row synthetic loan dataset
│   ├── kaplan_meier.py      # KM curves by credit band
│   ├── cox_ph.py            # Cox PH hazard ratios
│   ├── chiizer.py           # Risk chiizer — bin variables, compute curves
│   └── predict_survival.py  # Predict survival for new applicant
└── reports/
    └── survival_results.json
```

## Interpretation Guide

| Metric | Low Risk | High Risk |
|--------|----------|-----------|
| Survival curve | Declines slowly | Declines rapidly |
| Median survival time | High (many months) | Low (few months) |
| Hazard ratio | < 1 | > 1 |
| 12-month survival S(12) | Close to 1 | Well below 1 |

## Business Insight

> Survival analysis gives more information than binary default models — it tells you **WHEN** default is likely, not just **IF**.

This enables:
- **Proactive outreach** to borrowers approaching high-risk periods
- **Tiered monitoring schedules** based on where a loan sits on its survival curve
- **Risk-based pricing** refined by time-to-default estimates
- **Expected loss term structure** for accounting and capital requirements