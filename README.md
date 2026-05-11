# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to predict **when** a loan borrower is likely to default — not just the probability of default (as in binary classification). Survival analysis is essential in credit risk because it handles censored data properly and models the full time-to-event distribution.

## Core Concepts

### Survival Function S(t)
The probability that a borrower has **not** defaulted by time t:
S(t) = P(T > t)

### Hazard Function h(t)
The instantaneous rate of default at time t, given survival to that point.

### Kaplan-Meier Estimator
Non-parametric estimator of S(t) that naturally handles right-censored data.

### Cox Proportional Hazards Model
Semi-parametric regression model:
h(t | X) = h_0(t) · exp(β₁X₁ + β₂X₂ + ...)

The **hazard ratio** exp(βⱼ) tells us how much the hazard (default risk) increases per unit increase in predictor Xⱼ, holding other factors constant.

## Business Application

| Question | Binary Default Model | Survival Analysis |
|----------|---------------------|------------------|
| Will this borrower default? | Yes/No | Yes/No + probability |
| **When** will they default? | No | Conditional survival probabilities at each t |
| What if they survive 12 months? | No | S(12) — probability of surviving past 12 months |
| How does credit score affect timing? | No | HR: 1.5x hazard for subprime vs. prime |

## Key Outputs
- **Kaplan-Meier curves** by credit score band (Poor <580, Fair 580-669, Good 670-739, Excellent 740+)
- **Median time to default** for each band
- **Cox PH coefficients** — rank risk drivers by hazard ratio
- **12/24-month survival probabilities** by segment
- **Predicted survival curve** for new applicants

## Files
- `src/data_loader.py` — synthetic loan data with ~35% censored at 24 months
- `src/kaplan_meier.py` — KM curves per credit band
- `src/cox_ph.py` — Cox PH regression and hazard ratios
- `src/chiizer.py` — bin continuous variables into risk categories
- `src/predict_survival.py` — predict survival for new applicants
- `run_pipeline.py` — orchestrate full analysis pipeline

## Business Insight
Survival analysis gives more information than binary default models — it tells you **WHEN** default is likely, not just **IF**.
