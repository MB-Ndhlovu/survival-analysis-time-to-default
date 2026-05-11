# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, predicting **when** a borrower is likely to default rather than merely **if** they will default. This temporal dimension is critical for pricing, provisioning, and capital allocation in lending portfolios.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis models the time until an event of interest occurs. In credit risk, the event is **default** (loan non-repayment). Unlike classical classification, survival models handle:

- **Censoring**: Many loans haven't defaulted by the observation date — they are "still alive"
- **Time-varying risk**: Default probability changes over the loan's lifecycle
- **Non-parametric estimation**: No assumptions about the underlying distribution

### Key Terms

| Term | Definition |
|------|------------|
| **Survival Function S(t)** | Probability that default has NOT occurred by time t |
| **Hazard Function λ(t)** | Instantaneous default rate at time t, given survival until t |
| **Censoring** | Observation ends before default (right-censored) |
| **Kaplan-Meier** | Non-parametric estimate of S(t) from censored data |
| **Cox PH** | Semi-parametric proportional hazards model |

### The Survival Function

$$S(t) = P(T > t)$$

Where T is the random variable for time-to-default. S(t) gives the probability of surviving (not defaulting) past time t.

### Censoring Mechanism

~35% of observations are right-censored at 24 months (loan still performing, observation ended). This is **non-informative censoring** — the fact that we stopped observing a borrower doesn't change their underlying risk.

## Business Application

### Why Survival Analysis for Credit Risk?

Standard binary default models answer: "Will this borrower default?" (0/1)

Survival analysis answers: "When is default most likely? At what point in the loan lifecycle does risk peak?"

This enables:

1. **Better pricing**: Risk-based pricing with time-dependent component
2. **Provisioning**: Accurate expected loss estimates at different horizons
3. **Portfolio management**: Identify when in the loan lifecycle risk concentrates
4. **Early warning systems**: Target interventions at high-risk time windows

### Credit Score Bands

We bin borrowers into risk segments matching industry standards:

| Band | Score Range | Risk Profile |
|------|-------------|--------------|
| Deep Subprime | < 580 | Highest risk |
| Subprime | 580–669 | Elevated risk |
| Near Prime | 670–739 | Moderate risk |
| Prime | ≥ 740 | Lowest risk |

## Methodology

1. **Data Generation**: Synthetic loan dataset (n=5000) with realistic distributions
2. **Kaplan-Meier Estimation**: Non-parametric survival curves by credit band
3. **Cox Proportional Hazards**: Identify which factors accelerate or delay default
4. **Risk Chiizer**: Discretize continuous variables and compute segment-specific survival
5. **Prediction**: Conditional survival curves for new applicants

## Files

```
survival-analysis-time-to-default/
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
- Median survival time (time to 50% default rate)
- Cox PH hazard ratios — which factors matter most
- 12-month and 24-month survival probabilities by segment
- Predicted survival curve for a new applicant