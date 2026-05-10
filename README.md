# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, specifically to model **time-to-default** for loans. Unlike binary classification models that only predict *whether* a borrower will default, survival analysis predicts *when* default is likely to occur — enabling better risk pricing and proactive intervention.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a family of statistical methods for analyzing time-to-event data. Key concepts:

- **Survival Function S(t)**: The probability that an event (e.g., default) has NOT occurred by time `t`. `S(t) = P(T > t)`
- **Hazard Function h(t)**: The instantaneous rate of event occurrence at time `t`, given survival up to `t`
- **Censoring**: When a loan is prepaid, paid off, or the observation window closes before default occurs — these observations are "censored" rather than having an observed event

### Kaplan-Meier Estimator

Non-parametric estimator of the survival function. Handles right-censoring natively. Produces step functions that drop at each observed event time.

### Cox Proportional Hazards Model

Semi-parametric regression model for survival data:

```
h(t) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)
```

Assumes covariates have multiplicative effect on the hazard that is constant over time (proportional hazards assumption).

## Business Application in Credit Risk

### Why Survival Analysis for Credit?

Traditional default models treat all defaulted loans equally, regardless of when default occurred. Survival analysis adds the **time dimension**:

| Model Type | Answers |
|------------|---------|
| Binary Classification | Will this borrower default? (Yes/No) |
| Survival Analysis | When is default most likely? What % survive past 12 months? |

### Practical Applications

1. **Risk-Based Pricing**: Loans with worse survival curves get higher rates
2. **Provisioning**: Expected loss = Σ (survival_probability × exposure_at_time × LGD)
3. **Early Warning Systems**: Identify borrowers whose survival curve drops sharply at specific months
4. **Portfolio Monitoring**: Track how segment-level survival curves shift over time

### Credit Score Bands

We segment borrowers into standard credit score bands:

- **Deep Subprime**: Score < 580
- **Subprime**: 580–669
- **Near Prime**: 670–739
- **Prime**: 740+

Higher scores → better survival curves (lower default hazard).

## Files

```
├── README.md
├── requirements.txt
├── run_pipeline.py          # Execute full pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Generate synthetic loan data
│   ├── kaplan_meier.py       # Kaplan-Meier curves by segment
│   ├── cox_ph.py             # Cox PH regression model
│   ├── chiizer.py            # Risk chiizer — bin continuous vars
│   └── predict_survival.py   # Predict survival for new applicant
└── reports/
    └── survival_results.json # JSON output of key metrics
```

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Key Outputs

- Kaplan-Meier survival curves by credit score band
- Median time to default for each segment
- Cox PH coefficients and hazard ratios
- 12-month and 24-month survival probabilities by segment
- Predicted survival curve for a new loan applicant