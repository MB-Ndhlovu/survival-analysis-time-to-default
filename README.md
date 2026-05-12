# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to model time-to-default for credit risk. Unlike binary default models that only predict *whether* a borrower will default, survival analysis reveals *when* default is most likely to occur — providing materially more information for credit portfolio management, pricing, and reserves.

## Survival Analysis Concepts

### Key Ideas
- **Survival function S(t)**: Probability that a borrower survives (does not default) beyond time `t`
- **Hazard function h(t)**: Instantaneous rate of default at time `t`, given survival to that point
- **Censoring**: Many borrowers will never default within the observation window — their true time-to-default is *censored*. Survival analysis handles this correctly via maximum-likelihood estimation.
- **Kaplan-Meier**: Non-parametric estimator of the survival function — produces step-function curves stratified by cohort
- **Cox Proportional Hazards**: Semi-parametric model that relates covariates (credit score, LTV, DTI, etc.) to the hazard rate, giving interpretable hazard ratios

### Business Application in Credit Risk
| Insight | Binary Model | Survival Analysis |
|---|---|---|
| "Will this borrower default?" | ✓ | ✓ |
| "When will defaults peak?" | ✗ | ✓ |
| "Which segment has shortest median survival?" | ✗ | ✓ |
| "What is 12-month survival probability by band?" | ✗ | ✓ |
| "How does DTI affect timing of default?" | ✗ | ✓ |

A loan book with 35% censoring at 24 months is realistic — most loans are healthy and censored by the observation cutoff. Ignoring censoring biases your model toward longer observed times.

## Files
```
.
├── README.md
├── requirements.txt
├── run_pipeline.py
├── reports/
│   └── survival_results.json
└── src/
    ├── __init__.py
    ├── data_loader.py      ← generates 5000 synthetic loans
    ├── kaplan_meier.py     ← KM curves by credit band
    ├── cox_ph.py           ← Cox PH model, hazard ratios
    ├── chiizer.py          ← risk chiizer — binned survival
    └── predict_survival.py ← predict for new applicant
```

## Usage
```bash
pip install -r requirements.txt
python run_pipeline.py
```