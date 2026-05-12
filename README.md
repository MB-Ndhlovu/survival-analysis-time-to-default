# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, predicting not just *whether* a loan defaults, but *when* default is most likely to occur.

## Survival Analysis Concepts

### Why Survival Analysis?

Traditional credit scoring models output a binary prediction: default / no default. Survival analysis goes further by modeling the **time to event**, answering:
- What is the probability a borrower survives (doesn't default) beyond month 12? 24?
- Which risk factors accelerate or delay default timing?
- How does default risk change over the life of the loan?

### Key Terms

- **Survival Function S(t)**: Probability that the event (default) has not occurred by time t
- **Hazard Function h(t)**: Instantaneous rate of default at time t, given survival to that point
- **Censoring**: Observations where default hasn't occurred by the observation end — treated differently in estimation
- **Kaplan-Meier Estimator**: Non-parametric estimate of S(t) from censored data
- **Cox Proportional Hazards**: Semi-parametric model relating covariates to hazard rate

### Business Application

In credit risk, survival analysis enables:
- **Better pricing**: Adjust loan pricing based on expected time-to-default
- **Provision calculations**: More accurate expected loss estimates
- **Portfolio monitoring**: Track cohort survival curves over time
- **Early warning systems**: Identify borrowers approaching high-risk periods

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