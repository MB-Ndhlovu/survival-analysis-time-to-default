# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, answering the question: **when** is a borrower likely to default, not just **if** they will default.

## Survival Analysis Concepts

- **Survival Function S(t)**: Probability that the event (default) has not occurred by time t
- **Hazard Function h(t)**: Instantaneous rate of default at time t, given survival up to t
- **Censoring**: Observations where default hasn't occurred by the observation end (censored at 24 months)
- **Kaplan-Meier Estimator**: Non-parametric estimator of the survival function
- **Cox Proportional Hazards Model**: Semi-parametric model for the hazard ratio

## Business Application

Survival analysis provides more granular risk information than binary default models:
- **12-month and 24-month survival probabilities** by credit segment
- **Median time to default** — the time at which 50% of borrowers in a segment have defaulted
- **Hazard ratios** from Cox PH show which factors most increase default risk
- **Risk chiizer** bins continuous variables into interpretable risk categories

## Files

| File | Description |
|------|-------------|
| `src/data_loader.py` | Generates 5000 synthetic loan records with survival data |
| `src/kaplan_meier.py` | Fits KM curves by credit score band, plots survival functions |
| `src/cox_ph.py` | Fits Cox PH model, computes hazard ratios |
| `src/chiizer.py` | Bins variables into risk categories, computes segment survival |
| `src/predict_survival.py` | Predicts survival curve for a new loan applicant |
| `run_pipeline.py` | Executes full pipeline, saves results |

## Key Outputs

- Kaplan-Meier survival curves by credit score band
- Median survival times per segment
- Cox PH coefficients and hazard ratios
- 12-month and 24-month survival probabilities
- Predicted survival curve for new applicant

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py
```