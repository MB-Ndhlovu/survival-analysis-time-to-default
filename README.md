# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default classification to predict **when** default is likely to occur—not just whether it will happen.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis is a branch of statistics that models time-to-event data. In credit risk, the "event" is loan default, and the "time" is months until default.

### Key Concepts
- **Survival Function S(t)**: The probability that a loan survives beyond time t without defaulting
- **Hazard Function h(t)**: The instantaneous rate of default at time t, given survival up to t
- **Censoring**: When a loan does not default during the observation period—it is "censored" at the observation end
- **Kaplan-Meier Estimator**: Non-parametric estimator of the survival function
- **Cox Proportional Hazards Model**: Semi-parametric model for the hazard function

### Why Survival Analysis for Credit Risk?
- Binary models only answer: "Will this loan default?"
- Survival analysis answers: "When will this loan default, and what's the probability it survives 12/24 months?"
- Enables better pricing, provisioning, and risk management
- Handles censored data correctly (loans that haven't defaulted yet)

## Business Application

This analysis segments borrowers by credit score bands and identifies which factors most influence default timing:

| Credit Band | Score Range | Risk Profile |
|-------------|-------------|--------------|
| Deep Subprime | < 580 | High risk, fast default timing |
| Subprime | 580-669 | Elevated risk |
| Near Prime | 670-739 | Moderate risk |
| Prime | 740+ | Low risk, extended survival |

## Files

- `src/data_loader.py` — Synthetic loan data generator (5000 loans, ~35% censored at 24 months)
- `src/kaplan_meier.py` — Kaplan-Meier survival curves by credit band
- `src/cox_ph.py` — Cox Proportional Hazards model for risk factor analysis
- `src/chiizer.py` — Risk chiizer: bin variables into categories and compare survival
- `src/predict_survival.py` — Predict survival curve for a new applicant
- `run_pipeline.py` — Execute full analysis pipeline

## Requirements

```
lifelines>=0.27.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
```