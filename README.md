# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling. Unlike traditional binary default models, survival analysis answers not just *if* a default will occur, but *when* — providing richer information for risk assessment, pricing, and capital allocation.

## Survival Analysis Concepts

**Survival Function S(t)** — The probability that a loan survives beyond time t without default.

**Hazard Function h(t)** — The instantaneous failure rate at time t, given survival up to t.

**Censoring** — When a loan is paid off, refinanced, or hasn't defaulted by the observation end, we only know it survived to that point. This is called right-censoring.

**Kaplan-Meier Estimator** — Non-parametric estimator of the survival function that properly handles censored data.

**Cox Proportional Hazards Model** — Semi-parametric model that estimates the effect of covariates on the hazard rate.

## Business Application

For credit risk managers, survival analysis enables:
- **Better risk pricing** — Price loans based on expected default timing, not just probability
- **Loss forecasting** — Estimate when defaults are most likely to occur
- **Portfolio management** — Identify high-risk segments early
- **Capital requirements** — More accurate expected loss calculations

## Files

- `src/data_loader.py` — Generate synthetic loan data with survival outcomes
- `src/kaplan_meier.py` — Kaplan-Meier survival curves by credit score band
- `src/cox_ph.py` — Cox PH model for hazard ratio estimation
- `src/chiizer.py` — Discretize variables into risk categories
- `src/predict_survival.py` — Predict survival for new applicants
- `run_pipeline.py` — Execute full analysis pipeline