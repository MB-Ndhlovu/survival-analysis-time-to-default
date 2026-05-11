# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modelling — specifically, time-to-default prediction for loans. Instead of a binary "will default / won't default" model, survival analysis answers the more valuable question: **when** is a borrower likely to default, and how does that risk evolve over time?

## Survival Analysis Concepts

### Key Terminology
- **Survival Function S(t)**: The probability that a borrower has NOT defaulted by time `t`. Starts at 1 (all borrowers alive at t=0) and decreases as defaults occur.
- **Hazard Function h(t)**: The instantaneous rate of default at time `t`, given survival up to that point.
- **Censoring**: When a borrower fully repays their loan or the observation window closes, we know they survived until that point — but not beyond. These observations are "censored".
- **Median Survival Time**: The time `t` at which S(t) = 0.5 — half of borrowers have defaulted.

### Methods Implemented
1. **Kaplan-Meier Estimator**: Non-parametric estimation of the survival function. Produces step functions showing survival probability over time for different cohorts.
2. **Cox Proportional Hazards Model**: Semi-parametric regression model that estimates the effect of covariates (credit score, income, LTV, etc.) on the hazard rate.
3. **Risk Chiizer**: Discretises continuous risk factors into bins and compares survival curves across bins to identify risk drivers.

## Business Application in Credit Risk

Traditional PD (Probability of Default) models give a single number. Survival analysis extends this by providing:
- **12-month and 24-month survival probabilities** by segment
- **Time-varying risk profiles** — a borrower's risk level changes as their credit history evolves
- **Hazard ratios** from Cox PH showing which factors increase default risk most (e.g., "borrowers with LTV > 90% have 2.3x the hazard rate of those with LTV < 80%")
- **Segmented insight** — Kaplan-Meier curves broken down by credit score band give instant visual comparison of risk tiers

This supports:
- Risk-based pricing
- Early warning systems (identifying borrowers whose survival curve is dropping fast)
- Reserve calculations using survival probabilities at different horizons
- Portfolio monitoring by segment

## Project Structure
```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Synthetic loan data generation
│   ├── kaplan_meier.py     # KM curves by credit score band
│   ├── cox_ph.py           # Cox PH regression + hazard ratios
│   ├── chiizer.py          # Risk chiizer — bin analysis
│   └── predict_survival.py # Predict survival for new applicant
├── reports/
│   └── survival_results.json
└── km_curves.png
```

## Pipeline Output
Run `python run_pipeline.py` to execute the full analysis:
1. Generate 5,000 synthetic loan records with ~35% censored at 24 months
2. Fit Kaplan-Meier curves for 4 credit score bands
3. Fit Cox PH model and extract hazard ratios
4. Build risk chiizer — survival curves by DTI and LTV bins
5. Predict survival curve for a sample new applicant
6. Save results to `reports/survival_results.json` and `km_curves.png`