# Time-to-Default Survival Analysis

## Overview
This project applies survival analysis techniques to credit risk modeling, specifically analyzing **time-to-default** for loans. Unlike binary default classifiers, survival analysis answers the question: *when* is a borrower most likely to default, not just *if* they will default.

## Survival Analysis Concepts

### Key Ideas
- **Survival Function S(t)**: The probability that a borrower has NOT defaulted by time `t`
- **Hazard Function h(t)**: The instantaneous failure rate at time `t`, given survival until `t`
- **Censoring**: Some borrowers haven't defaulted yet (right-censored) — we know they survived at least until `time_end`
- **Median Survival Time**: Time at which 50% of borrowers have defaulted

### Why Survival Analysis for Credit Risk?
1. **Handles censored data** — 35% of loans in this dataset are censored at 24 months
2. **Time-varying risk** — default risk changes over the loan lifecycle
3. **Segment-level insights** — survival curves reveal which credit bands deteriorate fastest
4. **More than binary** — a 700 credit score borrower may be 3x more likely to default by month 18 vs month 6

## Business Application
- **Origination**: Set risk-based pricing by expected time-to-default
- **Provisioning**: Reserve calculation based on probability of default at each future period
- **Early warning**: Identify borrowers approaching high-hazard periods
- **Portfolio monitoring**: Track survival curves across economic cycles

## Methods Used
| Method | Purpose |
|--------|---------|
| Kaplan-Meier | Non-parametric survival curves by segment |
| Cox Proportional Hazards | Identify which factors increase/decrease default hazard |
| Risk Chiizer | Bin continuous variables into interpretable risk categories |

## Project Structure
```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── reports/
│   └── survival_results.json
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── kaplan_meier.py
    ├── cox_ph.py
    ├── chiizer.py
    └── predict_survival.py
```