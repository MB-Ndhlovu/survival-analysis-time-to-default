# Project 6: Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to loan default data, answering the critical question in credit risk: **not just whether a borrower will default, but WHEN**.

## Survival Analysis Concepts

### What is Survival Analysis?

Survival analysis is a branch of statistics that studies the time until an event occurs — often called "time-to-event" analysis. Unlike traditional regression, it handles:

- **Censoring**: Some observations haven't experienced the event by the end of the study period
- **Truncated data**: Not all subjects enter the study at the same time
- **Skewed distributions**: Event times are typically right-skewed

### Key Functions

- **Survival Function S(t)**: Probability that the event has not occurred by time t
- **Hazard Function h(t)**: Instantaneous rate of event occurrence at time t
- **Cumulative Hazard H(t)**: Total hazard accumulated up to time t

### Key Techniques

1. **Kaplan-Meier Estimator**: Non-parametric estimate of the survival function
2. **Cox Proportional Hazards Model**: Semi-parametric regression model for survival data
3. **Risk Chiizer**: Binning continuous variables into risk categories for segment analysis

## Business Application in Credit Risk

### Why Survival Analysis for Loan Default?

Traditional default models output a binary probability (will default / won't default). Survival analysis enhances this by:

| Traditional Model | Survival Analysis |
|-------------------|------------------|
| Will borrower default? | When is default most likely? |
| Static probability | Dynamic risk over time |
| Nocensoring handling | Handles censored loans naturally |
| Point-in-time view | Time-varying risk profile |

### Use Cases

1. **Risk-Based Pricing**: Adjust interest rates based on expected time-to-default
2. **Provisioning**: Calculate expected loss reserves at different time horizons
3. **Early Warning Systems**: Identify borrowers approaching high-risk periods
4. **Segmentation**: Stratify portfolio by survival probability curves

### Key Metrics

- **12-month / 24-month Survival Probability**: Probability the loan survives past specific milestones
- **Median Time-to-Default**: Time at which 50% of loans in a segment have defaulted
- **Hazard Ratios**: Relative risk increase per unit change in predictor variables

## Project Structure

```
survival-analysis-time-to-default/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Generate synthetic loan data
│   ├── kaplan_meier.py      # Kaplan-Meier survival curves
│   ├── cox_ph.py           # Cox Proportional Hazards model
│   ├── chiizer.py          # Risk chiizer - binning & segmentation
│   └── predict_survival.py  # Predict survival for new applicants
├── run_pipeline.py         # Execute full analysis pipeline
└── reports/
    └── survival_results.json
```

## Results Interpretation

### Credit Score Bands

Loans are segmented into FICO-style bands:
- **< 580**: Subprime — Highest default hazard
- **580-669**: Near-prime — Elevated risk
- **670-739**: Prime — Moderate risk
- **740+**: Super-prime — Lowest risk

### Cox PH Coefficients

Positive coefficients indicate factors that **increase** default hazard (reduce survival time). Key predictors:

- Higher credit scores → Lower hazard (protective)
- Higher income → Lower hazard (protective)
- Higher debt-to-income → Higher hazard (risky)
- Higher interest rates → Higher hazard (risky)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py
```