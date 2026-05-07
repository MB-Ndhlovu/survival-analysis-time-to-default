# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand **when** default is likely to occur.

## Survival Analysis Concepts

### What is Survival Analysis?
Survival analysis is a branch of statistics that models time-to-event data. In credit risk, the "event" is loan default, and "survival" means the loan remaining in good standing.

### Key Concepts

- **Survival Function S(t)**: The probability that a loan survives beyond time t without defaulting
- **Hazard Function h(t)**: The instantaneous rate of default at time t, given survival until t
- **Censoring**: Loans that haven't defaulted by the observation end date are "censored" — we know they survived at least that long
- **Median Survival Time**: The time at which 50% of loans have defaulted (or S(t) = 0.5)

### Why Survival Analysis for Credit Risk?

Traditional default models answer: *Will this borrower default?* (yes/no)

Survival analysis answers: *When will this borrower default?* and *What factors accelerate or reduce default risk over time?*

This enables:
- More precise pricing based on expected lifetime loss
- Better LTV calculations incorporating time
- Targeted intervention timing for at-risk portfolios
- Segment-specific survival curves for risk-based pricing

## Files

| File | Description |
|------|-------------|
| `src/data_loader.py` | Generates 5000 synthetic loan records with survival times |
| `src/kaplan_meier.py` | Non-parametric survival curves by credit score band |
| `src/cox_ph.py` | Semi-parametric hazard model identifying risk factors |
| `src/chiizer.py` | Bins continuous variables into risk categories |
| `src/predict_survival.py` | Predicts survival curve for new loan applicants |
| `run_pipeline.py` | Executes full analysis pipeline |

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Business Application

Credit lenders use survival analysis to:
1. Estimate expected loss timing for loan portfolios
2. Set risk-adjusted pricing by segment
3. Trigger early intervention before default probability peaks
4. Compare portfolio performance across economic cycles