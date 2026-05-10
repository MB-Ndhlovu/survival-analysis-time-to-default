# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand **when** default is likely to occur.

## Survival Analysis Concepts

### Why Survival Analysis?

Traditional credit scoring answers: **Will this borrower default?** (Yes/No)

Survival analysis answers: **When will this borrower default?** and **What factors accelerate or delay default?**

### Key Terminology

- **Survival Function S(t)**: Probability that a borrower has NOT defaulted by time `t`
- **Hazard Function h(t)**: Instantaneous rate of default at time `t`, given survival up to `t`
- **Censoring**: When a loan is still active (not defaulted or paid off) at observation end
- **Median Survival Time**: Time at which 50% of loans have defaulted

### Methods Used

1. **Kaplan-Meier Estimation**: Non-parametric survival curves by cohort
2. **Cox Proportional Hazards**: Semi-parametric model identifying risk factors
3. **Risk Chiizer**: Binning continuous variables to reveal survival differentials

## Business Application

Credit risk managers use survival analysis to:
- Set risk-based pricing by time-to-default
- Optimize collection strategies (intervene before high-hazard periods)
- Stress-test portfolio behavior across time horizons
- Segment borrowers for差异化产品设计

## Files

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Synthetic loan data generation
│   ├── kaplan_meier.py      # KM curves by credit band
│   ├── cox_ph.py            # Cox PH hazard ratio analysis
│   ├── chiizer.py           # Risk chiizer (binning + survival)
│   └── predict_survival.py  # Score new applicant
├── run_pipeline.py          # Execute full analysis
└── reports/
    └── survival_results.json
```

## Key Outputs

| Metric | Description |
|--------|-------------|
| Kaplan-Meier Curves | Survival probability over time by credit band |
| Median Time-to-Default | Days until 50% default rate per segment |
| Hazard Ratios | Cox PH coefficients showing which factors increase default risk |
| 12/24-Month Survival | Probability borrower survives past milestone |
| Applicant Score | Predicted survival curve for new loan application |

## Interpretation Guide

- **Hazard Ratio > 1**: Factor increases default risk
- **Hazard Ratio < 1**: Factor decreases default risk (protective)
- **Survival Curve higher**: Better prognosis (lower default probability)
- **Median Survival lower**: Faster time-to-default (worse credit quality)