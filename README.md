# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit risk modeling, moving beyond binary default prediction to understand **when** default is likely to occur.

## Survival Analysis Concepts

### Key Terms

- **Survival Function S(t)**: The probability that a loan does not default beyond time `t`
- **Hazard Function h(t)**: The instantaneous rate of default at time `t`, given survival up to `t`
- **Censoring**: Loans that haven't defaulted by the observation end (lost to follow-up)
- **Median Survival Time**: Time at which 50% of loans have defaulted

### Why Survival Analysis for Credit Risk?

Traditional default models answer: *Will this loan default?* (binary classification)

Survival analysis answers: *When will this loan default, and what factors accelerate or delay default?*

This enables:
- More precise pricing by time-period
- Better reserve calculations
- Targeted early intervention strategies

## Files

- `src/data_loader.py` - Synthetic loan data generator with censorship
- `src/kaplan_meier.py` - Non-parametric survival curves by credit band
- `src/cox_ph.py` - Semi-parametric hazard model for risk factors
- `src/chiizer.py` - Discretize variables into risk categories
- `src/predict_survival.py` - Predict survival for new applicants
- `run_pipeline.py` - Execute full analysis pipeline

## Installation

```bash
pip install -r requirements.txt
python run_pipeline.py
```