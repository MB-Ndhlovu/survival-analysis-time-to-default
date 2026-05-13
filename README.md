# Time-to-Default Survival Analysis

## Overview

This project applies survival analysis techniques to credit default prediction. Unlike binary classification models that only answer "will this borrower default?", survival analysis answers "when will default occur?" — giving lenders more actionable information for risk management and pricing.

## Core Concepts

### Survival Function S(t)
The probability that a borrower has NOT defaulted by time `t`:
$$S(t) = P(T > t)$$

### Hazard Function h(t)
The instantaneous failure rate at time `t`, given survival up to `t`:
$$h(t) = \lim_{dt \to 0} \frac{P(t \leq T < t+dt \mid T \geq t)}{dt}$$

### Kaplan-Meier Estimator
Non-parametric estimator of the survival function that handles censored data naturally:
$$\hat{S}(t) = \prod_{t_i < t} \frac{n_i - d_i}{n_i}$$
where $n_i$ is the number at risk and $d_i$ is the number of defaults at time $t_i$.

### Cox Proportional Hazards Model
Semi-parametric model that estimates the effect of covariates on hazard:
$$h(t) = h_0(t) \cdot \exp(\beta_1 x_1 + \beta_2 x_2 + ...)$$

### Censoring
Borrowers who haven't defaulted by the observation end are "censored" — their true default time is unknown but at least 24 months. The Kaplan-Meier estimator properly handles this.

## Business Application

In credit risk, survival analysis enables:
- **Risk-based pricing**: Charge higher rates to borrowers likely to default early
- **Expected loss forecasting**: Calculate lifetime expected loss more accurately
- **Portfolio management**: Identify segments with deteriorating survival curves
- **Regulatory capital**: Better estimate of loss given default timing

## Files

| File | Description |
|------|-------------|
| `src/data_loader.py` | Generate synthetic loan data with 5000 observations |
| `src/kaplan_meier.py` | Fit and visualize Kaplan-Meier survival curves |
| `src/cox_ph.py` | Fit Cox PH model, compute hazard ratios |
| `src/chiizer.py` | Bin variables into risk categories, compare curves |
| `src/predict_survival.py` | Predict survival curve for new applicants |
| `run_pipeline.py` | Execute full analysis pipeline |

## Credit Score Bands

| Band | Score Range | Risk Profile |
|------|-------------|--------------|
| Deep Subprime | < 580 | High risk |
| Subprime | 580-669 | Elevated risk |
| Near Prime | 670-739 | Moderate risk |
| Prime | 740+ | Low risk |

## Key Metrics

- **Median survival time**: Time at which 50% of borrowers have defaulted
- **12-month survival probability**: P(no default in first year)
- **24-month survival probability**: P(no default in first 2 years)
- **Hazard ratio**: Multiplicative effect on default risk per unit change in predictor