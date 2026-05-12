# Time-to-Default Survival Analysis

Credit risk modeling goes beyond binary default prediction. This project applies **survival analysis** to loan data to model *when* a borrower is likely to default — not just *if*. The key output is a survival curve: given a new applicant, we estimate the probability they remain non-defaulted over time.

## Core Concepts

### Survival Function S(t)
The probability that a borrower has *not* defaulted by time `t`:
$$S(t) = P(T > t)$$

### Kaplan-Meier Estimator
Non-parametric estimator of $S(t)$ that handles censored observations:
$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$
where $d_i$ = defaults at time $t_i$, $n_i$ = borrowers at risk just before $t_i$.

### Cox Proportional Hazards Model
Semi-parametric model for the hazard function:
$$h(t) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + ...)$$

**Hazard Ratio** = $\exp(\beta)$: HR > 1 means higher default risk relative to baseline.

### Censoring
~35% of observations are right-censored at 24 months (loan paid off, still active, or lost to follow-up). Survival analysis correctly uses this partial information.

## Files

| File | Description |
|------|-------------|
| `src/data_loader.py` | Generates 5,000 synthetic loan records with survival times |
| `src/kaplan_meier.py` | KM curves by credit score band, median survival times |
| `src/cox_ph.py` | Cox PH model fitting, hazard ratios, coefficient interpretation |
| `src/chiizer.py` | Risk chiizer — bin continuous variables into risk categories |
| `src/predict_survival.py` | Predict survival curve for a new loan applicant |
| `run_pipeline.py` | Execute full pipeline, print results, save outputs |

## Business Insight

A logistic regression tells you: *"This applicant has a 20% probability of default."*

Survival analysis tells you: *"This applicant has a 94% chance of surviving 12 months, but only 71% at 24 months — watch them closely between months 12–18 when default risk peaks."*

**This is the difference between IF and WHEN.**

## Credit Score Bands

| Band | Score Range | Risk Profile |
|------|-------------|-------------|
| Deep Subprime | < 580 | High risk |
| Subprime | 580–669 | Elevated risk |
| Near Prime | 670–739 | Moderate risk |
| Prime | 740+ | Low risk |

## Key Outputs

- Kaplan-Meier survival curves stratified by credit score band
- Median survival time (time to 50% default rate) per band
- Cox PH coefficients — which factors drive default risk most
- 12-month and 24-month survival probabilities by segment
- Predicted survival curve for a new applicant

## Dependencies

```
lifelines>=0.27.0
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
```

Install: `pip install -r requirements.txt`

## Running

```bash
python run_pipeline.py
```