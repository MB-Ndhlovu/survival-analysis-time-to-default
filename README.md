# Time-to-Default Survival Analysis

Credit risk modeling with survival analysis — Kaplan-Meier and Cox Proportional Hazards.

## What is Survival Analysis?

Survival analysis models the **time until an event occurs**. In credit risk, the event is loan default. Unlike binary classification (default / no default), survival analysis answers:

- What is the probability a borrower survives past 12 months? 24 months?
- Which factors accelerate or delay default?
- What is the median time to default for different credit segments?

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Survival Function S(t)** | Probability that no default has occurred by time t |
| **Hazard Function h(t)** | Instantaneous rate of default at time t, given survival to t |
| **Censoring** | Observation ends without event (paid off, still active) |
| **Kaplan-Meier** | Non-parametric estimate of S(t) from censored data |
| **Cox PH** | Semi-parametric model linking covariates to hazard rate |

## Business Value

> "Survival analysis tells you WHEN default is likely, not just IF."

With survival curves, lenders can:
- Price loans based on expected lifetime at risk
- Set risk-based monitoring alerts at specific months
- Segment borrowers by time-to-default, not just probability
- Improve loss given default (LGD) estimates by incorporating timing

## Project Structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── kaplan_meier.py
│   ├── cox_ph.py
│   ├── chiizer.py
│   └── predict_survival.py
└── reports/
    └── survival_results.json
```

## Credit Score Bands

| Band | Score Range | Risk Level |
|------|-------------|------------|
| Deep Subprime | < 580 | Very High |
| Subprime | 580–669 | High |
| Near Prime | 670–739 | Medium |
| Prime | 740+ | Low |

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Outputs

- Kaplan-Meier survival curves per credit band
- Median survival times
- Cox PH hazard ratios
- 12-month and 24-month survival probabilities
- Predicted survival curve for new applicants