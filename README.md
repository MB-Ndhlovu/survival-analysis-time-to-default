# Time-to-Default Survival Analysis

A credit risk survival analysis project using Kaplan-Meier estimation and Cox Proportional Hazards modeling to predict *when* loan defaults are likely to occur—not just whether they occur.

## Concept

Traditional credit scoring treats default as a binary outcome. Survival analysis goes further: it models the **time to default**, revealing:
- The probability of default at each month in a loan's life
- Which risk factors accelerate or delay default
- Expected median survival time by credit segment
- Tail-risk exposure at specific horizons (12-mo, 24-mo)

## Survival Analysis Fundamentals

### Key Terms
- **Survival Function S(t)**: Probability that default has not occurred by time `t`
- **Hazard Function h(t)**: Instantaneous default rate at time `t`
- **Censoring**: Loans that haven't defaulted by observation end (treated correctly, not ignored)
- **Median Survival Time**: Time at which S(t) = 0.50

### Methods
1. **Kaplan-Meier**: Non-parametric survival curve estimation; produces step-functions by credit segment for visual comparison
2. **Cox Proportional Hazards**: Semi-parametric regression that models hazard as a function of covariates; yields interpretable hazard ratios

## Business Application

Survival analysis enables:
- **Proactive credit monitoring**: Identify borrowers at risk of default at specific future windows
- **Risk-adjusted pricing**: Adjust loan terms based on expected time-to-default
- **Reserve modeling**: More precise loss given default timing for IFRS 9 / CECL compliance
- **Portfolio segmentation**: Rank segments by survival probability at key horizons

## Files

| File | Description |
|------|-------------|
| `src/data_loader.py` | Synthetic loan data generator (5,000 records) |
| `src/kaplan_meier.py` | KM curves by credit score band + median survival |
| `src/cox_ph.py` | Cox PH model fit + hazard ratio interpretation |
| `src/chiizer.py` | Risk chiizer: binned variable survival analysis |
| `src/predict_survival.py` | New applicant survival function prediction |
| `run_pipeline.py` | End-to-end execution + JSON report output |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```