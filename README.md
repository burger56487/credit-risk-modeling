# Credit Risk Modeling — Bank Default Prediction

An end-to-end credit risk modeling project on the **Home Credit Default Risk** dataset:
from raw multi-table data loading and business understanding, through feature engineering
and model validation, to a reproducible scorecard with documented metrics.

The project follows the way credit risk work is actually reviewed in a bank: every feature
has a documented business meaning, every model has out-of-sample validation, and every
reported number can be reproduced from the raw data.

## Status

**Step 1 (data acquisition and business understanding) — complete.**

Roadmap:

| Step | Content | Status |
|------|---------|--------|
| 1 | Data acquisition, table structure, data dictionary, raw loading checks | Done |
| 2 | Data cleaning: missing values, the `DAYS_EMPLOYED` sentinel, outliers | Planned |
| 3 | Feature engineering on the 1:N tables (bureau, previous applications, installments) | Planned |
| 4 | Feature selection and business-driven feature review | Planned |
| 5 | Model training with out-of-sample validation (logistic regression baseline, GBDT) | Planned |
| 6 | Model evaluation: AUC, KS, gain/lift, calibration, cutoff selection | Planned |
| 7 | Scorecard / interpretability (coefficients, SHAP) and monitoring metrics | Planned |

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle).

The raw data is **not** committed to this repository (size and licence). To run the code,
download the competition files and place them in `data/raw/`:

```
data/raw/
  application_train.csv
  bureau.csv
  bureau_balance.csv
  previous_application.csv
  POS_CASH_balance.csv
  installments_payments.csv
  credit_card_balance.csv
```

The main table has one row per loan application (about 300k rows) with the target
`TARGET` (1 = default / payment difficulty, 0 = repaid).

## Repository structure

```
credit-risk-modeling/
  src/data_layer/load_raw.py    # Step 1: raw loading and integrity checks
  tests/test_load_raw.py        # unit tests for the loading layer
  data/data_dictionary.md       # field-level business meaning (Step 1 output)
  .github/workflows/ci.yml      # CI: run tests on every push
```

## How to run

```bash
python -m pip install -r requirements.txt

# place the Kaggle CSVs in data/raw/ first
python -m src.data_layer.load_raw

# run the tests
pytest -q
```

## Business notes already captured in Step 1

- `EXT_SOURCE_1/2/3` are external credit scores (0–1) and historically the strongest
  predictors in this dataset.
- `DAYS_BIRTH` and `DAYS_EMPLOYED` are negative day counts relative to the application
  date; they must be converted to years before use.
- `DAYS_EMPLOYED = 365243` is a sentinel for "not employed / missing" (about 1000 years)
  and must be cleaned; leaving it in distorts the model.
- `CODE_GENDER` carries fairness and compliance risk and needs an explicit treatment
  decision rather than being used by default.

## Reproducibility

Every script is deterministic where possible, the raw data is never modified in place, and
all cleaning and feature rules are documented and covered by unit tests.

## Disclaimer

This is a research and portfolio project on a public dataset. It is not a production credit
decision system and must not be used for real lending decisions.
