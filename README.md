# Credit Risk Modeling — Bank Default Prediction

An end-to-end credit risk modeling project on the **Home Credit Default Risk** dataset:
from raw multi-table data loading and business understanding, through feature engineering
and model validation, to a reproducible scorecard with documented metrics.

The project follows the way credit risk work is actually reviewed in a bank: every feature
has a documented business meaning, every model has out-of-sample validation, and every
reported number can be reproduced from the raw data.

## Status

**Steps 1–2 (data acquisition and database feature layer) — complete.**

Roadmap:

| Step | Content | Status |
|------|---------|--------|
| 1 | Data acquisition, table structure, data dictionary, raw loading checks | Done |
| 2 | PostgreSQL ingestion and SQL aggregation of the 1:N bureau table | Done |
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
  src/data_layer/load_raw.py       # Step 1: raw loading and integrity checks
  src/data_layer/ingest_to_db.py   # Step 2: chunked CSV -> PostgreSQL ingestion
  sql/01_create_tables.sql         # Step 2: raw table DDL
  sql/02_aggregate_bureau.sql      # Step 2: bureau -> one row per customer
  sql/03_build_model_table.sql     # Step 2: LEFT JOIN into the modelling table
  tests/test_load_raw.py           # loading-layer tests
  tests/test_aggregation.py        # SQL aggregation and LEFT JOIN semantics
  tests/test_ingest_to_db.py       # ingestion tests (SQLite stand-in)
  data/data_dictionary.md          # field-level business meaning (Step 1 output)
  docker-compose.yml               # local PostgreSQL
  .github/workflows/ci.yml         # CI: pytest + PostgreSQL SQL validation
```

## How to run

```bash
python -m pip install -r requirements.txt

# place the Kaggle CSVs in data/raw/ first
python -m src.data_layer.load_raw

# run the tests
pytest -q
```

## Step 2: PostgreSQL feature layer

```bash
# 1. start a local PostgreSQL
docker compose up -d

# 2. create the raw tables, then ingest the CSVs (chunked, idempotent)
export CREDITRISK_DB_URL="postgresql+psycopg2://creditrisk:creditrisk@localhost:5432/creditrisk"
psql "$CREDITRISK_DB_URL" -f sql/01_create_tables.sql
python -m src.data_layer.ingest_to_db

# 3. aggregate the 1:N bureau table and build the modelling table
psql "$CREDITRISK_DB_URL" -f sql/02_aggregate_bureau.sql
psql "$CREDITRISK_DB_URL" -f sql/03_build_model_table.sql
```

**Why a database layer instead of one big `pandas` join?** The 1:N tables have
millions of rows, so the aggregation is pushed into SQL where it is versioned,
reviewable and closer to a production pipeline; `pandas` is reserved for the
modelling layer.

**Design decisions**

- `LEFT JOIN` keeps every application, including customers with no bureau record.
- Count features default to `0` (`COALESCE`), while amount features stay `NULL`,
  because "zero credit" and "unknown credit" are different states; Step 4 decides
  the missing-value policy.
- `MAX(days_credit)` is the most recent bureau record because the day counts are
  negative.
- Ingestion only writes the columns defined in the DDL, lower-cases CSV headers,
  and clears the table first so the load is idempotent.

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
