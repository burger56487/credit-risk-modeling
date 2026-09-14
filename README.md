# Credit Risk Modeling — Bank Default Prediction

An end-to-end credit risk modeling project on the **Home Credit Default Risk** dataset:
from raw multi-table data loading and business understanding, through feature engineering
and model validation, to a reproducible scorecard with documented metrics.

The project follows the way credit risk work is actually reviewed in a bank: every feature
has a documented business meaning, every model has out-of-sample validation, and every
reported number can be reproduced from the raw data.

## Status

**Steps 1–8 complete** (data layer, database feature layer, leakage-safe splitting,
cleaning decisions, business features, train-only binning and WOE/IV encoding, the
end-to-end logistic-regression baseline, and the gradient-boosting comparison).

Roadmap:

| Step | Content | Status |
|------|---------|--------|
| 1 | Data acquisition, table structure, data dictionary, raw loading checks | Done |
| 2 | PostgreSQL ingestion and SQL aggregation of the 1:N bureau table | Done (demonstration layer, see limitations) |
| 3 | Train / validation / test splitting with leakage discipline | Done |
| 4 | EDA and cleaning plan; median imputation dropped in favour of a missing bin | Done (decisions implemented in Step 5) |
| 5 | Business features + train-only quantile binning | Done |
| 6 | WOE / IV encoding on the training bins (zero cells, unknown bins, smoothing) | Done |
| 7 | Logistic-regression scorecard baseline, evaluation and monitoring metrics | Done |
| 8 | Gradient-boosting comparison on the same split, scored fairly against the scorecard | Done |
| 9 | Scorecard scaling: probability to points, and per-variable score decomposition | Planned |

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
  src/data_layer/split_data.py     # Step 3: stratified and temporal splitting
  src/features/engineering.py      # Step 5: business features + train-only binning
  src/features/woe.py              # Step 6: WOE/IV encoder + audit reports
  src/models/logistic.py           # Step 7: end-to-end logistic baseline + metrics
  src/models/boosting.py           # Step 8: gradient-boosting comparison pipeline
  sql/01_create_tables.sql         # Step 2: raw table DDL
  sql/02_aggregate_bureau.sql      # Step 2: bureau -> one row per customer
  sql/03_build_model_table.sql     # Step 2: LEFT JOIN into the modelling table
  tests/test_load_raw.py           # loading-layer tests
  tests/test_aggregation.py        # SQL aggregation and LEFT JOIN semantics
  tests/test_ingest_to_db.py       # ingestion tests (SQLite stand-in)
  tests/test_split_data.py         # split sizes, bad rates, no-leakage checks
  tests/test_engineering.py        # sentinel handling, ratios, binning discipline
  tests/test_woe.py                # WOE/IV maths, alignment, unknown-bin policy
  tests/test_logistic.py           # baseline chain, fitted-state discipline, metrics
  tests/test_boosting.py           # tree pipeline, duplicate/missing handling
  tests/test_comparison_runner.py  # Step 8 runner guards and written reports
  scripts/run_step_08_comparison.py  # one shared split: logistic vs boosting
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

## Step 3: splitting and leakage discipline

```python
from src.data_layer.split_data import stratified_split, temporal_split

split = stratified_split(df)          # Home Credit: no timestamp available
print(split.summary())                # sizes and bad rates per split

split = temporal_split(df, time_col="issue_d")   # timestamped data
print(split.meta["time_ranges"])      # auditable time boundaries
```

**Why out-of-time (OOT) validation matters.** A credit model is applied to
future applicants, so a random split can overstate performance: the training set
may contain information that is contemporaneous with (or later than) the test
period. The standard three-way split is `train` (fit parameters) | `valid`
(tuning and cut-off selection) | `test` (opened once, at final evaluation).

**Accurate wording (important).** The Home Credit main table has **no application
timestamp**, so the split used here is a *stratified random holdout* — it is
**not** a true out-of-time test and this project does not describe it as one.
`DataSplit.meta["mode"]` records `stratified_holdout` for that case. The module
also implements `temporal_split`, a genuine OOT split for datasets that do have a
timestamp (for example LendingClub's `issue_d`), recorded as `temporal_oot`.

**Leakage rules enforced in this project**

1. The test set is opened once. All tuning, feature selection and cut-off choice
   use train/validation only.
2. No feature may use information dated after the application moment (label
   leakage).
3. Binning and WOE must be fitted on the training set only and then applied to
   validation and test (binning in Step 5, WOE in Step 6).
4. The ID column is kept in `X` for traceability and must be dropped before model
   training (Step 7).

## Business notes already captured in Step 1

- `EXT_SOURCE_1/2/3` are external credit scores (0–1) and historically the strongest
  predictors in this dataset.
- `DAYS_BIRTH` and `DAYS_EMPLOYED` are negative day counts relative to the application
  date; they must be converted to years before use.
- `DAYS_EMPLOYED = 365243` is a special code (about 1000 years) that must be
  handled separately. It is **not** treated as proof that the applicant is
  unemployed, and no risk statement is attached to it in advance.
- `TARGET` records payment difficulty under the dataset's own definition; the
  public description does not specify a full overdue threshold and observation
  window, so it is not labelled as a confirmed "90+ days past due" definition.
- `CODE_GENDER` carries fairness and compliance risk and needs an explicit treatment
  decision rather than being used by default.

## Step 5: business features and train-only binning

```python
import pandas as pd
from src.features.engineering import (
    TrainQuantileBinner,
    build_business_features,
)

# Row-wise rules: no statistic is fitted, so they can be applied split by split.
train_features = build_business_features(split.X_train)
valid_features = build_business_features(split.X_valid)

# Edges are learned parameters: fit on train, then apply unchanged.
binner = TrainQuantileBinner(n_bins=5).fit(train_features)
train_bins = binner.transform(train_features)
valid_bins = binner.transform(valid_features)

pd.testing.assert_index_equal(train_bins.index, split.y_train.index)
```

**Design decisions**

- Fixed row-wise rules (special-code detection, negative amounts, ratios) need no
  fitting; quantile edges are learned parameters and are fitted on the training
  split only.
- Missing values are preserved and form their own bin. The earlier
  median-imputation plan is deliberately dropped, so "missing" stays visible to
  the scorecard.
- Low-cardinality columns keep distinct values separate, so a binary flag is not
  merged into one bin; constant and all-missing columns have defined behaviour.
- Values outside the training range fall into the end bins. If a column was
  all-missing during training but has values later, it is labelled
  `训练外非缺失箱` and needs an explicit fallback rule at the WOE step.
- Ratios use only strictly positive income as the denominator; missing, zero or
  negative income produces a missing ratio instead of a distorted number.

## Step 6: WOE encoding and IV screening

```python
from src.features.woe import (
    WOEEncoder,
    candidate_features_by_iv,
    save_woe_reports,
)

encoder = WOEEncoder(alpha=0.5, min_bin_samples=20, unknown_policy="neutral")
train_woe = encoder.fit_transform(train_bins, split.y_train)  # train labels only
valid_woe = encoder.transform(valid_bins)                     # apply, never refit

candidates = candidate_features_by_iv(encoder, iv_threshold=0.02)
save_woe_reports(encoder, valid_bins, "reports/step_06", candidates)
```

**Conventions**

- Direction: `WOE = ln(bad share / good share)`, where bad = target 1. Positive
  WOE means the bin is more common among bad samples. WOE is not a default
  probability.
- Smoothing is symmetric and added to both classes, so the smoothed bad and good
  shares each still sum to one:
  `bad_share = (B_i + alpha) / (B + alpha*K)`, and the same for good.
- IV is computed from the same smoothed distributions:
  `IV = sum((bad_share - good_share) * WOE)`. As a rough guide only:
  `<0.02` weak, `0.02–0.1` some information, `0.1–0.3` clearly informative,
  `>=0.3` worth a careful review (very high values can indicate leakage, a
  high-cardinality variable or sparse bins — high IV is not automatically good).
- Bins unseen in training are encoded as `0` under the neutral policy (no
  evidence either way, not "safe") and always listed in the unknown-bin report;
  the strict policy raises instead. Bins that *did* appear in training, including
  a missing bin, are encoded from their learned WOE.
- Small bins and single-class bins are flagged but never removed automatically.

**Discipline**

- The encoder only accepts binned string columns; raw missing values and numeric
  columns are rejected, so a later step cannot silently encode raw data.
- The label must be index- and order-aligned with the features; misalignment
  raises instead of producing a wrong mapping.
- WOE/IV use the label, so inside cross-validation both the binner and the
  encoder must be refitted on each training fold; the IV screen is a univariate
  candidate filter, not the final variable selection.

## Step 7: logistic-regression baseline

```python
import pandas as pd
from src.models.logistic import (
    RiskLogisticModel,
    evaluate_probabilities,
    save_baseline_reports,
)

# Start from the raw, un-imputed split. The chain below is fitted inside fit().
model = RiskLogisticModel(
    n_bins=5,          # same binning as Steps 5-6
    alpha=0.5,         # smoothing, avoids infinite WOE in single-class bins
    iv_threshold=0.02, # pre-declared IV screen
    C=1.0,             # inverse regularisation strength, fixed (not tuned)
    max_iter=2000,
)
model.fit(split.X_train, split.y_train)

train_probability = model.predict_bad_probability(split.X_train)
valid_probability = model.predict_bad_probability(split.X_valid)

# Reference point: no applicant features at all, just the training bad rate.
constant = pd.Series(model.training_bad_rate_, index=split.y_valid.index)

metrics = pd.DataFrame(
    {
        "训练集回代诊断": evaluate_probabilities(split.y_train, train_probability),
        "验证集逻辑回归": evaluate_probabilities(split.y_valid, valid_probability),
        "验证集常数基准": evaluate_probabilities(split.y_valid, constant),
    }
).T

save_baseline_reports(
    model, metrics, model.unknown_bin_report(split.X_valid), "reports/step_07"
)
print(model.selection_report_)   # per-feature keep/delete reason
print(model.coefficient_report_)
```

`fit` learns the whole chain from raw application features — business rules,
train-only quantile edges, train-only WOE, the IV screen, removal of constant
and exactly duplicated columns, then a regularised logistic regression.
`predict*` re-applies those fitted objects, so scoring cannot silently re-fit a
binner or change the feature order. Wrapping the chain does not grant data
permissions: passing validation data to `fit` would still leak.

**Configuration choices, and what they do not claim**

| Choice | Value | Reason |
|--------|-------|--------|
| Bins | 5 | Inherited from Step 5; not claimed to be optimal |
| Smoothing | 0.5 | Stops single-class bins producing infinite WOE |
| IV screen | 0.02 | Pre-declared candidate filter, not final selection |
| Regularisation | `C = 1.0`, fixed | No large search in this baseline |
| Class weight | none | Keeps the output interpretable as the sample's own bad rate |
| Column handling | drop constants and exact duplicates | Removes obvious redundancy only |
| Final test set | not used | Kept sealed |

Note the direction of `C`: it is the **inverse** penalty strength, so a smaller
`C` means a stronger penalty.

**What is deliberately not claimed**

- Training-set performance is an in-sample refit diagnostic, not generalisation;
  only the validation numbers speak to that, and the test set stays sealed.
- No significance statements. After IV screening and regularisation, ordinary
  regression p-values do not apply, so coefficients are reported as model
  associations with a sign and a magnitude only.
- Regularisation does not remove collinearity. Highly correlated features can
  still coexist, so a coefficient is a conditional association inside this
  model, not a causal effect.
- No class weighting, up-sampling or down-sampling, so the reference probability
  keeps its meaning; calibration is assessed separately.
- A negative coefficient is not automatically a bug, and it is not silently
  flipped to a positive number.
- `predict`'s 0.5 threshold exists only to satisfy the standard classifier
  interface; it is not an approval policy.
- Non-convergence is escalated to a `RuntimeError` instead of being ignored.
- Reports are aggregated in this repository; no per-applicant predictions are
  published.

The validation numbers themselves are produced by running the script on the real
split and are not asserted anywhere in the code or the tests.

## Step 8: gradient-boosting comparison

```bash
# needs the modelling table from the Step 2 SQL layer
python scripts/run_step_08_comparison.py \
  --model-table data/processed/model_table.csv \
  --out-dir reports/step_08
```

The runner cuts **one** stratified split, fits both pipelines on that same train
split, scores both on that same validation split, and leaves the test split
sealed. It writes the three-way validation comparison, the in-sample training
diagnostics, both feature reports and an `experiment_metadata.json` that records
the parameters, the environment versions and the SHA-256 of the input file.

The two pipelines are deliberately different:

| | Step 7 scorecard | Step 8 tree |
|---|---|---|
| Features | business features → bins → WOE → IV screen | business features → constants and exact duplicates dropped |
| Label use | WOE and IV are fitted on training labels | only the tree fit uses labels |
| Missing values | own bin, then a WOE value | kept as missing, handled natively |
| Class weight | none | none |
| Early stopping | n/a | none: the validation set is not passed to `fit` |

**Configuration choices**

| Parameter | Value | Role |
|-----------|-------|------|
| Rounds | 300 | fixed training budget |
| Learning rate | 0.05 | small step per round |
| `num_leaves` | 15 | limits single-tree complexity |
| `min_child_samples` | 100 | discourages very fine splits |
| `reg_lambda` | 1.0 | limits leaf output size |
| Threads | 1 | keeps test and reproduction stable |

`min_child_samples` is approximate in this framework; it is not a hard per-leaf
database constraint. A fixed seed and `deterministic=True` improve
reproducibility but do not guarantee bit-identical results across operating
systems, library versions and compilers.

**Two details worth naming**

- The retained feature count differs between the two models. That is the
  designed consequence of comparing two whole procedures, not a missing step:
  the tree branch does not inherit the IV screen, and a variable with weak
  univariate IV can still matter through interactions.
- The recorded rounds are the *configured* budget and the *actual* completed
  rounds. The framework can finish early without any early stopping, so the
  configured 300 must not be reported as "300 effective trees".

**Reading the comparison** (the numbers are descriptive, not a verdict)

1. Tree better on both ranking and probability loss: the non-linear pipeline
   added value on this validation split. It still does not follow that it will
   win in future, that the gap is statistically significant, or that it is
   deployable.
2. Tree better at ranking but worse on log loss: the ordering improved while the
   probabilities became over-confident; check overfitting and calibration before
   declaring a win.
3. Tree much better in-sample, barely better out-of-sample: the model mostly
   learned noise. Fewer leaves, larger leaf size, stronger regularisation or
   early stopping belong to a *later* experiment, not to a quiet edit after
   seeing the result.
4. The scorecard is already close: that is a useful finding, not a failure. It
   supports choosing the easier-to-review pipeline, subject to the business
   constraints.

Split gain and logistic coefficients are not comparable quantities. Split gain
is a training-time statistic, not a causal contribution and not the benefit
measured on validation data, so it is exported for inspection only.

## Current limitations (kept explicit)

- The Home Credit split in this repository is a stratified random holdout, **not**
  a true out-of-time test; only `temporal_split` provides OOT.
- `TARGET` is payment difficulty under the dataset's own definition, not a
  confirmed "90+ days past due" label.
- `DAYS_EMPLOYED = 365243` is treated as a special code, not as evidence of
  unemployment.
- The bureau aggregates describe the bureau records available in the data; they do
  not reconstruct the customer's full delinquency history.
- The PostgreSQL layer is a demonstration, not yet a validated end-to-end
  pipeline. Known TODOs: run truncation and load inside one transaction, handle
  very wide tables when copying, and refresh feature tables after new loads.
- This is a research project on a public dataset; passing tests does not make the
  model fit for real lending decisions.
- The Step 6 reports are audit artefacts, not deployment files: the bin edges,
  encoder, feature order and model must be versioned together.
- The Step 7 baseline is a first reference point, not a validated scorecard: the
  regularisation stays fixed, calibration, stability over time and threshold
  selection are not yet done, and the metrics table is only meaningful once the
  script has been run on the real split.
- The Step 7 evaluation metrics are computed on a stratified random holdout, so
  they describe this sample, not the model's behaviour on a future applicant
  population.
- The Step 8 comparison has no early stopping, no parameter search and no
  probability calibration, and no paired significance test: it is a first
  side-by-side reading of two pipelines, not evidence that one model is better.
- LightGBM can accept missing values natively, but that does not mean an unseen
  missing pattern is scored reliably; missingness distribution still needs
  monitoring.
- The Step 8 runner records parameters, environment versions and an input file
  digest, which is not yet a full reproduction receipt: the sample split list,
  the code version and a dependency lock file are still missing.

## Reproducibility

Every script is deterministic where possible, the raw data is never modified in place, and
all cleaning and feature rules are documented and covered by unit tests.

## Disclaimer

This is a research and portfolio project on a public dataset. It is not a production credit
decision system and must not be used for real lending decisions.
