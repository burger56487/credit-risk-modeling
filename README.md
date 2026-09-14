# Credit Risk Modeling — Bank Default Prediction

An end-to-end credit risk modeling project on the **Home Credit Default Risk** dataset:
from raw multi-table data loading and business understanding, through feature engineering
and model validation, to a reproducible scorecard with documented metrics.

The project follows the way credit risk work is actually reviewed in a bank: every feature
has a documented business meaning, every model has out-of-sample validation, and every
reported number can be reproduced from the raw data.

## Status

**Steps 1–12 complete** (data layer, database feature layer, leakage-safe splitting,
cleaning decisions, business features, train-only binning and WOE/IV encoding, the
end-to-end logistic-regression baseline, the gradient-boosting comparison, the
scorecard scaling with per-variable decomposition, the paired resampling comparison
of the two pipelines, the calibration and distribution-stability diagnostics, and the
internal explanation and reason-code prototype).

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
| 9 | Scorecard scaling: probability to points, and per-variable score decomposition | Done |
| 10 | Discrimination validation with paired resampling confidence intervals | Done |
| 11 | Probability calibration diagnostics and input-distribution stability | Done |
| 12 | Model explanation and reason-code prototype | Done |
| 13 | Approval threshold and business-policy offline simulation | Planned |

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
  src/data_layer/model_table.py    # modelling-table loading and integrity guards
  src/features/engineering.py      # Step 5: business features + train-only binning
  src/features/woe.py              # Step 6: WOE/IV encoder + audit reports
  src/models/logistic.py           # Step 7: end-to-end logistic baseline + metrics
  src/models/boosting.py           # Step 8: gradient-boosting comparison pipeline
  src/models/scorecard.py          # Step 9: score scale and points decomposition
  src/models/configs.py            # frozen model configurations shared by runners
  src/evaluation/discrimination.py # Step 10: ranking metrics + paired bootstrap
  src/evaluation/calibration_stability.py  # Step 11: calibration + drift checks
  src/explain/local.py             # Step 12: local attribution + reason codes
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
  tests/test_scorecard.py          # scale identities, boundaries, decomposition
  tests/test_scorecard_runner.py   # Step 9 runner end to end
  tests/test_discrimination.py     # weighted metrics, pairing, interval method
  tests/test_discrimination_runner.py  # Step 10 runner end to end
  tests/test_calibration_stability.py  # calibration maths, PSI, edge cases
  tests/test_diagnostics_runner.py     # Step 11 runner end to end
  tests/test_local_explanation.py      # identity, grouping and code boundaries
  tests/test_explanation_runner.py     # Step 12 runner end to end
  tests/conftest.py                # synthetic modelling table shared by runners
  scripts/run_step_08_comparison.py  # one shared split: logistic vs boosting
  scripts/run_step_09_scorecard.py   # score scale, decomposition and audit
  scripts/run_step_10_discrimination.py  # paired bootstrap on the same split
  scripts/run_step_11_diagnostics.py     # calibration + stability reports
  scripts/run_step_12_explanations.py    # aggregate explanation audit only
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

## Step 9: score scaling and per-variable decomposition

```bash
python scripts/run_step_09_scorecard.py \
  --model-table data/processed/model_table.csv \
  --out-dir reports/step_09
```

The runner refits the Step 7 logistic pipeline on the same train split as Step 8,
converts the validation scores, checks the two identities below before writing
anything, and then saves the scale, the per-bin points table and an audit file.
The sealed test split is not opened and the validation labels are not used to
adjust the scale.

```
logit(p) = intercept + Σ_j (coef_j · w_j)
S        = (offset − factor · intercept) + Σ_j (−factor · coef_j · w_j)
           \_______ base points ______/   \___ variable points ___/
```

```python
from src.models.scorecard import LogisticScorecard, ScoreScale

scale = ScoreScale(base_score=600.0, base_bad_good_odds=1.0 / 50.0,
                   points_to_double_odds=20.0)
scorecard = LogisticScorecard(model=logistic_model, scale=scale)

scores = scorecard.score(valid_raw)          # log_odds, probability, raw/display
parts = scorecard.contributions(valid_raw)   # unrounded variable points
scorecard.training_bin_table()               # WOE, coefficient and points per bin
```

**Conventions that are easy to state wrongly**

- What doubles is the **bad-to-good odds**, not the probability, the score or a
  loss. At `1:50` odds the probability is `1/51 ≈ 1.96%`, and the next anchor at
  `1:25` is `3.85%`, not `3.92%`.
- The anchor score (600) is a scale definition only. It is not a cut-off, and it
  is not a claim that the dataset's real odds are 1:50.
- The **base points are not the anchor score**: they also absorb the model
  intercept. In the demo run below the anchor is 600 and the base points are 505.
- The scale is a monotone transform: it adds no information, does not make a
  poorly calibrated probability accurate, and does not remove overfitting.
- Higher score means lower model-estimated risk, so any ranking metric computed
  from scores must be read in the opposite direction from the probability one.
- Splitting a total into parts is not causation. A negative variable
  contribution means the variable lowers the total in this decomposition; it is
  not a valid standalone rejection reason.

**Numerical discipline**

- Scores come from the model's linear output, not from a rounded probability.
  That stays finite even where the probability has already saturated.
- A probability of exactly 0 or 1 raises instead of being clipped silently.
- Variable points are never rounded before summing; rounding happens once, at
  the end, and only for the display column (`np.rint`, halves to even).
- Unknown bins keep the neutral fallback: WOE zero, variable points zero. This is
  not "no risk" — the other variables and the base points still apply, and the
  unknown-bin rate still needs monitoring. With the neutral policy the raw score
  for an all-unknown row equals the base points.
- Two tolerances gate the written reports: `1e-9` for the component
  reconstruction and `1e-12` for the probability round trip. These are
  engineering acceptance settings for the current scale, not permanent
  mathematical guarantees, and extreme scale parameters need their own checks.
- The scorecard holds a deep copy of the fitted model, so later edits to the
  source object cannot silently move the scorecard. That copy is object
  isolation, not release governance: code, data, dependency and artefact
  versions still have to be managed together.

**On a synthetic demo table** (6000 rows, bad rate 35.5%, not real Home Credit
data), the runner produced base points 505.40, a maximum component
reconstruction error of 1.1e-13 and a maximum probability round-trip error of
3.9e-16, with validation scores spanning roughly 336 to 687. Those numbers
exercise the arithmetic; they say nothing about real credit performance.

## Step 10: paired resampling of the two pipelines

```bash
python scripts/run_step_10_discrimination.py \
  --model-table data/processed/model_table.csv \
  --out-dir reports/step_10 --n-bootstrap 2000
```

The question this step answers is not "which number is bigger" but "how stable
is the gap". Both models are refitted with the frozen configuration from
[configs.py](src/models/configs.py) on the same train split, scored on the same
validation rows, then that validation set is resampled **in pairs**:

```
fixed validation rows + fixed predictions
            │
            ▼
  resample label 0 and label 1 separately, with replacement
            │
      one shared draw
      ┌─────┴─────┐
      ▼           ▼
  logistic     boosting
      └─────┬─────┘
            ▼
  record the paired difference, repeat, take percentiles
```

```python
from src.evaluation.discrimination import paired_stratified_bootstrap

comparison = paired_stratified_bootstrap(
    y=split.y_valid,
    baseline_probability=logistic_valid_probability,
    challenger_probability=boosting_valid_probability,
    n_bootstrap=2000, confidence_level=0.95, random_state=42,
)
comparison.summary    # point value, both marginal intervals, difference interval
comparison.draws      # every draw, kept for audit
comparison.metadata   # protocol and quality notes
```

**Why the resampling must be paired.** The two models score the same applicants,
so their metrics are correlated. Each draw therefore uses one occurrence-count
vector for both models, and the difference interval is read directly from the
distribution of `challenger − baseline`. Subtracting the endpoints of the two
separate intervals is not the paired difference interval and is much wider; on
the demo run below the naive subtraction spans `−0.0298 … 0.0286` (width 0.058)
against a paired interval of `−0.0079 … 0.0072` (width 0.015).

**Metric conventions**

- Inputs are raw probabilities: larger means the model thinks label 1 is more
  likely. Credit scores run the other way, so scores are never fed in here.
- Average precision is `Σ (recall_k − recall_{k−1}) · precision_k`, the same
  convention as Step 7, not a trapezoidal area.
- AUC uses trapezoidal integration over the ROC points, with tied predictions
  accumulated as a group so the result cannot depend on row order.
- Metric values are cross-checked against scikit-learn's weighted
  implementations, and occurrence weights are cross-checked against physically
  duplicating the sampled rows.

**Demo run on the synthetic table** (1200 validation rows, 774 good / 426 bad,
2000 draws, 3.4 seconds; synthetic data, not Home Credit):

| Metric | Baseline | Baseline interval | Challenger | Challenger interval | Difference | Difference interval |
|---|---|---|---|---|---|---|
| AUC | 0.9270 | 0.9111 – 0.9407 | 0.9265 | 0.9109 – 0.9397 | −0.0005 | −0.0079 – 0.0072 |
| KS | 0.6992 | 0.6614 – 0.7450 | 0.7157 | 0.6791 – 0.7553 | +0.0165 | −0.0151 – 0.0474 |
| Average precision | 0.8769 | 0.8482 – 0.9033 | 0.8772 | 0.8490 – 0.9038 | +0.0003 | −0.0143 – 0.0156 |

**How to read a difference interval**

- Interval entirely above zero: under the frozen models, this validation sample
  and the stratified-resampling assumption, the challenger ranks higher and the
  interval excludes zero. It still is not a claim about future business
  performance.
- Interval containing zero (as in the demo above): this sample does not support a
  stable positive difference. It is **not** proof that the two models are
  identical — "not enough evidence of a difference" and "evidence of no
  difference" are different statements.
- Stable but tiny difference: also ask whether the improvement has business
  meaning and what the extra complexity and calibration cost is.
- No significance probabilities are reported here, no metric is picked after the
  fact to headline it, and the three intervals are separate marginal intervals
  without a joint coverage guarantee.

The bootstrap optimises away repeated sorting: each model is sorted once and a
draw is just a weight vector, which is equivalent to duplicating rows but avoids
re-ranking 2000 times.

## Step 11: calibration and distribution stability

```bash
python scripts/run_step_11_diagnostics.py \
  --model-table data/processed/model_table.csv \
  --out-dir reports/step_11
```

Three questions that must not be merged into one number:

| Question | Method | Answered by |
|---|---|---|
| Does the ranking work? | AUC / KS / average precision | Step 10 |
| Are the probabilities accurate? | calibration bins, Brier, log loss | this step |
| Did the population move? | drift index on a fixed reference binning, missing and out-of-range checks | this step |

Good ranking does not imply good calibration, and a stable distribution does not
imply a working model.

**Calibration diagnostics**

- Bin edges are fixed equal-width probability intervals chosen in advance, never
  from the validation labels. The rule is right-closed, with an inner-edge value
  joining the lower bin, 0 in the first bin and 1 in the last.
- An empty bin keeps a missing observed rate; it is never recorded as zero risk.
  A bin with no observed events still gets a Wilson interval whose upper bound is
  above zero — ten samples with no event do not prove a zero rate.
- Bins below `min_bin_samples` are flagged so weak evidence is not read as a
  rule, and the per-bin table is always reported because a single overall number
  can hide over- and under-estimation that cancel out.
- The bin-level intervals describe the observed label rate inside that bin. They
  exclude model re-training and are not joint intervals across bins.
- Probabilities of exactly 0 or 1 are allowed here (unlike the Step 9 log-odds
  conversion, which requires a value strictly inside the range). Log loss uses
  the library's floating-point protection, so a confidently wrong endpoint
  gives a large finite penalty rather than a mathematical infinity.
- No calibrator is fitted. The validation set is used to diagnose probabilities,
  not to fit one and then re-report a "post-calibration" effect on the same data.

**Distribution stability**

- Reference bins come from the training split and are re-used unchanged on
  current data, because a re-fitted binning would compare two different rulers.
  Missing values always have their own bin, so they can never be dropped from
  the share comparison.
- Smoothing is applied to the shares with one shared `epsilon`, so two samples
  with identical proportions give an index of exactly zero even when their sizes
  differ. Adding a constant to counts instead would not have that property.
- The index still depends on the binning and the smoothing strength, so values
  are only comparable within one configuration.
- A constant reference column has a single numeric bin, so values far outside its
  range can still land in that bin and produce a zero index. The report therefore
  also carries the current missing rate, the share of bins never seen in the
  reference, and the shares below and above the reference range.
- Equal per-variable distributions do not prove that the correlation structure
  between variables is unchanged.

**What this step is not.** It is not a value-at-risk coverage test, and the index
is not a model-validity test: no empirical cut-off such as "above 0.25 means
failed" is encoded anywhere. It is also not a time-stability validation — with no
reliable application date it compares a random holdout against the training
distribution, and the training side is in-sample, so part of any gap may be
overfitting rather than a change in the applicant population.

**Demo run on the synthetic table** (3600 training / 1200 validation rows,
10 calibration bins; synthetic data, not Home Credit):

| Model | Mean prediction | Observed rate | Overall bias | Brier | Log loss | Bin-weighted absolute error |
|---|---|---|---|---|---|---|
| Logistic | 0.3734 | 0.3550 | +0.0184 | 0.1026 | 0.3298 | 0.0231 |
| Boosting | 0.3669 | 0.3550 | +0.0119 | 0.1035 | 0.3380 | 0.0362 |

Both models slightly over-estimate on average, and the per-bin table shows why
the overall number is not enough: the largest single-bin gap for the logistic
model is +0.1645 in a bin holding 39 rows. Prediction drift is small
(`0.0075` logistic, `0.0158` boosting) and feature drift tops out at `0.0091`,
which is expected on a random holdout and is **not** evidence of future-month
stability.

This step also fixed a numerical edge in Step 10: `ranking_metrics` weights were
accumulated directly, so several very large but individually finite weights could
overflow when summed. Weights are now normalised by their maximum first, which
leaves the metrics unchanged but keeps the accumulation finite; a weight that
would underflow to zero during normalisation is rejected instead of being
silently dropped.

## Step 12: internal explanations and reason codes

```bash
python scripts/run_step_12_explanations.py \
  --model-table data/processed/model_table.csv \
  --out-dir reports/step_12 --explain-rows 200 --top-k 3
```

The runner explains a fixed random sample of validation applications (drawn
without labels, not cherry-picked), and writes **aggregate** reports only. Per
application contributions and reason codes stay in memory for local review; they
are not published by this repository.

Both pipelines are explained in the log-odds space, where `eta = f0 + Σ φ_j` and
`p = 1 / (1 + exp(−eta))`, but they are not the same kind of object:

| | Logistic regression | Gradient boosting |
|---|---|---|
| Attribution | `coefficient × WOE` | framework's native tree-path contributions |
| Base term `f0` | model intercept | the framework's own base value |
| Reference meaning | all WOE at zero — not an average customer | tree-path reference — not a chosen customer profile |
| Relation to Step 9 | points = `−factor × φ_j` (same quantity, rescaled) | no scorecard points exist |

**What is verified, not assumed**

- Contributions plus the base term reconstruct the model's raw output, and
  `expit` of that reconstruction equals the model's own probability. In the demo
  run the maximum reconstruction error was `8.9e-16` (logistic) and `1.9e-14`
  (tree), well inside the `1e-8` acceptance bound.
- Business grouping sums contributions **with their signs** inside a group, so
  the total is preserved; grouping never turns a +0.7 and a −0.6 into 1.3.
- Every feature must be registered in `BASE_FEATURE_GROUPS`, otherwise the
  explainer raises instead of inventing a plausible-sounding reason from an
  unknown variable name. Reason codes come only from positive group
  contributions, above a small numerical-noise floor.

**Boundaries that are enforced in wording and in code**

- Step 9's per-variable points are a linear decomposition, **not** Shapley values;
  this step keeps calling them "logistic linear contributions".
- Step 8's training split gain answers "which variables reduced loss during
  training", not "why this application scored high"; it is not used as a local
  explanation here.
- The two references differ, so contribution magnitudes are not comparable
  across models. Both may still be inspected to see which information each model
  leans on.
- A contribution is not a percentage point: `+0.2` in log-odds is not "risk up by
  20%". Positive means it pushes the model output up relative to the reference,
  and it does not mean the application should be rejected — a positive group can
  be offset by negative ones.
- Missing values, the `365243` sentinel and first-seen bins are data-coverage
  flags. They never become "unemployed", "fraudulent" or "concealed" statements.
- Reason codes are worded as `…相对于当前解释基准推高模型风险输出，请结合数据
  来源进行内部复核`, never as "reject the loan because …". Producing a formal
  decision would need approval policy, legal review, customer-notification rules
  and a data-usage mandate that this project does not have.
- Age and its proxies stay an internal diagnostic. They are not cleared for use
  in real credit decisions or customer communications.
- The explainer works on a deep copy of the fitted model, so later edits to the
  source object cannot silently change the explanations. That is object
  isolation, not version management for code, data, model or reason-code rules.
- Tree-path attribution is not the only possible attribution scheme, and it
  fixes no external background sample. Additivity does not make an explanation
  unique and does not make it causal, especially when features are correlated.

**Demo run on the synthetic table** (200 validation applications, `top_k = 3`;
the mean signed contribution is not centred on zero because the reference is not
the average applicant — the base term carries the average level):

| Model | Top groups by mean absolute contribution |
|---|---|
| Logistic | 还款金额与收入相对规模 1.7595 (−0.4756), 外部评分信息 1.0483 (−0.1500), 本次还款金额 0.4766 (−0.1110) |
| Boosting | 还款金额与收入相对规模 2.2529 (−0.2141), 外部评分信息 1.2205 (−0.1070), 本次还款金额 0.3405 (+0.0072) |

The two models lean on the same information here, but the numbers are not
comparable magnitudes: the references differ, and the label is the dataset's own
definition rather than a verified regulatory default.

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
- The Step 9 scale is a presentation choice, not a calibrated risk statement:
  the probabilities behind it are uncalibrated and unweighted, no cut-off policy
  has been chosen, and no per-applicant scores are published by this repository.
- If a non-linear probability calibration is added later, the linear
  per-variable points no longer correspond exactly to the calibrated
  probability; the two must then be labelled and versioned separately.
- The Step 10 intervals cover one source of uncertainty only: resampling
  validation records under frozen models and a fixed label ratio. They exclude
  re-splitting, re-training, feature-selection variance, and any change in the
  future good/bad mix — which matters most for average precision.
- Applications are treated as approximately independent sampling units. Unique
  application IDs do not prove that the same person, household or period of
  shared stress is absent, so undetected dependence is outside the interval.
- The validation set has been inspected in earlier steps. These are development
  numbers under fixed models, not a pre-registered confirmatory experiment, and
  resampling cannot remove the resulting selection bias.
- Two thousand draws is a compute budget, not a guarantee that the tail
  percentiles are stable for every sample size and confidence level.
- The Step 11 drift numbers compare a random holdout with the training
  distribution. Without a reliable application timestamp they are not a
  time-stability check, and the training side is an in-sample fit, so part of any
  gap may be overfitting rather than population change.
- Calibration is diagnosed on a validation set that has already been inspected;
  no calibrator is fitted, and if one is added later it needs its own calibration
  split with a re-declared evaluation protocol.
- In real monitoring, input drift can be refreshed quickly but calibration and
  performance need matured labels. Applications whose outcome window has not
  closed must not be recorded as good, which this public dataset cannot support.
- Step 12 explanations describe how each model used its inputs; they are not
  causal statements and cannot show what would happen if a value changed. Group
  contributions can cancel internally, so the per-variable detail has to be kept
  alongside the group view.
- The internal reason codes are an unreviewed prototype: no approval policy,
  legal review, customer-notification wording or data-usage mandate stands behind
  them, and no approval threshold has been chosen.
- Age-related features remain in the internal explanation only; their use in
  real credit decisions would need a separate fairness and compliance review.

## Reproducibility

Every script is deterministic where possible, the raw data is never modified in place, and
all cleaning and feature rules are documented and covered by unit tests.

## Disclaimer

This is a research and portfolio project on a public dataset. It is not a production credit
decision system and must not be used for real lending decisions.
