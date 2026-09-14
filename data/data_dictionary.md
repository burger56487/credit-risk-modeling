# Data dictionary — Home Credit Default Risk

## Table relationships

| Table | Content | Grain / relation |
|-------|---------|------------------|
| `application_train.csv` | Application information + target | One row per customer (`SK_ID_CURR`, the main key) |
| `application_test.csv` | Label-free test split (Kaggle only; ignored here) | One row per customer |
| `bureau.csv` | Credit history at other institutions | 1:N via `SK_ID_CURR`; own key `SK_ID_BUREAU` |
| `bureau_balance.csv` | Monthly balances of bureau records | 1:N via `SK_ID_BUREAU` (must join through `bureau`) |
| `previous_application.csv` | Previous applications at Home Credit | 1:N via `SK_ID_CURR`; own key `SK_ID_PREV` |
| `POS_CASH_balance.csv` | Monthly POS / cash loan snapshots | 1:N via `SK_ID_PREV` |
| `installments_payments.csv` | Repayment history | 1:N via `SK_ID_PREV` |
| `credit_card_balance.csv` | Monthly credit-card balances | 1:N via `SK_ID_PREV` |

## Main table fields (selection)

| Field | Business meaning | Risk relevance |
|-------|------------------|----------------|
| `TARGET` | Payment-difficulty label under the dataset's own definition (1/0); this project does not add a term or an overdue threshold of its own | Target variable |
| `AMT_INCOME_TOTAL` | Annual income | Repayment capacity |
| `AMT_CREDIT` | Loan amount | Risk exposure |
| `AMT_ANNUITY` | Annuity / instalment amount | Repayment burden |
| `DAYS_BIRTH` | Days since birth (negative) | Age; younger applicants historically default more |
| `DAYS_EMPLOYED` | Days since employment start (negative) | Employment stability |
| `EXT_SOURCE_1/2/3` | External credit scores (0–1) | Strongest historical predictors |
| `NAME_CONTRACT_TYPE` | Cash loan vs revolving loan | Product dimension |
| `CODE_GENDER` | Gender | Fairness / compliance risk — requires an explicit decision |

## Known data traps

1. `DAYS_BIRTH` and `DAYS_EMPLOYED` are negative day counts relative to the application
   date; convert to years before using them in a model.
2. `DAYS_EMPLOYED = 365243` is a sentinel for "not employed / missing" (about 1000 years);
   it must be cleaned, otherwise it distorts model behaviour.
3. `EXT_SOURCE_*` are continuous scores, not categorical variables.

## Step 2 database projection (revised contract)

The database pipeline loads a **projection**, not the whole raw file:

| File | Target table | Loaded columns | Primary key |
|------|--------------|----------------|-------------|
| `application_train.csv` | `application_train` | 12 | `sk_id_curr` |
| `bureau.csv` | `bureau` | 7 | `sk_id_bureau` |

The other five downloaded files (`bureau_balance`, `previous_application`,
`POS_CASH_balance`, `installments_payments`, `credit_card_balance`) belong to the
dataset inventory but are **not** inputs of this database pipeline and have no
tables yet. Their candidate keys are recorded as *to be verified*, not as
constraints: for example `(sk_id_prev, num_instalment_version,
num_instalment_number)` in the instalments file is not guaranteed unique, because
one instalment can be paid in several parts.

### Corrected business meaning of the loaded fields

| Field | Meaning used here |
|-------|-------------------|
| `target` | Payment-difficulty label under the dataset's own definition. Not "default = 1, repaid = 0" with an implied term or 90-day threshold. |
| `credit_day_overdue` | Overdue days on that visible bureau record. It is not a count of historical overdue events by itself. |
| `bureau_overdue_cnt` (derived) | Number of visible bureau records whose overdue days are greater than zero. A NULL overdue value is not treated as an observed overdue event. |
| `credit_active` / `bureau_active_cnt` (derived) | `bureau_active_cnt` counts visible bureau records whose status is "Active". It does not claim that each of them carries a balance, and a NULL status is not counted as active. |
| `days_credit` / `bureau_days_credit_recent` (derived) | Per the original field definition: days before the application when that historical credit started. It is **not** a bureau query timestamp. |

### Aggregation and NULL behaviour

- `SUM(amt_credit_sum_debt)` **ignores NULLs**: one record with a known debt of 40
  and one with unknown debt gives 40, which is the sum of the *known* values, not
  the customer's total debt. Every load records how many of these values were
  missing, so a zero count is never read as complete evidence of no risk.
- COUNT-style features are `0` when there is no bureau record, while amount
  features stay `NULL`, because "no record" and "unknown amount" are different
  states.
