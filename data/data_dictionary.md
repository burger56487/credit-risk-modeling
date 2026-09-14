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
| `TARGET` | Default / payment difficulty (1/0) | Target variable |
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
