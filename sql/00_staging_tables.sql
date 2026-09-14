-- Step 2 (revised): transaction-local staging tables.
--
-- These are TEMP tables with the same column contract as the persistent source
-- tables plus NOT NULL / primary-key constraints, so a bad file is rejected while
-- it is still staging and never reaches a published table. They are dropped by
-- the server when the transaction ends, so one load cannot pollute the next.
--
-- The loader reads this file; it must not be replaced by a copy inside a test.

CREATE TEMP TABLE IF NOT EXISTS stg_application_train (
    sk_id_curr          INTEGER PRIMARY KEY,
    target              SMALLINT NOT NULL,
    name_contract_type  VARCHAR(50),
    code_gender         VARCHAR(10),
    amt_income_total    NUMERIC,
    amt_credit          NUMERIC,
    amt_annuity         NUMERIC,
    days_birth          INTEGER,
    days_employed       INTEGER,
    ext_source_1        NUMERIC,
    ext_source_2        NUMERIC,
    ext_source_3        NUMERIC,
    CONSTRAINT stg_application_train_target_binary
        CHECK (target IN (0, 1))
) ON COMMIT DROP;

CREATE TEMP TABLE IF NOT EXISTS stg_bureau (
    sk_id_bureau        INTEGER PRIMARY KEY,
    sk_id_curr          INTEGER NOT NULL,
    credit_active       VARCHAR(20),
    days_credit         INTEGER,
    amt_credit_sum      NUMERIC,
    amt_credit_sum_debt NUMERIC,
    credit_day_overdue  INTEGER
) ON COMMIT DROP;
