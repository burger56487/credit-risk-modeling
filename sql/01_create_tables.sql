-- Step 2 (revised): persistent objects for the raw projection and derived tables.
--
-- This file only initialises structure. It never rebuilds or empties content, so
-- it is safe to run against an existing database and it is not part of a load.
-- An existing database with an older structure needs the explicit migration in
-- sql/05_migrate_step2_contract.sql; CREATE TABLE IF NOT EXISTS will not fix it.
--
-- Column names are lower case because the loader lower-cases every CSV header.
-- Business semantics of the fields are attached as comments; see also
-- data/data_dictionary.md.

-- Projection of the labelled application file: 12 columns of the raw 120+.
CREATE TABLE IF NOT EXISTS application_train (
    sk_id_curr          INTEGER PRIMARY KEY,
    -- Payment-difficulty label under the dataset's own definition. This project
    -- does not add a term or an overdue threshold of its own.
    target              SMALLINT NOT NULL,
    name_contract_type  VARCHAR(50),
    code_gender         VARCHAR(10),
    amt_income_total    NUMERIC,
    amt_credit          NUMERIC,
    amt_annuity         NUMERIC,
    days_birth          INTEGER,   -- negative: days before the application
    days_employed       INTEGER,   -- negative; 365243 is a placeholder code
    ext_source_1        NUMERIC,
    ext_source_2        NUMERIC,
    ext_source_3        NUMERIC,
    CONSTRAINT application_train_target_binary
        CHECK (target IN (0, 1))
);

-- Projection of the bureau file: 7 columns of the raw 17.
-- Records whose sk_id_curr is not in the labelled application file are legal
-- source records, so no foreign key is declared here.
CREATE TABLE IF NOT EXISTS bureau (
    sk_id_bureau        INTEGER PRIMARY KEY,
    sk_id_curr          INTEGER NOT NULL,
    -- Status of the visible bureau record; NULL status is not "active".
    credit_active       VARCHAR(20),
    -- Per the original field definition: days before the application when the
    -- historical credit started. It is not a bureau query timestamp.
    days_credit         INTEGER,
    amt_credit_sum      NUMERIC,
    amt_credit_sum_debt NUMERIC,
    -- Overdue days on the visible bureau record; NULL is not "no overdue".
    credit_day_overdue  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_bureau_sk_id_curr
    ON bureau (sk_id_curr);

-- One row per application in the labelled file. Count columns are NOT NULL
-- because "no bureau record" is a known zero; amount columns stay nullable
-- because "unknown amount" is not zero.
CREATE TABLE IF NOT EXISTS feat_bureau (
    sk_id_curr                 INTEGER PRIMARY KEY,
    bureau_cnt                 INTEGER NOT NULL,
    bureau_active_cnt          INTEGER NOT NULL,
    bureau_credit_sum_total    NUMERIC,
    bureau_credit_sum_avg      NUMERIC,
    bureau_debt_total          NUMERIC,
    bureau_overdue_days_max    INTEGER,
    bureau_overdue_cnt         INTEGER NOT NULL,
    bureau_days_credit_recent  INTEGER
);

-- The modelling wide table. Fields are listed explicitly: the contract is the
-- field list, not whatever the application projection happens to contain.
CREATE TABLE IF NOT EXISTS model_input (
    sk_id_curr                 INTEGER PRIMARY KEY,
    target                     SMALLINT NOT NULL,
    name_contract_type         VARCHAR(50),
    code_gender                VARCHAR(10),
    amt_income_total           NUMERIC,
    amt_credit                 NUMERIC,
    amt_annuity                NUMERIC,
    days_birth                 INTEGER,
    days_employed              INTEGER,
    ext_source_1               NUMERIC,
    ext_source_2               NUMERIC,
    ext_source_3               NUMERIC,
    bureau_cnt                 INTEGER NOT NULL,
    bureau_active_cnt          INTEGER NOT NULL,
    bureau_credit_sum_total    NUMERIC,
    bureau_credit_sum_avg      NUMERIC,
    bureau_debt_total          NUMERIC,
    bureau_overdue_cnt         INTEGER NOT NULL,
    bureau_overdue_days_max    INTEGER,
    bureau_days_credit_recent  INTEGER,
    CONSTRAINT model_input_target_binary
        CHECK (target IN (0, 1))
);

-- Audit of successful loads. The row is written in the same transaction as the
-- data it describes, so it can never survive a rolled-back load.
CREATE TABLE IF NOT EXISTS load_batches (
    load_id           TEXT PRIMARY KEY,
    contract_version  TEXT NOT NULL,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    input_files       JSONB NOT NULL,
    headers           JSONB NOT NULL,
    counts            JSONB NOT NULL,
    linkage           JSONB NOT NULL,
    missing_values    JSONB NOT NULL,
    reconciliation    JSONB NOT NULL,
    script_digests    JSONB NOT NULL
);

-- Audit of failed attempts, written after the main transaction rolled back.
-- Only the failure type is stored, never raw field values.
CREATE TABLE IF NOT EXISTS load_failures (
    failure_id    BIGSERIAL PRIMARY KEY,
    load_id       TEXT NOT NULL,
    attempted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage         TEXT NOT NULL,
    error_type    TEXT NOT NULL,
    error_code    TEXT
);
