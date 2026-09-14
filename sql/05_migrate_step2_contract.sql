-- Step 2 (revised): migrate an existing database to the current contract.
--
-- Run this once, deliberately, before the first load with the new loader.
-- It is NOT part of a load and it never deletes rows: if existing data violates a
-- constraint the script raises and leaves the database as it was, so the operator
-- decides what to do.
--
--   1. back up the research database (pg_dump) and keep the dump;
--   2. run this file with psql -v ON_ERROR_STOP=1;
--   3. the loader re-checks the structure before every load.

BEGIN;

-- application_train: the label becomes NOT NULL plus a binary check.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM application_train WHERE target IS NULL) THEN
        RAISE EXCEPTION
            'application_train.target 存在空值，迁移中止；请先人工确认这些记录';
    END IF;

    IF EXISTS (
        SELECT 1 FROM application_train WHERE target NOT IN (0, 1)
    ) THEN
        RAISE EXCEPTION
            'application_train.target 存在非零一取值，迁移中止';
    END IF;

    ALTER TABLE application_train ALTER COLUMN target SET NOT NULL;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'application_train_target_binary'
    ) THEN
        ALTER TABLE application_train
            ADD CONSTRAINT application_train_target_binary
            CHECK (target IN (0, 1));
    END IF;
END $$;

-- bureau: sk_id_curr must be present, and duplicates in the key are fatal.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM bureau WHERE sk_id_curr IS NULL) THEN
        RAISE EXCEPTION 'bureau.sk_id_curr 存在空值，迁移中止';
    END IF;

    IF EXISTS (
        SELECT sk_id_bureau FROM bureau
        GROUP BY sk_id_bureau HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION 'bureau.sk_id_bureau 存在重复，迁移中止';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'bureau'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE bureau ADD PRIMARY KEY (sk_id_bureau);
    END IF;

    ALTER TABLE bureau ALTER COLUMN sk_id_curr SET NOT NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_bureau_sk_id_curr ON bureau (sk_id_curr);

-- Derived tables are rebuilt structure only: they hold no source of truth, and
-- the next load refills them inside its own transaction.
DROP TABLE IF EXISTS model_input;
DROP TABLE IF EXISTS feat_bureau;

CREATE TABLE feat_bureau (
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

CREATE TABLE model_input (
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
    CONSTRAINT model_input_target_binary CHECK (target IN (0, 1))
);

-- Audit tables are additive.
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

CREATE TABLE IF NOT EXISTS load_failures (
    failure_id    BIGSERIAL PRIMARY KEY,
    load_id       TEXT NOT NULL,
    attempted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage         TEXT NOT NULL,
    error_type    TEXT NOT NULL,
    error_code    TEXT
);

COMMIT;
