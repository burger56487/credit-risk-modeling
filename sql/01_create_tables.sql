-- Step 2: create the raw tables in PostgreSQL.
-- Column names are lower case to match the ingestion script, which lower-cases
-- every CSV header before writing to the database.

-- Main table: application information + target.
CREATE TABLE IF NOT EXISTS application_train (
    sk_id_curr          INTEGER PRIMARY KEY,
    target              SMALLINT,              -- 1 = default, 0 = repaid
    name_contract_type  VARCHAR(50),
    code_gender         VARCHAR(10),
    amt_income_total    NUMERIC,
    amt_credit          NUMERIC,
    amt_annuity         NUMERIC,
    days_birth          INTEGER,               -- negative: days before application
    days_employed       INTEGER,               -- negative; 365243 is a sentinel
    ext_source_1        NUMERIC,
    ext_source_2        NUMERIC,
    ext_source_3        NUMERIC
    -- The raw CSV has 120+ columns; only the core modelling columns are kept here.
);

-- Credit history at other institutions (1:N with the main table).
CREATE TABLE IF NOT EXISTS bureau (
    sk_id_bureau        INTEGER PRIMARY KEY,
    sk_id_curr          INTEGER,               -- foreign key to application_train
    credit_active       VARCHAR(20),           -- Active / Closed / ...
    days_credit         INTEGER,               -- negative: days before application
    amt_credit_sum      NUMERIC,               -- total credit granted
    amt_credit_sum_debt NUMERIC,               -- current debt
    credit_day_overdue  INTEGER                -- days past due
);

CREATE INDEX IF NOT EXISTS idx_bureau_sk_id_curr
    ON bureau (sk_id_curr);
