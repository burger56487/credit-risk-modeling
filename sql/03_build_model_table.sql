-- Step 2 (revised): refresh the modelling wide table.
--
-- Run inside the load transaction, after 02_aggregate_bureau.sql. The table
-- structure is declared in 01_create_tables.sql, so this script only refreshes
-- content and lists every column explicitly — no "SELECT a.*" that would silently
-- widen the contract.
--
-- Semantics kept from the earlier version:
--   * the labelled application projection is the base, LEFT JOINed to the
--     aggregate, so applications without a bureau record are kept;
--   * count features default to 0 (no record is a known zero);
--   * amount features stay NULL when there is no record (unknown is not zero).

DELETE FROM model_input;

INSERT INTO model_input (
    sk_id_curr,
    target,
    name_contract_type,
    code_gender,
    amt_income_total,
    amt_credit,
    amt_annuity,
    days_birth,
    days_employed,
    ext_source_1,
    ext_source_2,
    ext_source_3,
    bureau_cnt,
    bureau_active_cnt,
    bureau_credit_sum_total,
    bureau_credit_sum_avg,
    bureau_debt_total,
    bureau_overdue_cnt,
    bureau_overdue_days_max,
    bureau_days_credit_recent
)
SELECT
    a.sk_id_curr,
    a.target,
    a.name_contract_type,
    a.code_gender,
    a.amt_income_total,
    a.amt_credit,
    a.amt_annuity,
    a.days_birth,
    a.days_employed,
    a.ext_source_1,
    a.ext_source_2,
    a.ext_source_3,
    COALESCE(b.bureau_cnt, 0)              AS bureau_cnt,
    COALESCE(b.bureau_active_cnt, 0)       AS bureau_active_cnt,
    b.bureau_credit_sum_total,
    b.bureau_credit_sum_avg,
    b.bureau_debt_total,
    COALESCE(b.bureau_overdue_cnt, 0)      AS bureau_overdue_cnt,
    b.bureau_overdue_days_max,
    b.bureau_days_credit_recent
FROM application_train a
LEFT JOIN feat_bureau b
    ON a.sk_id_curr = b.sk_id_curr;
