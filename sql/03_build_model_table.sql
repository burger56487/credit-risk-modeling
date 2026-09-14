-- Step 2: build the modelling wide table.
-- LEFT JOIN keeps every application, including customers with no bureau record.
-- Count features default to 0 (no records); amount features stay NULL because
-- "zero credit" and "unknown credit" are different states (handled in Step 4).

DROP TABLE IF EXISTS model_input;

CREATE TABLE model_input AS
SELECT
    a.*,
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
