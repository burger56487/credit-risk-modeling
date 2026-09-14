-- Step 2: aggregate the bureau table to one row per customer.
-- Business meaning: overall credit behaviour at other institutions.

DROP TABLE IF EXISTS feat_bureau;

CREATE TABLE feat_bureau AS
SELECT
    sk_id_curr,

    -- Number of credit records elsewhere (borrowing activity).
    COUNT(*)                                    AS bureau_cnt,

    -- Records currently active (current external debt).
    SUM(CASE WHEN credit_active = 'Active'
             THEN 1 ELSE 0 END)                 AS bureau_active_cnt,

    -- Total and average credit granted (exposure).
    SUM(amt_credit_sum)                         AS bureau_credit_sum_total,
    AVG(amt_credit_sum)                         AS bureau_credit_sum_avg,

    -- Current debt (repayment pressure).
    SUM(amt_credit_sum_debt)                    AS bureau_debt_total,

    -- Worst overdue days (strong risk signal).
    MAX(credit_day_overdue)                     AS bureau_overdue_days_max,

    -- Number of records with any overdue history.
    SUM(CASE WHEN credit_day_overdue > 0
             THEN 1 ELSE 0 END)                 AS bureau_overdue_cnt,

    -- Most recent bureau record: days are negative, so MAX is the latest.
    MAX(days_credit)                            AS bureau_days_credit_recent

FROM bureau
GROUP BY sk_id_curr;

CREATE INDEX IF NOT EXISTS idx_feat_bureau_sk_id_curr
    ON feat_bureau (sk_id_curr);
