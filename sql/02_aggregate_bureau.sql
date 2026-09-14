-- Step 2 (revised): refresh the bureau aggregate.
--
-- Run inside the load transaction. It clears this transaction's view of the
-- aggregate table and refills it from the newly staged bureau content; the table
-- object, its primary key and its types stay in place.
--
-- Counting rules, kept explicit because they are easy to over-read:
--   * bureau_active_cnt counts visible records whose status is exactly 'Active';
--     a NULL status is not active, and this is not a count of accounts with a
--     balance;
--   * bureau_overdue_cnt counts records whose overdue days are greater than zero,
--     so a NULL overdue value is not an observed overdue event;
--   * SUM(amt_credit_sum_debt) ignores NULLs: it is the sum of the *known* debts,
--     not the customer's total debt. Missing-value counts are recorded per load.

DELETE FROM feat_bureau;

INSERT INTO feat_bureau (
    sk_id_curr,
    bureau_cnt,
    bureau_active_cnt,
    bureau_credit_sum_total,
    bureau_credit_sum_avg,
    bureau_debt_total,
    bureau_overdue_days_max,
    bureau_overdue_cnt,
    bureau_days_credit_recent
)
SELECT
    sk_id_curr,
    COUNT(*)                                        AS bureau_cnt,
    SUM(CASE WHEN credit_active = 'Active'
             THEN 1 ELSE 0 END)                     AS bureau_active_cnt,
    SUM(amt_credit_sum)                             AS bureau_credit_sum_total,
    AVG(amt_credit_sum)                             AS bureau_credit_sum_avg,
    SUM(amt_credit_sum_debt)                        AS bureau_debt_total,
    MAX(credit_day_overdue)                         AS bureau_overdue_days_max,
    SUM(CASE WHEN credit_day_overdue > 0
             THEN 1 ELSE 0 END)                     AS bureau_overdue_cnt,
    -- Day counts are negative, so MAX is the most recent credit start.
    MAX(days_credit)                                AS bureau_days_credit_recent
FROM bureau
GROUP BY sk_id_curr;
