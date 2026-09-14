"""Step 2 tests: verify the SQL aggregation logic on a SQLite in-memory database.

SQLite is used because the aggregation only relies on COUNT / SUM / MAX / CASE
WHEN, which behave the same as in PostgreSQL. The PostgreSQL-specific DDL is
validated separately by the ``sql`` CI job.
"""
import pandas as pd
from sqlalchemy import create_engine


def test_bureau_aggregation_logic():
    """Bureau signals aggregate correctly per customer."""
    engine = create_engine("sqlite:///:memory:")

    bureau = pd.DataFrame({
        "sk_id_curr":         [1, 1, 2],
        "credit_active":      ["Active", "Closed", "Active"],
        "amt_credit_sum":     [1000.0, 2000.0, 500.0],
        "credit_day_overdue": [5, 0, 0],
    })
    bureau.to_sql("bureau", engine, index=False)

    query = """
        SELECT
            sk_id_curr,
            COUNT(*) AS bureau_cnt,
            SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END)
                AS bureau_active_cnt,
            SUM(amt_credit_sum) AS bureau_credit_sum_total,
            MAX(credit_day_overdue) AS bureau_overdue_days_max,
            SUM(CASE WHEN credit_day_overdue > 0 THEN 1 ELSE 0 END)
                AS bureau_overdue_cnt
        FROM bureau
        GROUP BY sk_id_curr
    """
    result = pd.read_sql(query, engine).set_index("sk_id_curr")

    # Customer 1: two records, one active, 3000 total, worst overdue 5 days.
    assert result.loc[1, "bureau_cnt"] == 2
    assert result.loc[1, "bureau_active_cnt"] == 1
    assert result.loc[1, "bureau_credit_sum_total"] == 3000.0
    assert result.loc[1, "bureau_overdue_days_max"] == 5
    assert result.loc[1, "bureau_overdue_cnt"] == 1

    # Customer 2: one active record, 500 total, no overdue.
    assert result.loc[2, "bureau_cnt"] == 1
    assert result.loc[2, "bureau_overdue_cnt"] == 0


def test_left_join_keeps_customers_without_bureau_and_keeps_amounts_null():
    """The model table keeps all applications; count features default to 0 and
    amount features stay NULL when there is no bureau record."""
    engine = create_engine("sqlite:///:memory:")

    pd.DataFrame({"sk_id_curr": [1, 2]}).to_sql(
        "application_train", engine, index=False)
    pd.DataFrame({
        "sk_id_curr":     [1],
        "bureau_cnt":     [2],
        "bureau_debt_total": [1500.0],
    }).to_sql("feat_bureau", engine, index=False)

    query = """
        SELECT
            a.sk_id_curr,
            COALESCE(b.bureau_cnt, 0) AS bureau_cnt,
            b.bureau_debt_total
        FROM application_train a
        LEFT JOIN feat_bureau b ON a.sk_id_curr = b.sk_id_curr
    """
    result = pd.read_sql(query, engine).set_index("sk_id_curr")

    assert len(result) == 2                       # customer without bureau kept
    assert result.loc[2, "bureau_cnt"] == 0       # count defaults to zero
    assert pd.isna(result.loc[2, "bureau_debt_total"])  # amount stays unknown
