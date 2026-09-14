"""Step 2 tests for the CSV ingestion layer (SQLite stand-in for PostgreSQL)."""
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.data_layer.ingest_to_db import ingest_csv


def _make_table(engine, name="bureau"):
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE {name} ("
            "sk_id_curr INTEGER, credit_active TEXT, amt_credit_sum NUMERIC)"
        ))


def test_ingest_csv_only_writes_columns_defined_in_the_table(tmp_path):
    """Extra CSV columns are dropped and headers are lower-cased."""
    csv = tmp_path / "bureau.csv"
    pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "CREDIT_ACTIVE": ["Active", "Closed"],
        "AMT_CREDIT_SUM": [100.0, 200.0],
        "EXTRA_COLUMN": ["x", "y"],          # not in the target table
    }).to_csv(csv, index=False)

    engine = create_engine("sqlite:///:memory:")
    _make_table(engine)
    rows = ingest_csv(engine, "bureau.csv", "bureau", data_dir=tmp_path)

    assert rows == 2
    stored = pd.read_sql("SELECT * FROM bureau", engine)
    assert list(stored.columns) == ["sk_id_curr", "credit_active", "amt_credit_sum"]
    assert stored["sk_id_curr"].tolist() == [1, 2]


def test_ingest_csv_is_idempotent(tmp_path):
    """Running the load twice must not duplicate rows."""
    csv = tmp_path / "bureau.csv"
    pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "CREDIT_ACTIVE": ["Active", "Closed"],
        "AMT_CREDIT_SUM": [100.0, 200.0],
    }).to_csv(csv, index=False)

    engine = create_engine("sqlite:///:memory:")
    _make_table(engine)
    ingest_csv(engine, "bureau.csv", "bureau", data_dir=tmp_path)
    ingest_csv(engine, "bureau.csv", "bureau", data_dir=tmp_path)

    count = pd.read_sql("SELECT COUNT(*) AS n FROM bureau", engine)["n"].iloc[0]
    assert count == 2


def test_ingest_csv_raises_for_missing_file(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    _make_table(engine)
    with pytest.raises(FileNotFoundError):
        ingest_csv(engine, "missing.csv", "bureau", data_dir=tmp_path)
