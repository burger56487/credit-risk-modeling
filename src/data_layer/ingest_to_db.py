"""Step 2: ingest the raw CSVs into PostgreSQL.

This layer only moves data; cleaning happens in Step 4. The target tables must
already exist (see ``sql/01_create_tables.sql``); the script appends into them
so that the column types defined in SQL are preserved.
"""
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Credentials come from the environment; never hard-code a password.
DEFAULT_DB_URL = "postgresql+psycopg2://creditrisk:creditrisk@localhost:5432/creditrisk"

# CSV file -> target table, in load order.
CSV_TABLE_MAP = [
    ("application_train.csv", "application_train"),
    ("bureau.csv", "bureau"),
]


def get_engine(db_url: str | None = None) -> Engine:
    """Create a database engine from an explicit URL or ``CREDITRISK_DB_URL``."""
    return create_engine(db_url or os.environ.get("CREDITRISK_DB_URL", DEFAULT_DB_URL))


def _table_columns(engine: Engine, table_name: str) -> list[str]:
    """Return the column names of an existing table."""
    return [col["name"] for col in inspect(engine).get_columns(table_name)]


def _clear_table(engine: Engine, table_name: str) -> None:
    """Empty the table while keeping its schema, so the load is idempotent."""
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text(f"TRUNCATE TABLE {table_name}"))
        else:  # e.g. SQLite in unit tests
            conn.execute(text(f"DELETE FROM {table_name}"))


def ingest_csv(
    engine: Engine,
    csv_name: str,
    table_name: str,
    chunksize: int = 50_000,
    data_dir: Path = RAW_DATA_DIR,
) -> int:
    """Load one CSV into an existing table and return the number of rows.

    Only the columns defined in the target table are written; extra CSV columns
    are dropped and missing required columns raise an error.
    """
    csv_path = Path(data_dir) / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件：{csv_path}")

    target_columns = _table_columns(engine, table_name)
    _clear_table(engine, table_name)

    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk.columns = [c.lower() for c in chunk.columns]
        missing = [c for c in target_columns if c not in chunk.columns]
        if missing:
            raise ValueError(f"{csv_name} 缺少目标表 {table_name} 需要的列：{missing}")
        chunk = chunk[target_columns]
        chunk.to_sql(
            table_name,
            engine,
            if_exists="append",   # never replace: the SQL schema defines the types
            index=False,
            method="multi",
            chunksize=1_000,      # limit parameters per statement
        )
        total_rows += len(chunk)
        print(f"  [{table_name}] 已导入 {total_rows} 行...")

    print(f"[OK] {csv_name} → 表 {table_name}，共 {total_rows} 行。")
    return total_rows


def ingest_all(engine: Engine, pairs: Iterable[tuple[str, str]] = CSV_TABLE_MAP,
               data_dir: Path = RAW_DATA_DIR) -> dict[str, int]:
    """Ingest every configured CSV and return {table: rows}."""
    return {table: ingest_csv(engine, csv, table, data_dir=data_dir)
            for csv, table in pairs}


if __name__ == "__main__":
    ingest_all(get_engine())
