"""Step 2 (revised): fixed field contract, structure check and staging load.

Two different file lists are kept apart on purpose:

* ``load_raw.EXPECTED_FILES`` is the **dataset inventory** (7 files) used for a
  completeness check of the downloaded data;
* ``PIPELINE_CONTRACT`` below is the **database pipeline requirement** (2 files)
  and the columns this pipeline actually loads.

The current scope is an explicit projection: 12 of the 120+ application columns
and 7 of the 17 bureau columns. Extra source columns are allowed, but they are
recorded as "not loaded in this version" instead of being silently ignored, and
the contract — not the database structure — decides which columns are loaded.

Connection configuration has no default: a missing ``CREDITRISK_DB_URL`` (or an
explicit URL argument) is an error, so a run can never silently fall back to some
other database.
"""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy import types as satypes

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

DB_URL_ENV = "CREDITRISK_DB_URL"

CONTRACT_VERSION = "数据契约第一版"


@dataclass(frozen=True)
class FileContract:
    """One source file, its target table, its loaded columns and its key."""

    file_name: str
    table: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    staging_table: str

    def __post_init__(self):
        if not self.columns:
            raise ValueError(f"{self.file_name} 的列清单不能为空。")
        missing = [c for c in self.primary_key if c not in self.columns]
        if missing:
            raise ValueError(f"{self.file_name} 的主键列必须属于装载列：{missing}")


# The database pipeline requirement: two files, explicit column lists.
PIPELINE_CONTRACT = (
    FileContract(
        file_name="application_train.csv",
        table="application_train",
        columns=(
            "sk_id_curr",
            "target",
            "name_contract_type",
            "code_gender",
            "amt_income_total",
            "amt_credit",
            "amt_annuity",
            "days_birth",
            "days_employed",
            "ext_source_1",
            "ext_source_2",
            "ext_source_3",
        ),
        primary_key=("sk_id_curr",),
        staging_table="stg_application_train",
    ),
    FileContract(
        file_name="bureau.csv",
        table="bureau",
        columns=(
            "sk_id_bureau",
            "sk_id_curr",
            "credit_active",
            "days_credit",
            "amt_credit_sum",
            "amt_credit_sum_debt",
            "credit_day_overdue",
        ),
        primary_key=("sk_id_bureau",),
        staging_table="stg_bureau",
    ),
)

# Expected column type family per table, used by the structure check. The family
# is deliberately coarse: the loader cares that a column accepts the right kind
# of value, not that it was declared with one exact spelling.
EXPECTED_TYPES = {
    "application_train": {
        "sk_id_curr": "integer",
        "target": "smallint",
        "name_contract_type": "character varying",
        "code_gender": "character varying",
        "amt_income_total": "numeric",
        "amt_credit": "numeric",
        "amt_annuity": "numeric",
        "days_birth": "integer",
        "days_employed": "integer",
        "ext_source_1": "numeric",
        "ext_source_2": "numeric",
        "ext_source_3": "numeric",
    },
    "bureau": {
        "sk_id_bureau": "integer",
        "sk_id_curr": "integer",
        "credit_active": "character varying",
        "days_credit": "integer",
        "amt_credit_sum": "numeric",
        "amt_credit_sum_debt": "numeric",
        "credit_day_overdue": "integer",
    },
    "feat_bureau": {
        "sk_id_curr": "integer",
        "bureau_cnt": "integer",
        "bureau_active_cnt": "integer",
        "bureau_credit_sum_total": "numeric",
        "bureau_credit_sum_avg": "numeric",
        "bureau_debt_total": "numeric",
        "bureau_overdue_days_max": "integer",
        "bureau_overdue_cnt": "integer",
        "bureau_days_credit_recent": "integer",
    },
    "model_input": {
        "sk_id_curr": "integer",
        "target": "smallint",
        "name_contract_type": "character varying",
        "code_gender": "character varying",
        "amt_income_total": "numeric",
        "amt_credit": "numeric",
        "amt_annuity": "numeric",
        "days_birth": "integer",
        "days_employed": "integer",
        "ext_source_1": "numeric",
        "ext_source_2": "numeric",
        "ext_source_3": "numeric",
        "bureau_cnt": "integer",
        "bureau_active_cnt": "integer",
        "bureau_credit_sum_total": "numeric",
        "bureau_credit_sum_avg": "numeric",
        "bureau_debt_total": "numeric",
        "bureau_overdue_cnt": "integer",
        "bureau_overdue_days_max": "integer",
        "bureau_days_credit_recent": "integer",
    },
}

# Primary keys the structure check requires on persistent tables.
EXPECTED_PRIMARY_KEYS = {
    "application_train": ("sk_id_curr",),
    "bureau": ("sk_id_bureau",),
    "feat_bureau": ("sk_id_curr",),
    "model_input": ("sk_id_curr",),
}

DERIVED_SCRIPTS = (
    "02_aggregate_bureau.sql",
    "03_build_model_table.sql",
)


def required_files() -> tuple[str, ...]:
    """Database-pipeline requirement, derived from the contract itself."""
    return tuple(item.file_name for item in PIPELINE_CONTRACT)


def contract_for(file_name: str) -> FileContract:
    for item in PIPELINE_CONTRACT:
        if item.file_name == file_name:
            return item
    raise KeyError(f"文件 {file_name} 不在当前装载契约内。")


def get_engine(db_url: str | None = None) -> Engine:
    """Build an engine from an explicit URL or ``CREDITRISK_DB_URL``.

    There is no default connection string on purpose: a missing configuration
    must stop the run instead of pointing somewhere unintended.
    """
    if db_url is not None:
        if not isinstance(db_url, str) or not db_url.strip():
            raise ValueError("显式传入的数据库连接串不能是空白字符串。")
        return create_engine(db_url)

    configured = os.environ.get(DB_URL_ENV)
    if not configured or not configured.strip():
        raise RuntimeError(
            f"未配置数据库连接：请设置环境变量 {DB_URL_ENV}，"
            "本模块不再提供默认连接串兜底。"
        )

    return create_engine(configured)


def normalize_headers(headers) -> list[str]:
    """Lower-case and strip CSV headers, refusing normalisation collisions.

    Two source columns that only differ by case or padding would otherwise be
    renamed by the reader and slip through as one column.
    """
    normalized = []
    collisions = {}
    seen = {}

    for raw in headers:
        name = str(raw).strip().lower()
        if not name:
            raise ValueError("文件表头存在空白列名。")
        if name in seen:
            collisions.setdefault(name, [seen[name]]).append(str(raw))
        else:
            seen[name] = str(raw)
        normalized.append(name)

    if collisions:
        raise ValueError(
            "表头规范化后出现重复列名，拒绝装载："
            + "; ".join(
                f"{name} <- {sources}" for name, sources in collisions.items()
            )
        )

    return normalized


def resolve_columns(contract: FileContract, headers) -> tuple[list[str], list[str]]:
    """Return (loaded columns, extra columns) for one source file.

    A missing required column is fatal; extra columns are reported so the load
    record can state which source fields this version does not load.
    """
    normalized = normalize_headers(headers)

    missing = [c for c in contract.columns if c not in normalized]
    if missing:
        raise ValueError(
            f"{contract.file_name} 缺少装载契约要求的列：{missing}"
        )

    extra = [c for c in normalized if c not in contract.columns]

    return list(contract.columns), extra


def script_digest(file_name: str, sql_dir: Path = SQL_DIR) -> str:
    """SHA-256 of a SQL script, recorded with every successful load."""
    return hashlib.sha256((Path(sql_dir) / file_name).read_bytes()).hexdigest()


def _table_columns(engine: Engine, table_name: str) -> list[str]:
    """Return the column names of an existing table."""
    return [col["name"] for col in inspect(engine).get_columns(table_name)]


def type_family(column_type) -> str:
    """Coarse type family of a reflected column, for the structure check.

    Using SQLAlchemy's own classes avoids spelling differences such as
    ``varchar(50)`` versus ``character varying``.
    """
    if isinstance(column_type, satypes.SmallInteger):
        return "smallint"
    if isinstance(column_type, satypes.Integer):
        return "integer"
    if isinstance(column_type, satypes.Numeric):
        return "numeric"
    if isinstance(column_type, satypes.String):
        return "character varying"
    return str(column_type).lower()


def check_schema(engine: Engine) -> dict:
    """Verify that the persistent tables match the frozen contract.

    The loader calls this before every load: an older structure must be migrated
    deliberately rather than discovered halfway through a refresh.
    """
    problems = []
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table, expected in EXPECTED_TYPES.items():
        if table not in existing_tables:
            problems.append(f"缺少表 {table}")
            continue

        actual = {
            col["name"]: type_family(col["type"])
            for col in inspector.get_columns(table)
        }

        for column, family in expected.items():
            if column not in actual:
                problems.append(f"{table}.{column} 缺失")
                continue

            if actual[column] != family:
                problems.append(
                    f"{table}.{column} 类型为 {actual[column]}，期望 {family}"
                )

        primary_key = inspector.get_pk_constraint(table).get(
            "constrained_columns"
        ) or []
        expected_key = list(EXPECTED_PRIMARY_KEYS.get(table, ()))

        if list(primary_key) != expected_key:
            problems.append(
                f"{table} 主键为 {list(primary_key)}，期望 {expected_key}"
            )

    if problems:
        raise RuntimeError(
            "数据库结构与当前契约不一致，请先执行 sql/05_migrate_step2_contract.sql；"
            + "；".join(problems)
        )

    return {"表": sorted(EXPECTED_TYPES), "契约版本": CONTRACT_VERSION}


# --------------------------------------------------------------------------- #
# Legacy single-file loader.
#
# It empties a target table in its own transaction and then appends chunk by
# chunk, each chunk committing on its own, and it accepts any file/table pair.
# The next commit replaces it with the atomic staged pipeline, so this block is
# kept only until that lands.
# --------------------------------------------------------------------------- #
def _clear_table(engine: Engine, table_name: str) -> None:
    """Empty the table while keeping its schema (legacy behaviour)."""
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
    """Legacy: load one CSV into an existing table and return its row count."""
    import pandas as pd

    csv_path = Path(data_dir) / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件：{csv_path}")

    target_columns = _table_columns(engine, table_name)
    _clear_table(engine, table_name)

    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk.columns = normalize_headers(chunk.columns)
        missing = [c for c in target_columns if c not in chunk.columns]
        if missing:
            raise ValueError(f"{csv_name} 缺少目标表 {table_name} 需要的列：{missing}")
        chunk = chunk[target_columns]
        chunk.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )
        total_rows += len(chunk)

    return total_rows


def ingest_all(engine: Engine, data_dir: Path = RAW_DATA_DIR) -> dict[str, int]:
    """Legacy: ingest every contract file with the single-file loader."""
    return {
        item.table: ingest_csv(engine, item.file_name, item.table, data_dir=data_dir)
        for item in PIPELINE_CONTRACT
    }


if __name__ == "__main__":
    ingest_all(get_engine())
