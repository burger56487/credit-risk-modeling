"""Step 2 (revised): atomic staged load from two source files into four tables.

The pipeline publishes ``application_train``, ``bureau``, ``feat_bureau`` and
``model_input`` as one unit:

    check configuration, contract and input files
        -> one database transaction
        -> transaction-level writer lock (bounded wait)
        -> structure check against the frozen contract
        -> temp staging tables with the same column contract
        -> chunked native bulk copy (same connection)
        -> validate read/staged counts
        -> replace the two persistent source tables (delete + insert)
        -> run the repository aggregation and wide-table scripts
        -> reconcile the four tables
        -> write the success record
        -> one commit

Any failure rolls the whole thing back: the previous usable version stays intact,
because nothing is published and no success record is written. Failures are
recorded afterwards, on a separate connection, and only by error type and code —
never with raw field values.

Scope and vocabulary:

* ``load_raw.EXPECTED_FILES`` is the **dataset inventory** (7 files).
* ``PIPELINE_CONTRACT`` is the **database requirement**: 2 files with explicit
  column lists (12 and 7 columns). Extra source columns are reported as "not
  loaded in this version"; the contract, not the database, decides the fields.

Backends: this pipeline supports PostgreSQL only, because it relies on ``COPY``,
advisory locks and temporary tables. In-memory databases are used for fast
semantic tests elsewhere and are not evidence of transactional behaviour.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy import types as satypes
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

DB_URL_ENV = "CREDITRISK_DB_URL"

CONTRACT_VERSION = "数据契约第一版"

# Fixed key for the transaction-level write lock of this pipeline. It only
# constrains writers that follow the same protocol; it is not a permission.
STEP2_LOCK_KEY = 574448537378

STAGING_SCRIPT = "00_staging_tables.sql"
SCHEMA_SCRIPT = "01_create_tables.sql"
DERIVED_SCRIPTS = (
    "02_aggregate_bureau.sql",
    "03_build_model_table.sql",
)


class LoadStageError(RuntimeError):
    """A load failed at a named stage; the caller may record it as such."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


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

# Coarse type families the loader requires; the structure check compares against
# these instead of one exact spelling.
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

EXPECTED_PRIMARY_KEYS = {
    "application_train": ("sk_id_curr",),
    "bureau": ("sk_id_bureau",),
    "feat_bureau": ("sk_id_curr",),
    "model_input": ("sk_id_curr",),
}

# Fields whose missing counts are recorded with every load, because their zero
# counts must not be read as complete evidence of no risk.
MISSING_VALUE_FIELDS = {
    "application_train": (
        "amt_income_total",
        "amt_credit",
        "amt_annuity",
        "days_birth",
        "days_employed",
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
    ),
    "bureau": (
        "credit_active",
        "days_credit",
        "amt_credit_sum",
        "amt_credit_sum_debt",
        "credit_day_overdue",
    ),
}

# The application columns that must survive the wide-table join unchanged.
PAIRED_FIELDS = (
    "target",
    "amt_income_total",
    "amt_credit",
    "days_birth",
    "ext_source_1",
    "code_gender",
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

    There is no default connection string: a missing configuration stops the run
    instead of pointing somewhere unintended, and an invalid explicit URL is not
    silently replaced by the environment value.
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

    Two source columns that differ only by case or padding would otherwise be
    folded into one by the reader and slip through unnoticed.
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
    """Return (loaded columns in contract order, extra column names)."""
    normalized = normalize_headers(headers)

    missing = [c for c in contract.columns if c not in normalized]
    if missing:
        raise ValueError(f"{contract.file_name} 缺少装载契约要求的列：{missing}")

    extra = [c for c in normalized if c not in contract.columns]

    return list(contract.columns), extra


def read_sql(file_name: str, sql_dir: Path = SQL_DIR) -> str:
    """Read a repository SQL script; the pipeline never inlines a copy of it."""
    path = Path(sql_dir) / file_name
    if not path.is_file():
        raise LoadStageError("契约与文件检查", f"缺少 SQL 脚本：{path}")
    return path.read_text(encoding="utf-8")


def script_digest(file_name: str, sql_dir: Path = SQL_DIR) -> str:
    """SHA-256 of a SQL script, recorded with every successful load."""
    return hashlib.sha256(
        (Path(sql_dir) / file_name).read_bytes()
    ).hexdigest()


def type_family(column_type) -> str:
    """Coarse type family of a reflected column, for the structure check."""
    if isinstance(column_type, satypes.SmallInteger):
        return "smallint"
    if isinstance(column_type, satypes.Integer):
        return "integer"
    if isinstance(column_type, satypes.Numeric):
        return "numeric"
    if isinstance(column_type, satypes.String):
        return "character varying"
    return str(column_type).lower()


def error_code(exc: BaseException) -> str | None:
    """SQLSTATE of a database error, used instead of its (value-bearing) text."""
    for candidate in (exc, getattr(exc, "orig", None)):
        code = getattr(candidate, "pgcode", None)
        if code:
            return str(code)
    return None


def check_schema(connection_or_engine) -> dict:
    """Verify that the persistent tables match the frozen contract.

    The loader calls this before every load: an older structure has to be
    migrated deliberately rather than discovered halfway through a refresh.
    """
    problems = []
    inspector = inspect(connection_or_engine)
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
            elif actual[column] != family:
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
        raise LoadStageError(
            "结构检查",
            "数据库结构与当前契约不一致，请先执行 "
            f"sql/{SCHEMA_SCRIPT} 或 sql/05_migrate_step2_contract.sql；"
            + "；".join(problems),
        )

    return {"表": sorted(EXPECTED_TYPES), "契约版本": CONTRACT_VERSION}


@dataclass(frozen=True)
class InputSnapshot:
    """Controlled local copies of the input files, with their digests."""

    directory: Path
    entries: tuple[dict, ...]

    def entry_for(self, file_name: str) -> dict:
        for entry in self.entries:
            if entry["文件"] == file_name:
                return entry
        raise KeyError(file_name)


def snapshot_inputs(
    data_dir: Path = RAW_DATA_DIR,
    work_dir: Path | None = None,
) -> InputSnapshot:
    """Copy the contract files locally and digest those copies.

    The digest must describe the bytes that are actually loaded; hashing a file
    and later re-reading a path that may have changed would not prove that.
    """
    directory = (
        Path(work_dir)
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="creditrisk-load-"))
    )
    directory.mkdir(parents=True, exist_ok=True)

    entries = []

    for item in PIPELINE_CONTRACT:
        source = Path(data_dir) / item.file_name
        if not source.is_file():
            raise LoadStageError(
                "契约与文件检查", f"找不到输入文件：{source}"
            )

        target = directory / item.file_name
        digest = hashlib.sha256()
        size = 0

        with source.open("rb") as source_handle, target.open("wb") as target_handle:
            while True:
                block = source_handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                target_handle.write(block)

        entries.append(
            {
                "文件": item.file_name,
                "表": item.table,
                "sha256": digest.hexdigest(),
                "字节数": size,
            }
        )

    return InputSnapshot(directory=directory, entries=tuple(entries))


def cleanup_snapshot(snapshot: InputSnapshot | None) -> None:
    if snapshot is not None:
        shutil.rmtree(snapshot.directory, ignore_errors=True)


def _staging_columns(connection, table_name: str) -> list[str]:
    return [col["name"] for col in inspect(connection).get_columns(table_name)]


def _create_staging(connection) -> None:
    """Create the temp staging tables from the repository script."""
    connection.exec_driver_sql(read_sql(STAGING_SCRIPT))

    for item in PIPELINE_CONTRACT:
        actual = _staging_columns(connection, item.staging_table)
        if actual != list(item.columns):
            raise LoadStageError(
                "暂存装载",
                f"{item.staging_table} 的列与契约不一致：{actual} != "
                f"{list(item.columns)}",
            )


def _copy_file_to_staging(
    connection,
    contract: FileContract,
    csv_path: Path,
    chunk_rows: int,
) -> dict:
    """Stream one file into its staging table with native bulk copy.

    Values are read as text (no float round-trip for amounts, and no automatic
    missing-value inference), then copied on the *same* connection and inside the
    *same* transaction. Multi-row inserts through the data library are not used.
    """
    columns = ", ".join(contract.columns)
    copy_sql = (
        f"COPY {contract.staging_table} ({columns}) "
        "FROM STDIN WITH (FORMAT csv, NULL '')"
    )
    raw_connection = connection.connection.driver_connection

    rows = 0
    header_info = None

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_rows,
        dtype=str,
        na_filter=False,
        keep_default_na=False,
    )

    for frame in reader:
        try:
            frame.columns = normalize_headers(frame.columns)
        except ValueError as exc:
            raise LoadStageError("暂存装载", str(exc)) from exc

        if header_info is None:
            try:
                loaded, extra = resolve_columns(contract, frame.columns)
            except ValueError as exc:
                raise LoadStageError("暂存装载", str(exc)) from exc
            header_info = {
                "规范化表头": list(frame.columns),
                "实际装载列": loaded,
                "未装载列": extra,
            }
        else:
            loaded = list(contract.columns)

        frame = frame.loc[:, loaded]

        buffer = io.BytesIO()
        frame.to_csv(
            buffer, index=False, header=False, encoding="utf-8", lineterminator="\n"
        )
        buffer.seek(0)

        with raw_connection.cursor() as cursor:
            cursor.copy_expert(copy_sql, buffer)

        rows += len(frame)

    if header_info is None:  # an empty file still has a header
        header = pd.read_csv(csv_path, nrows=0).columns
        loaded, extra = resolve_columns(contract, header)
        header_info = {
            "规范化表头": normalize_headers(header),
            "实际装载列": loaded,
            "未装载列": extra,
        }

    header_info["记录数"] = rows
    header_info["文件"] = contract.file_name
    header_info["表"] = contract.table

    return header_info


def _publish_source_table(connection, contract: FileContract) -> int:
    """Replace the persistent source table content inside the transaction.

    Delete + insert (not TRUNCATE, not drop/recreate): the table object, its
    primary key and its indexes stay, and readers keep seeing the previous
    committed version until this transaction commits. The cost is extra WAL and
    old row versions, which is the accepted trade-off for correctness here.
    """
    columns = ", ".join(contract.columns)

    connection.exec_driver_sql(f"DELETE FROM {contract.table}")
    result = connection.exec_driver_sql(
        f"INSERT INTO {contract.table} ({columns}) "
        f"SELECT {columns} FROM {contract.staging_table}"
    )

    return int(result.rowcount)


def _run_derived_scripts(connection) -> None:
    for name in DERIVED_SCRIPTS:
        connection.exec_driver_sql(read_sql(name))


def _scalar(connection, sql: str, parameters: tuple = ()) -> int:
    value = connection.exec_driver_sql(sql, parameters).scalar()
    return int(value or 0)


def _reconcile(connection, published: dict, file_records: dict) -> dict:
    """Cross-check the four tables and raise if any invariant fails."""
    counts = {
        "文件数据记录数": dict(file_records),
        "暂存表行数": {
            item.table: _scalar(
                connection, f"SELECT count(*) FROM {item.staging_table}"
            )
            for item in PIPELINE_CONTRACT
        },
        "持久源表行数": {
            item.table: _scalar(connection, f"SELECT count(*) FROM {item.table}")
            for item in PIPELINE_CONTRACT
        },
        "派生表行数": {
            "feat_bureau": _scalar(connection, "SELECT count(*) FROM feat_bureau"),
            "model_input": _scalar(connection, "SELECT count(*) FROM model_input"),
        },
    }

    checks = []

    for item in PIPELINE_CONTRACT:
        read_rows = file_records[item.file_name]
        staged_rows = counts["暂存表行数"][item.table]
        published_rows = published[item.table]
        stored_rows = counts["持久源表行数"][item.table]

        if read_rows != staged_rows:
            checks.append(
                f"{item.file_name} 读取 {read_rows} 行，暂存表 {staged_rows} 行"
            )
        if staged_rows != published_rows or published_rows != stored_rows:
            checks.append(
                f"{item.table} 暂存 {staged_rows} 行，替换 {published_rows} 行，"
                f"落库 {stored_rows} 行"
            )

    aggregate_expected = _scalar(
        connection, "SELECT count(DISTINCT sk_id_curr) FROM bureau"
    )
    if counts["派生表行数"]["feat_bureau"] != aggregate_expected:
        checks.append(
            "聚合表行数与征信申请数不一致："
            f"{counts['派生表行数']['feat_bureau']} != {aggregate_expected}"
        )

    applications = counts["持久源表行数"]["application_train"]
    if counts["派生表行数"]["model_input"] != applications:
        checks.append(
            f"建模宽表行数 {counts['派生表行数']['model_input']} != 申请数 {applications}"
        )

    key_mismatch = _scalar(
        connection,
        """
        SELECT count(*)
        FROM model_input m
        FULL OUTER JOIN application_train a USING (sk_id_curr)
        WHERE m.sk_id_curr IS NULL OR a.sk_id_curr IS NULL
        """,
    )
    if key_mismatch:
        checks.append(f"宽表与申请表主键集合不一致：{key_mismatch} 条")

    field_difference = " OR ".join(
        f"a.{name} IS DISTINCT FROM m.{name}" for name in PAIRED_FIELDS
    )
    field_mismatch = _scalar(
        connection,
        "SELECT count(*) FROM application_train a "
        "JOIN model_input m USING (sk_id_curr) "
        f"WHERE {field_difference}",
    )
    if field_mismatch:
        checks.append(f"联接前后申请字段发生变化：{field_mismatch} 条")

    linkage = {
        "有征信申请数": _scalar(
            connection,
            "SELECT count(*) FROM application_train a "
            "WHERE EXISTS (SELECT 1 FROM bureau b WHERE b.sk_id_curr = a.sk_id_curr)",
        ),
        "征信中不属于主表的申请数": _scalar(
            connection,
            "SELECT count(*) FROM bureau b "
            "WHERE NOT EXISTS "
            "(SELECT 1 FROM application_train a WHERE a.sk_id_curr = b.sk_id_curr)",
        ),
        "征信记录数": counts["持久源表行数"]["bureau"],
    }
    linkage["无征信申请数"] = applications - linkage["有征信申请数"]

    missing_values = {}
    for table, fields in MISSING_VALUE_FIELDS.items():
        expressions = ", ".join(
            f"count(*) FILTER (WHERE {field} IS NULL)" for field in fields
        )
        row = connection.exec_driver_sql(
            f"SELECT {expressions} FROM {table}"
        ).first()
        missing_values[table] = dict(zip(fields, (int(value) for value in row)))

    reconciliation = {
        "未通过检查": checks,
        "主键集合一致": not key_mismatch,
        "主表字段在联接前后一致": not field_mismatch,
        "聚合行数与征信申请数一致": counts["派生表行数"]["feat_bureau"]
        == aggregate_expected,
        "宽表行数与申请数一致": counts["派生表行数"]["model_input"]
        == applications,
    }

    if checks:
        raise LoadStageError("对账", "；".join(checks))

    return {
        "counts": counts,
        "linkage": linkage,
        "missing_values": missing_values,
        "reconciliation": reconciliation,
    }


def _insert_load_record(connection, record: dict) -> None:
    connection.exec_driver_sql(
        """
        INSERT INTO load_batches (
            load_id, contract_version, input_files, headers, counts,
            linkage, missing_values, reconciliation, script_digests
        ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
                  %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
        """,
        (
            record["装载编号"],
            record["数据契约版本"],
            json.dumps(record["输入文件"], ensure_ascii=False),
            json.dumps(record["表头与列"], ensure_ascii=False),
            json.dumps(record["数量"], ensure_ascii=False),
            json.dumps(record["关联"], ensure_ascii=False),
            json.dumps(record["缺失数量"], ensure_ascii=False),
            json.dumps(record["对账"], ensure_ascii=False),
            json.dumps(record["脚本摘要"], ensure_ascii=False),
        ),
    )


def load_pipeline(
    engine: Engine,
    load_id: str,
    *,
    data_dir: Path = RAW_DATA_DIR,
    chunk_rows: int = 50_000,
    lock_timeout_ms: int = 5_000,
) -> dict:
    """Publish both source tables and both derived tables as one unit."""
    if not isinstance(load_id, str) or not load_id.strip():
        raise ValueError("装载编号必须是非空字符串。")
    if chunk_rows < 1:
        raise ValueError("分块行数必须是正整数。")
    if lock_timeout_ms < 1:
        raise ValueError("锁等待超时必须是正整数毫秒。")
    if engine.dialect.name != "postgresql":
        raise LoadStageError(
            "契约与文件检查",
            "原子装载管道只支持 PostgreSQL；内存数据库仅用于快速语义测试，"
            "不构成事务能力证明。",
        )

    snapshot = None
    stage = "契约与文件检查"

    try:
        snapshot = snapshot_inputs(data_dir=data_dir)

        with engine.connect() as connection:
            with connection.begin():
                stage = "写入锁"
                connection.exec_driver_sql(
                    f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'"
                )
                try:
                    connection.exec_driver_sql(
                        "SELECT pg_advisory_xact_lock(%s)", (STEP2_LOCK_KEY,)
                    )
                except DBAPIError as exc:
                    if error_code(exc) == "55P03":
                        raise LoadStageError(
                            "写入锁",
                            "等待管道写入锁超时，另一个装载任务可能正在运行。",
                        ) from exc
                    raise

                stage = "结构检查"
                check_schema(connection)

                stage = "成功记录"
                already = connection.exec_driver_sql(
                    "SELECT 1 FROM load_batches WHERE load_id = %s", (load_id,)
                ).first()
                if already:
                    raise LoadStageError(
                        "成功记录", f"装载编号已存在：{load_id}"
                    )

                stage = "暂存装载"
                _create_staging(connection)

                header_payload = {}
                file_records = {}

                for item in PIPELINE_CONTRACT:
                    entry = snapshot.entry_for(item.file_name)
                    info = _copy_file_to_staging(
                        connection, item, snapshot.directory / item.file_name,
                        chunk_rows,
                    )
                    header_payload[item.file_name] = info
                    file_records[item.file_name] = info["记录数"]

                stage = "持久表替换"
                published = {
                    item.table: _publish_source_table(connection, item)
                    for item in PIPELINE_CONTRACT
                }

                stage = "派生表刷新"
                _run_derived_scripts(connection)

                stage = "对账"
                parts = _reconcile(
                    connection,
                    published=published,
                    file_records=file_records,
                )

                record = {
                    "装载编号": load_id,
                    "数据契约版本": CONTRACT_VERSION,
                    "输入文件": list(snapshot.entries),
                    "表头与列": header_payload,
                    "数量": parts["counts"],
                    "关联": parts["linkage"],
                    "缺失数量": parts["missing_values"],
                    "对账": parts["reconciliation"],
                    "脚本摘要": {
                        name: script_digest(name)
                        for name in (STAGING_SCRIPT, SCHEMA_SCRIPT) + DERIVED_SCRIPTS
                    },
                }

                stage = "成功记录"
                _insert_load_record(connection, record)

        return record

    except Exception as exc:
        # The transaction has rolled back by now; record the failure separately.
        record_failure(engine, load_id, stage, exc)
        raise

    finally:
        cleanup_snapshot(snapshot)


def record_failure(
    engine: Engine,
    load_id: str,
    stage: str,
    exc: BaseException,
) -> None:
    """Record a failed attempt on a separate connection, failing silently.

    Only the stage, the error type and the SQLSTATE are stored: database error
    text can quote raw field values, which must not end up in a public log. If the
    database itself is unwritable the original exception still propagates.
    """
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                """
                INSERT INTO load_failures (load_id, stage, error_type, error_code)
                VALUES (%s, %s, %s, %s)
                """,
                (load_id, stage, type(exc).__name__, error_code(exc)),
            )
    except Exception:  # noqa: BLE001 - the original failure must win
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-id", required=True, help="本次装载的唯一编号")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--lock-timeout-ms", type=int, default=5_000)
    arguments = parser.parse_args(argv)

    engine = get_engine(arguments.db_url)
    record = load_pipeline(
        engine,
        arguments.load_id,
        data_dir=arguments.data_dir,
        chunk_rows=arguments.chunk_rows,
        lock_timeout_ms=arguments.lock_timeout_ms,
    )

    print("装载成功：", record["装载编号"])
    print("输入文件：", json.dumps(record["输入文件"], ensure_ascii=False))
    print("数量：", json.dumps(record["数量"], ensure_ascii=False))
    print("关联：", json.dumps(record["关联"], ensure_ascii=False))
    print("缺失数量：", json.dumps(record["缺失数量"], ensure_ascii=False))


if __name__ == "__main__":
    main()
