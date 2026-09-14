"""Step 2 (revised) acceptance: the loader against a real PostgreSQL instance.

These tests are the transactional evidence the fast semantic tests cannot give:
they run the repository's real staging script, real aggregation and real
wide-table script through the real loader, then check what a failure leaves
behind.

They are skipped unless ``CREDITRISK_TEST_DB_URL`` points at a throwaway test
database whose name contains "test" (or ``CREDITRISK_TEST_DB_CONFIRM=yes`` is set
explicitly). The tests drop and rebuild the four business tables, so pointing them
at a development database would destroy it.
"""
import os
import threading
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.data_layer import ingest_to_db as ing


TEST_DB_URL = os.environ.get("CREDITRISK_TEST_DB_URL", "")

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not TEST_DB_URL,
        reason="未配置 CREDITRISK_TEST_DB_URL，跳过实库集成测试",
    ),
]


def _assert_throwaway_database(url: str) -> None:
    """Refuse to run destructive tests against a non-test database."""
    database = make_url(url).database or ""
    confirmed = os.environ.get("CREDITRISK_TEST_DB_CONFIRM") == "yes"

    if "test" not in database.lower() and not confirmed:
        raise RuntimeError(
            "实库测试会重建四张业务表，只允许连接到名称含 test 的数据库；"
            "如确需其他名称，请显式设置 CREDITRISK_TEST_DB_CONFIRM=yes。"
        )


BUSINESS_TABLES = (
    "model_input",
    "feat_bureau",
    "bureau",
    "application_train",
    "load_batches",
    "load_failures",
)

APPLICATION_FILE = "application_train.csv"
BUREAU_FILE = "bureau.csv"


def sample_applications() -> pd.DataFrame:
    """The fixed acceptance sample: three labelled applications."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
            "CODE_GENDER": ["F", "M", "F"],
            "AMT_INCOME_TOTAL": ["100000", "50000", "80000"],
            "AMT_CREDIT": ["200000", "100000", "150000"],
            "AMT_ANNUITY": ["10000", "5000", "7000"],
            "DAYS_BIRTH": ["-10000", "-12000", "-11000"],
            "DAYS_EMPLOYED": ["-2000", "365243", "-1500"],
            "EXT_SOURCE_1": ["0.7", "", "0.6"],
            "EXT_SOURCE_2": ["0.6", "0.5", ""],
            "EXT_SOURCE_3": ["0.5", "0.4", "0.55"],
        }
    )


def sample_bureau() -> pd.DataFrame:
    """Four bureau records, one of them belonging to an unlabelled application."""
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [101, 102, 103, 104],
            "SK_ID_CURR": [1, 1, 3, 999],
            "CREDIT_ACTIVE": ["Active", "Closed", "", "Active"],
            "DAYS_CREDIT": ["-10", "-30", "-5", "-1"],
            "AMT_CREDIT_SUM": ["100", "200", "", "500"],
            "AMT_CREDIT_SUM_DEBT": ["40", "", "", "50"],
            "CREDIT_DAY_OVERDUE": ["5", "0", "", "0"],
        }
    )


def write_sample(
    directory: Path,
    applications: pd.DataFrame | None = None,
    bureau: pd.DataFrame | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (applications if applications is not None else sample_applications()).to_csv(
        directory / APPLICATION_FILE, index=False
    )
    (bureau if bureau is not None else sample_bureau()).to_csv(
        directory / BUREAU_FILE, index=False
    )
    return directory


@pytest.fixture(scope="session")
def engine():
    _assert_throwaway_database(TEST_DB_URL)
    return create_engine(TEST_DB_URL)


@pytest.fixture
def database(engine):
    """Rebuild the schema from the repository DDL before each test."""
    with engine.begin() as connection:
        for table in BUSINESS_TABLES:
            connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        connection.exec_driver_sql(ing.read_sql(ing.SCHEMA_SCRIPT))

    return engine


def table_state(engine) -> dict:
    """Sorted content of the four business tables plus the audit counts."""
    state = {}

    for table, order in (
        ("application_train", "sk_id_curr"),
        ("bureau", "sk_id_bureau"),
        ("feat_bureau", "sk_id_curr"),
        ("model_input", "sk_id_curr"),
    ):
        state[table] = pd.read_sql(
            f"SELECT * FROM {table} ORDER BY {order}", engine
        )

    state["load_ids"] = pd.read_sql(
        "SELECT load_id FROM load_batches ORDER BY load_id", engine
    )["load_id"].tolist()
    state["failure_count"] = int(
        pd.read_sql("SELECT count(*) AS n FROM load_failures", engine)["n"].iloc[0]
    )

    return state


def assert_tables_unchanged(before: dict, after: dict) -> None:
    for key in ("application_train", "bureau", "feat_bureau", "model_input"):
        pd.testing.assert_frame_equal(before[key], after[key], check_dtype=False)


def assert_nothing_published(before: dict, after: dict) -> None:
    """A failed load must change neither the tables nor the success records."""
    assert_tables_unchanged(before, after)
    assert before["load_ids"] == after["load_ids"]


def run_load(engine, load_id: str, data_dir: Path, **overrides):
    options = {"chunk_rows": 2, "lock_timeout_ms": 5_000}
    options.update(overrides)
    return ing.load_pipeline(engine, load_id, data_dir=data_dir, **options)


def last_failure(engine) -> dict:
    frame = pd.read_sql(
        "SELECT load_id, stage, error_type, error_code FROM load_failures "
        "ORDER BY failure_id DESC LIMIT 1",
        engine,
    )
    assert len(frame) == 1
    return frame.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Success path
# --------------------------------------------------------------------------- #
def test_fixed_sample_loads_into_the_expected_tables(database, tmp_path):
    data_dir = write_sample(tmp_path / "raw")

    record = run_load(database, "验收_第一个成功装载", data_dir)

    assert record["数量"]["文件数据记录数"] == {
        APPLICATION_FILE: 3,
        BUREAU_FILE: 4,
    }
    assert record["数量"]["派生表行数"] == {"feat_bureau": 3, "model_input": 3}
    assert record["关联"] == {
        "有征信申请数": 2,
        "征信中不属于主表的申请数": 1,
        "征信记录数": 4,
        "无征信申请数": 1,
    }
    assert record["缺失数量"]["bureau"] == {
        "credit_active": 1,
        "days_credit": 0,
        "amt_credit_sum": 1,
        "amt_credit_sum_debt": 2,
        "credit_day_overdue": 1,
    }
    assert record["对账"]["未通过检查"] == []

    # The raw bureau table keeps the record that belongs to no labelled application.
    assert pd.read_sql("SELECT count(*) AS n FROM bureau", database)["n"].iloc[0] == 4

    aggregate = pd.read_sql(
        "SELECT sk_id_curr FROM feat_bureau ORDER BY sk_id_curr", database
    )["sk_id_curr"].tolist()
    assert aggregate == [1, 3, 999]

    wide = pd.read_sql(
        """
        SELECT sk_id_curr, bureau_cnt, bureau_credit_sum_total, bureau_debt_total,
               bureau_overdue_cnt
        FROM model_input ORDER BY sk_id_curr
        """,
        database,
    ).set_index("sk_id_curr")

    assert wide.index.tolist() == [1, 2, 3]
    assert wide.loc[1, "bureau_cnt"] == 2
    assert float(wide.loc[1, "bureau_credit_sum_total"]) == 300.0
    assert float(wide.loc[1, "bureau_debt_total"]) == 40.0
    assert wide.loc[1, "bureau_overdue_cnt"] == 1

    assert wide.loc[2, "bureau_cnt"] == 0
    assert pd.isna(wide.loc[2, "bureau_credit_sum_total"])
    assert pd.isna(wide.loc[2, "bureau_debt_total"])

    # A single record whose amounts are all unknown stays unknown, not zero.
    assert wide.loc[3, "bureau_cnt"] == 1
    assert pd.isna(wide.loc[3, "bureau_credit_sum_total"])
    assert pd.isna(wide.loc[3, "bureau_debt_total"])
    assert wide.loc[3, "bureau_overdue_cnt"] == 0

    stored = pd.read_sql(
        "SELECT load_id, contract_version, counts, linkage FROM load_batches",
        database,
    )
    assert stored["load_id"].tolist() == ["验收_第一个成功装载"]
    assert stored["contract_version"].iloc[0] == ing.CONTRACT_VERSION
    assert stored["counts"].iloc[0]["派生表行数"]["model_input"] == 3
    assert stored["linkage"].iloc[0]["无征信申请数"] == 1


def test_loader_does_not_use_library_multi_row_inserts(
    database, tmp_path, monkeypatch
):
    """The load path must be native COPY, not `DataFrame.to_sql`."""

    def forbidden(*args, **kwargs):
        raise AssertionError("装载路径不应调用 DataFrame.to_sql")

    monkeypatch.setattr(pd.DataFrame, "to_sql", forbidden, raising=False)

    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_不依赖多值插入", data_dir)

    assert pd.read_sql("SELECT count(*) AS n FROM bureau", database)["n"].iloc[0] == 4


def test_repeat_load_is_idempotent_but_keeps_history(database, tmp_path):
    data_dir = write_sample(tmp_path / "raw")

    run_load(database, "验收_重复装载_一", data_dir)
    first = table_state(database)

    run_load(database, "验收_重复装载_二", data_dir)
    second = table_state(database)

    assert_tables_unchanged(first, second)
    assert second["load_ids"] == ["验收_重复装载_一", "验收_重复装载_二"]
    assert second["failure_count"] == first["failure_count"] == 0


def test_changed_input_refreshes_every_table(database, tmp_path):
    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_变更装载_一", data_dir)

    applications = sample_applications()
    applications.loc[0, "AMT_INCOME_TOTAL"] = "123456"

    bureau = sample_bureau()
    bureau.loc[0, "AMT_CREDIT_SUM_DEBT"] = "999"
    bureau = bureau.drop(index=1)  # remove the second record of application 1

    write_sample(tmp_path / "raw", applications, bureau)
    run_load(database, "验收_变更装载_二", data_dir)

    assert pd.read_sql("SELECT count(*) AS n FROM bureau", database)["n"].iloc[0] == 3

    wide = pd.read_sql(
        """
        SELECT sk_id_curr, amt_income_total, bureau_cnt, bureau_credit_sum_total,
               bureau_debt_total
        FROM model_input ORDER BY sk_id_curr
        """,
        database,
    ).set_index("sk_id_curr")

    assert float(wide.loc[1, "amt_income_total"]) == 123456.0
    assert wide.loc[1, "bureau_cnt"] == 1
    assert float(wide.loc[1, "bureau_credit_sum_total"]) == 100.0
    assert float(wide.loc[1, "bureau_debt_total"]) == 999.0

    # The removed source record left no residue anywhere.
    assert 102 not in pd.read_sql("SELECT sk_id_bureau FROM bureau", database)[
        "sk_id_bureau"
    ].tolist()


def test_wide_source_file_is_projected_and_extra_columns_are_registered(
    database, tmp_path
):
    applications = sample_applications()
    for index in range(60):
        applications[f"EXTRA_COLUMN_{index}"] = f"值{index}"

    data_dir = write_sample(tmp_path / "raw", applications)
    record = run_load(database, "验收_超宽来源", data_dir)

    assert len(record["表头与列"][APPLICATION_FILE]["未装载列"]) == 60
    assert record["表头与列"][APPLICATION_FILE]["实际装载列"] == list(
        ing.contract_for(APPLICATION_FILE).columns
    )
    assert pd.read_sql("SELECT count(*) AS n FROM application_train", database)[
        "n"
    ].iloc[0] == 3
    columns = pd.read_sql(
        "SELECT * FROM application_train LIMIT 0", database
    ).columns.tolist()
    assert columns == list(ing.contract_for(APPLICATION_FILE).columns)


# --------------------------------------------------------------------------- #
# Failure matrix: everything must roll back and nothing may be published
# --------------------------------------------------------------------------- #
def _inject(monkeypatch, name: str, error: Exception):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(ing, name, boom)


def test_missing_input_file_changes_nothing(database, tmp_path):
    data_dir = write_sample(tmp_path / "raw")
    before = table_state(database)

    (data_dir / BUREAU_FILE).unlink()

    with pytest.raises(ing.LoadStageError):
        run_load(database, "验收_缺文件", data_dir)

    assert_nothing_published(before, table_state(database))
    assert last_failure(database)["stage"] == "契约与文件检查"


def test_missing_required_column_changes_nothing(database, tmp_path):
    applications = sample_applications().drop(columns=["AMT_CREDIT"])
    data_dir = write_sample(tmp_path / "raw", applications)
    before = table_state(database)

    with pytest.raises(ing.LoadStageError, match="缺少装载契约要求的列"):
        run_load(database, "验收_缺字段", data_dir)

    assert_nothing_published(before, table_state(database))
    assert last_failure(database)["stage"] == "暂存装载"


def test_normalised_duplicate_header_changes_nothing(database, tmp_path):
    applications = sample_applications()
    applications["TARGET"] = applications["TARGET"].astype(str)
    applications.rename(columns={"TARGET": "target"}, inplace=True)
    applications["TARGET"] = applications["target"].astype(str)
    data_dir = write_sample(tmp_path / "raw", applications)
    before = table_state(database)

    with pytest.raises(ing.LoadStageError, match="规范化后出现重复列名"):
        run_load(database, "验收_重复表头", data_dir)

    assert_nothing_published(before, table_state(database))


def test_invalid_value_in_a_later_chunk_is_not_published(database, tmp_path):
    """Rows before the bad one are staged but must never be published."""
    applications = sample_applications().astype({"SK_ID_CURR": "object"})
    applications.loc[2, "SK_ID_CURR"] = "不是数字"
    data_dir = write_sample(tmp_path / "raw", applications)

    run_load(database, "验收_基准装载", write_sample(tmp_path / "raw_ok"))
    before = table_state(database)

    with pytest.raises(Exception) as failure:
        run_load(database, "验收_后段非法", data_dir, chunk_rows=2)

    assert not isinstance(failure.value, AssertionError)
    assert_nothing_published(before, table_state(database))
    assert last_failure(database)["stage"] == "暂存装载"


def test_duplicate_key_across_chunks_is_rejected(database, tmp_path):
    applications = sample_applications().astype({"SK_ID_CURR": "object"})
    applications.loc[2, "SK_ID_CURR"] = applications.loc[0, "SK_ID_CURR"]
    data_dir = write_sample(tmp_path / "raw", applications)

    run_load(database, "验收_基准装载二", write_sample(tmp_path / "raw_ok"))
    before = table_state(database)

    with pytest.raises(Exception):
        run_load(database, "验收_跨块重复", data_dir, chunk_rows=2)

    assert_nothing_published(before, table_state(database))
    assert last_failure(database)["stage"] == "暂存装载"


def test_bureau_failure_after_main_staging_publishes_nothing(database, tmp_path):
    bureau = sample_bureau().astype({"SK_ID_BUREAU": "object"})
    bureau.loc[1, "SK_ID_BUREAU"] = "不是数字"
    data_dir = write_sample(tmp_path / "raw", bureau=bureau)

    run_load(database, "验收_基准装载三", write_sample(tmp_path / "raw_ok"))
    before = table_state(database)

    applications = sample_applications()
    applications.loc[0, "AMT_INCOME_TOTAL"] = "777777"
    write_sample(data_dir, applications, bureau)

    with pytest.raises(Exception):
        run_load(database, "验收_征信失败", data_dir)

    after = table_state(database)
    assert_nothing_published(before, after)
    # The new main-table values were staged but not published.
    assert float(after["application_train"].loc[0, "amt_income_total"]) == 100000.0


def test_failure_after_source_replacement_rolls_back_everything(
    database, tmp_path, monkeypatch
):
    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_基准装载四", data_dir)
    before = table_state(database)

    applications = sample_applications()
    applications.loc[0, "AMT_INCOME_TOTAL"] = "888888"
    write_sample(data_dir, applications)

    _inject(monkeypatch, "_run_derived_scripts", RuntimeError("注入派生刷新失败"))

    with pytest.raises(RuntimeError, match="注入派生刷新失败"):
        run_load(database, "验收_派生失败", data_dir)

    after = table_state(database)
    assert_nothing_published(before, after)
    assert float(after["application_train"].loc[0, "amt_income_total"]) == 100000.0
    assert last_failure(database)["stage"] == "派生表刷新"


def test_reconciliation_failure_leaves_no_success_record(
    database, tmp_path, monkeypatch
):
    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_基准装载五", data_dir)
    before = table_state(database)

    _inject(monkeypatch, "_reconcile", RuntimeError("注入对账失败"))

    with pytest.raises(RuntimeError, match="注入对账失败"):
        run_load(database, "验收_对账失败", data_dir)

    after = table_state(database)
    assert_nothing_published(before, after)
    assert after["load_ids"] == ["验收_基准装载五"]
    assert last_failure(database)["stage"] == "对账"


def test_duplicate_load_id_is_rejected_without_touching_the_tables(
    database, tmp_path
):
    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_重复编号", data_dir)
    before = table_state(database)

    with pytest.raises(ing.LoadStageError, match="装载编号已存在"):
        run_load(database, "验收_重复编号", data_dir)

    assert_nothing_published(before, table_state(database))


def test_failed_load_is_followed_by_a_successful_one(database, tmp_path):
    """A failed attempt must not poison the next load (no staging residue)."""
    applications = sample_applications().drop(columns=["AMT_ANNUITY"])
    broken = write_sample(tmp_path / "raw_broken", applications)
    good = write_sample(tmp_path / "raw_good")

    with pytest.raises(ing.LoadStageError):
        run_load(database, "验收_失败一次", broken)

    record = run_load(database, "验收_失败后成功", good)

    assert record["对账"]["未通过检查"] == []
    assert pd.read_sql("SELECT count(*) AS n FROM model_input", database)[
        "n"
    ].iloc[0] == 3

    staging_left = pd.read_sql(
        """
        SELECT count(*) AS n
        FROM information_schema.tables
        WHERE table_name IN ('stg_application_train', 'stg_bureau')
        """,
        database,
    )["n"].iloc[0]
    assert staging_left == 0


# --------------------------------------------------------------------------- #
# Concurrency and read snapshots
# --------------------------------------------------------------------------- #
def test_second_writer_times_out_without_mixing_versions(database, tmp_path):
    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_并发_基准", data_dir)
    before = table_state(database)

    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock():
        with database.connect() as connection:
            with connection.begin():
                connection.exec_driver_sql(
                    "SELECT pg_advisory_xact_lock(%s)", (ing.STEP2_LOCK_KEY,)
                )
                holding.set()
                release.wait(timeout=30)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    assert holding.wait(timeout=30), "写入锁没有被获取"

    try:
        with pytest.raises(ing.LoadStageError, match="等待管道写入锁超时"):
            run_load(database, "验收_并发_第二个", data_dir, lock_timeout_ms=500)
    finally:
        release.set()
        holder.join(timeout=30)

    assert_nothing_published(before, table_state(database))


def test_reader_sees_the_old_version_until_commit(
    database, tmp_path, monkeypatch
):
    data_dir = write_sample(tmp_path / "raw")
    run_load(database, "验收_快照_基准", data_dir)

    applications = sample_applications()
    applications.loc[0, "AMT_INCOME_TOTAL"] = "555555"
    write_sample(data_dir, applications)

    published = threading.Event()
    release = threading.Event()
    original = ing._run_derived_scripts

    def pause_after_publish(connection):
        published.set()
        release.wait(timeout=30)
        return original(connection)

    monkeypatch.setattr(ing, "_run_derived_scripts", pause_after_publish)

    result = {}

    def writer():
        result["record"] = run_load(database, "验收_快照_新版本", data_dir)

    worker = threading.Thread(target=writer)
    worker.start()
    assert published.wait(timeout=30), "写入事务没有到达发布点"

    try:
        # Still inside the writer's transaction: another connection sees the old row.
        during = pd.read_sql(
            "SELECT amt_income_total FROM model_input WHERE sk_id_curr = 1", database
        )["amt_income_total"].iloc[0]
        assert float(during) == 100000.0
    finally:
        release.set()
        worker.join(timeout=30)

    after = pd.read_sql(
        "SELECT amt_income_total FROM model_input WHERE sk_id_curr = 1", database
    )["amt_income_total"].iloc[0]
    assert float(after) == 555555.0
