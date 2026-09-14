"""Step 2 (revised) tests for the staged loader's non-database parts.

The old single-file tests asserted a dangerous behaviour — emptying a table in
its own transaction and then publishing file by file — so they were replaced.

Everything that touches a database (staging, copy, publish, rollback,
concurrency, read snapshots) lives in ``tests/test_postgres_pipeline.py`` and
runs against a real PostgreSQL instance. In-memory databases are not used here:
they cannot demonstrate transactional behaviour, and the pipeline refuses them.
"""
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_layer import ingest_to_db as ing


def write_inputs(directory: Path, applications: pd.DataFrame, bureau: pd.DataFrame):
    applications.to_csv(directory / "application_train.csv", index=False)
    bureau.to_csv(directory / "bureau.csv", index=False)


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()

    write_inputs(
        directory,
        pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "TARGET": [0, 1],
                "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
                "CODE_GENDER": ["F", "M"],
                "AMT_INCOME_TOTAL": ["100000", "50000"],
                "AMT_CREDIT": ["200000", "100000"],
                "AMT_ANNUITY": ["10000", "5000"],
                "DAYS_BIRTH": ["-10000", "-12000"],
                "DAYS_EMPLOYED": ["-2000", "365243"],
                "EXT_SOURCE_1": ["0.7", ""],
                "EXT_SOURCE_2": ["0.6", "0.5"],
                "EXT_SOURCE_3": ["0.5", "0.4"],
                "EXTRA_APPLICATION_COLUMN": ["a", "b"],
            }
        ),
        pd.DataFrame(
            {
                "SK_ID_BUREAU": [101, 102],
                "SK_ID_CURR": [1, 1],
                "CREDIT_ACTIVE": ["Active", "Closed"],
                "DAYS_CREDIT": ["-10", "-30"],
                "AMT_CREDIT_SUM": ["100", "200"],
                "AMT_CREDIT_SUM_DEBT": ["40", ""],
                "CREDIT_DAY_OVERDUE": ["5", "0"],
                "CREDIT_TYPE": ["Consumer credit", "Car loan"],
            }
        ),
    )
    return directory


def test_snapshot_digests_the_bytes_that_will_be_loaded(data_dir, tmp_path):
    snapshot = ing.snapshot_inputs(
        data_dir=data_dir, work_dir=tmp_path / "snapshot"
    )

    try:
        assert [entry["文件"] for entry in snapshot.entries] == list(
            ing.required_files()
        )

        for entry in snapshot.entries:
            copied = snapshot.directory / entry["文件"]
            assert copied.read_bytes() == (data_dir / entry["文件"]).read_bytes()
            assert entry["字节数"] == copied.stat().st_size
            assert len(entry["sha256"]) == 64
    finally:
        ing.cleanup_snapshot(snapshot)

    assert not (tmp_path / "snapshot").exists()


def test_snapshot_is_a_copy_not_a_reference(data_dir, tmp_path):
    """Changing the source after the snapshot must not change the snapshot."""
    snapshot = ing.snapshot_inputs(
        data_dir=data_dir, work_dir=tmp_path / "snapshot"
    )

    try:
        original = (
            snapshot.directory / "bureau.csv"
        ).read_text(encoding="utf-8")
        (data_dir / "bureau.csv").write_text("SK_ID_BUREAU\n999\n", encoding="utf-8")

        assert (
            snapshot.directory / "bureau.csv"
        ).read_text(encoding="utf-8") == original
    finally:
        ing.cleanup_snapshot(snapshot)


def test_missing_input_file_is_a_named_stage_failure(data_dir, tmp_path):
    (data_dir / "bureau.csv").unlink()

    with pytest.raises(ing.LoadStageError) as failure:
        ing.snapshot_inputs(data_dir=data_dir, work_dir=tmp_path / "snapshot")

    assert failure.value.stage == "契约与文件检查"
    assert "找不到输入文件" in str(failure.value)


def test_missing_work_dir_is_cleaned_up_even_without_inputs(tmp_path):
    with pytest.raises(ing.LoadStageError):
        ing.snapshot_inputs(data_dir=tmp_path, work_dir=tmp_path / "snapshot")

    ing.cleanup_snapshot(None)


def test_contract_rejects_non_postgres_backend_before_any_work(tmp_path):
    engine = create_engine("sqlite:///:memory:")

    with pytest.raises(ing.LoadStageError) as failure:
        ing.load_pipeline(engine, "装载一", data_dir=tmp_path)

    assert failure.value.stage == "契约与文件检查"
    assert "只支持 PostgreSQL" in str(failure.value)


def test_load_id_and_chunk_guards(tmp_path):
    engine = create_engine("sqlite:///:memory:")

    with pytest.raises(ValueError, match="装载编号"):
        ing.load_pipeline(engine, "  ")

    with pytest.raises(ValueError, match="分块行数"):
        ing.load_pipeline(engine, "装载一", data_dir=tmp_path, chunk_rows=0)

    with pytest.raises(ValueError, match="锁等待超时"):
        ing.load_pipeline(engine, "装载一", data_dir=tmp_path, lock_timeout_ms=0)


def test_failure_code_never_includes_the_message():
    class FakeError(Exception):
        pgcode = "23505"

    assert ing.error_code(FakeError("原始值 12345")) == "23505"
    assert ing.error_code(ValueError("普通错误")) is None


def test_error_code_reads_the_wrapped_driver_exception():
    class Driver:
        pgcode = "22P02"

    class Wrapper(Exception):
        orig = Driver()

    assert ing.error_code(Wrapper("外层")) == "22P02"
