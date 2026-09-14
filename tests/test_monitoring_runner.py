"""End-to-end test of the Step 14 monitoring runner."""
import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from src.monitoring.runner import load_runs


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_14_monitoring.py"
)
DASHBOARD_PATH = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "monitoring_dashboard.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_14_monitoring", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_runner(runner, table, database, *extra: str):
    runner.main(
        [
            "--model-table",
            str(table),
            "--database",
            str(database),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def monitored(runner, model_table_builder, tmp_path_factory):
    """One labeled run and one unlabeled run, with a fixed threshold."""
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200, separation=3.0).to_csv(table, index=False)

    database = tmp_path_factory.mktemp("artifacts") / "runs.sqlite3"
    run_runner(runner, table, database, "--threshold", "0.08")

    return table, database


def test_runner_appends_two_distinct_runs(monitored):
    _, database = monitored

    runs = load_runs(database)
    assert set(runs) == {
        "随机验证批次一_公开标签诊断_第一版",
        "随机验证批次一_无标签诊断_第一版",
    }

    unlabeled = runs["随机验证批次一_无标签诊断_第一版"]
    labeled = runs["随机验证批次一_公开标签诊断_第一版"]

    assert unlabeled["有标签诊断"] is None
    assert labeled["有标签诊断"] is not None
    assert labeled["标签口径"] == "公开研究标签"
    assert unlabeled["当前样本数"] == labeled["当前样本数"]

    for record in (unlabeled, labeled):
        assert record["运行状态"] == "完成"
        assert record["是否自动采取业务行动"] is False
        assert record["时间含义"] == "诊断执行时刻，不是申请日期"
        assert record["是否配置策略"] is True
        assert record["策略监控"]["模拟通过数"] is not None
        assert record["参考样本数"] > 0
        assert len(record["分布摘要"]) > 0
        assert record["解释审计"]["抽查样本数"] == 20
        assert record["解释审计"]["是否覆盖整批"] is False

    # The utility unit is an assumption, and the record says so.
    assert labeled["策略监控"]["标签情景评价"] is not None
    assert unlabeled["策略监控"]["标签情景评价"] is None


def test_without_threshold_the_strategy_module_is_skipped(
    runner, model_table_builder, tmp_path
):
    table = tmp_path / "table_no_rule.csv"
    model_table_builder(n=1200, separation=3.0).to_csv(table, index=False)
    database = tmp_path / "no_rule" / "runs.sqlite3"

    run_runner(runner, table, database)
    record = load_runs(database)["随机验证批次一_公开标签诊断_第一版"]

    assert record["是否配置策略"] is False
    assert record["策略规则"] is None
    assert record["策略监控"]["状态"] == "未配置"
    assert record["策略监控"]["模拟通过数"] is None


def test_duplicate_run_id_raises_instead_of_overwriting(monitored, runner):
    table, database = monitored

    with pytest.raises(sqlite3.IntegrityError):
        run_runner(runner, table, database, "--threshold", "0.08")

    # The original records are still there, unchanged.
    assert len(load_runs(database)) == 2


def test_small_batch_is_recorded_as_undetermined(
    runner, model_table_builder, tmp_path
):
    database = tmp_path / "small" / "runs.sqlite3"

    table = tmp_path / "small_table.csv"
    model_table_builder(n=60, separation=3.0).to_csv(table, index=False)

    run_runner(
        runner,
        table,
        database,
        "--min-batch-size",
        "200",
        "--model",
        "梯度提升树",
    )

    record = load_runs(database)["随机验证批次一_无标签诊断_第一版"]
    assert record["分布提示是否具备最低样本量"] is False
    assert any(
        item["项目"] == "统计提示样本量"
        for item in record["复核提示"]
    )
    # The tree branch has no scorecard bins, so the rate is not applicable.
    assert record["最高保留变量未知箱比例"] is None
    assert record["版本"]["模型版本"] == "梯度提升树固定配置第一版"


def test_dashboard_script_is_valid_python():
    """The dashboard needs streamlit to run, but it must at least parse."""
    source = DASHBOARD_PATH.read_text(encoding="utf-8")
    compile(source, str(DASHBOARD_PATH), "exec")

    # Read-only access lives in the tested loader, not in the view.
    assert "load_runs" in source
    assert "import sqlite3" not in source


def test_run_records_are_readable_without_writing(monitored):
    _, database = monitored

    with closing(sqlite3.connect(database)) as connection:
        before = connection.execute(
            "SELECT COUNT(*) FROM monitoring_runs"
        ).fetchone()[0]

    load_runs(database)
    load_runs(database)

    with closing(sqlite3.connect(database)) as connection:
        after = connection.execute(
            "SELECT COUNT(*) FROM monitoring_runs"
        ).fetchone()[0]

    assert before == after == 2
