"""Step 14 tests: batch monitoring, review notices and append-only records."""
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from src.models.logistic import RiskLogisticModel
from src.monitoring.runner import (
    BatchMonitor,
    MonitoringConfig,
    load_runs,
    run_and_record,
    save_run,
    validate_batch,
)
from src.strategy.approval import ApprovalRule


@pytest.fixture
def sample():
    n = 80
    index = pd.Index(range(7000, 7000 + n))
    labels = np.arange(n) % 2
    external_score = np.where(labels == 1, 0.2, 0.8)

    X = pd.DataFrame(
        {
            "sk_id_curr": np.arange(n),
            "amt_income_total": 100000.0,
            "amt_credit": 200000.0,
            "amt_annuity": 10000.0,
            "days_birth": -10957.5,
            "days_employed": -3652.5,
            "ext_source_1": external_score,
            "ext_source_2": external_score.copy(),
            "ext_source_3": 0.5,
            "bureau_cnt": 1,
            "bureau_debt_total": 20000.0,
        },
        index=index,
    )
    y = pd.Series(labels, index=index)
    return X, y


@pytest.fixture
def monitor(sample):
    X, y = sample
    model = RiskLogisticModel().fit(X, y)

    return BatchMonitor(
        model,
        reference_raw=X,
        versions={
            "模型版本": "测试模型一",
            "参考分布版本": "测试参考一",
            "策略版本": "测试策略一",
            "监控配置版本": "测试监控一",
        },
        rule=ApprovalRule(threshold=0.5),
        config=MonitoringConfig(
            min_batch_size=20,
            min_label_class_count=2,
            explanation_sample_size=4,
        ),
    )


def test_unlabeled_batch_does_not_report_observed_outcomes(sample, monitor):
    X, _ = sample
    report = monitor.evaluate(X)

    assert report["有标签诊断"] is None
    assert report["策略监控"]["标签情景评价"] is None
    assert report["策略监控"]["模拟通过数"] is not None
    assert report["是否自动采取业务行动"] is False


def test_complete_research_labels_enable_supervised_diagnostics(sample, monitor):
    X, y = sample
    report = monitor.evaluate(X, y, label_scope="公开研究标签")

    assert report["有标签诊断"] is not None
    assert report["策略监控"]["标签情景评价"] is not None
    assert report["解释审计"]["抽查样本数"] == 4
    assert report["解释审计"]["是否覆盖整批"] is False
    assert report["有标签诊断"]["标签口径"] == "公开研究标签"


def test_partial_labels_are_not_filled_or_silently_filtered(sample, monitor):
    X, y = sample
    partial = y.astype("float64")
    partial.iloc[0] = np.nan

    with pytest.raises(ValueError, match="标签"):
        monitor.evaluate(X, partial, label_scope="公开研究标签")


def test_fake_maturity_label_scope_rejected(sample, monitor):
    X, y = sample

    with pytest.raises(ValueError, match="成熟判定"):
        monitor.evaluate(X, y, label_scope="真实业务已成熟")


def test_small_batch_is_not_marked_statistically_normal(sample, monitor):
    X, _ = sample
    report = monitor.evaluate(X.iloc[:5])

    assert report["分布提示是否具备最低样本量"] is False
    assert any(
        item["项目"] == "统计提示样本量" for item in report["复核提示"]
    )


def test_unknown_bins_and_invalid_values_generate_review(sample, monitor):
    X, _ = sample
    current = X.copy()
    current["ext_source_1"] = np.nan
    current["ext_source_2"] = np.nan
    current["amt_credit"] = -1.0

    report = monitor.evaluate(current)
    items = {item["项目"] for item in report["复核提示"]}

    assert "未知箱" in items
    assert "数据无效值" in items
    assert report["无效值申请数"] == len(X)
    assert report["最高保留变量未知箱比例"] == pytest.approx(1.0)
    assert report["是否自动采取业务行动"] is False


def test_labels_cannot_change_fixed_rule_decisions(sample, monitor):
    X, y = sample

    first = monitor.evaluate(X, y, label_scope="公开研究标签")
    second = monitor.evaluate(X, 1 - y, label_scope="公开研究标签")

    assert (
        first["策略监控"]["模拟通过数"]
        == second["策略监控"]["模拟通过数"]
    )
    assert (
        first["策略监控"]["模拟通过率"]
        == second["策略监控"]["模拟通过率"]
    )


def test_monitoring_does_not_update_reference(sample, monitor):
    X, _ = sample
    before = monitor.evaluate(X)["分布摘要"]

    changed = X.copy()
    changed["amt_income_total"] = 1.0
    monitor.evaluate(changed)

    after = monitor.evaluate(X)["分布摘要"]
    assert before == after


def test_duplicate_application_ids_rejected(sample, monitor):
    X, _ = sample
    duplicated = X.copy()
    duplicated["sk_id_curr"] = 1

    with pytest.raises(ValueError, match="编号"):
        monitor.evaluate(duplicated)


def test_duplicate_run_id_is_not_overwritten(tmp_path):
    database = tmp_path / "runs.sqlite3"
    first = {"运行编号": "运行一", "运行状态": "完成"}

    save_run(database, first)

    with pytest.raises(sqlite3.IntegrityError):
        save_run(
            database,
            {"运行编号": "运行一", "运行状态": "修改后的内容"},
        )

    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT payload FROM monitoring_runs WHERE record_id = ?",
            ("运行一",),
        ).fetchone()

    assert json.loads(row[0]) == first


def test_failed_computation_is_recorded_and_still_raises(
    sample, monitor, tmp_path
):
    X, y = sample
    partial = y.astype("float64")
    partial.iloc[0] = np.nan
    database = tmp_path / "runs.sqlite3"

    with pytest.raises(ValueError):
        run_and_record(
            monitor,
            X,
            database_path=database,
            run_id="失败运行一",
            batch_id="测试批次",
            y=partial,
            label_scope="公开研究标签",
        )

    with closing(sqlite3.connect(database)) as connection:
        payload = connection.execute(
            "SELECT payload FROM monitoring_runs"
        ).fetchone()[0]

    record = json.loads(payload)
    assert record["运行状态"] == "失败"
    assert "有标签诊断" not in record


def test_undefined_value_saved_as_null_but_infinity_rejected(tmp_path):
    database = tmp_path / "runs.sqlite3"

    save_run(database, {"运行编号": "空值记录", "未定义比例": np.nan})

    with closing(sqlite3.connect(database)) as connection:
        payload = connection.execute(
            "SELECT payload FROM monitoring_runs"
        ).fetchone()[0]

    assert json.loads(payload)["未定义比例"] is None

    with pytest.raises(ValueError, match="无穷"):
        save_run(database, {"运行编号": "异常记录", "指标": np.inf})


def test_input_is_not_modified(sample, monitor):
    X, y = sample
    before_X = X.copy(deep=True)
    before_y = y.copy(deep=True)

    monitor.evaluate(X, y, label_scope="公开研究标签")

    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(y, before_y)


def test_completed_run_records_status_and_notices(sample, monitor, tmp_path):
    X, y = sample
    database = tmp_path / "runs.sqlite3"

    record = run_and_record(
        monitor,
        X,
        database_path=database,
        run_id="完成运行一",
        batch_id="测试批次一",
        y=y,
        label_scope="公开研究标签",
    )

    assert record["运行状态"] == "完成"
    assert record["是否自动采取业务行动"] is False
    assert record["时间含义"] == "诊断执行时刻，不是申请日期"
    assert record["版本"]["模型版本"] == "测试模型一"

    stored = load_runs(database)
    assert list(stored) == ["完成运行一"]
    assert stored["完成运行一"]["运行状态"] == "完成"


def test_load_runs_is_read_only_and_reports_missing_records(tmp_path):
    with pytest.raises(FileNotFoundError, match="运行记录库"):
        load_runs(tmp_path / "absent.sqlite3")

    empty = tmp_path / "empty.sqlite3"
    with closing(sqlite3.connect(empty)) as connection:
        connection.execute("CREATE TABLE other (value TEXT)")
        connection.commit()

    with pytest.raises(ValueError, match="结构不完整"):
        load_runs(empty)

    database = tmp_path / "runs.sqlite3"
    save_run(database, {"运行编号": "运行甲", "运行状态": "完成"})
    save_run(database, {"运行编号": "运行乙", "运行状态": "失败"})

    runs = load_runs(database, limit=1)
    assert list(runs) == ["运行乙"]  # newest first


def test_monitor_configuration_and_batch_guards(sample, monitor):
    X, y = sample

    with pytest.raises(ValueError, match="正整数"):
        MonitoringConfig(min_batch_size=0)

    with pytest.raises(ValueError, match="分布复核阈值"):
        MonitoringConfig(psi_review_threshold=0.0)

    with pytest.raises(ValueError, match="不能超过一"):
        MonitoringConfig(calibration_gap_threshold=1.5)

    with pytest.raises(ValueError, match="版本"):
        BatchMonitor(
            RiskLogisticModel().fit(X, y),
            reference_raw=X,
            versions={"模型版本": "只填一个"},
        )

    with pytest.raises(ValueError, match="申请编号"):
        validate_batch(X.assign(sk_id_curr=np.nan))

    with pytest.raises(ValueError, match="标签口径"):
        monitor.evaluate(X, y, label_scope="  ")
