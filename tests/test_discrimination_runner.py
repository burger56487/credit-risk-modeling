"""End-to-end test of the Step 10 paired-comparison runner."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.discrimination import METRIC_NAMES


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_10_discrimination.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_10_discrimination", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def step_10_output(runner, model_table_builder, tmp_path_factory):
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200).to_csv(table, index=False)

    out_dir = tmp_path_factory.mktemp("reports") / "step_10"
    runner.main(
        [
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
            "--n-bootstrap",
            "200",
        ]
    )

    return table, out_dir


def test_runner_writes_intervals_draws_and_protocol(step_10_output):
    _, out_dir = step_10_output

    assert {path.name for path in out_dir.iterdir()} == {
        "paired_metric_intervals.csv",
        "bootstrap_draws.csv",
        "validation_protocol.json",
    }


def test_summary_reports_baseline_challenger_and_difference(step_10_output):
    _, out_dir = step_10_output
    summary = pd.read_csv(
        out_dir / "paired_metric_intervals.csv", index_col=0
    )

    assert list(summary.index) == list(METRIC_NAMES)
    assert {
        "基准值",
        "基准下界",
        "基准上界",
        "对照值",
        "对照下界",
        "对照上界",
        "差值",
        "差值下界",
        "差值上界",
    } == set(summary.columns)

    for metric in METRIC_NAMES:
        row = summary.loc[metric]
        assert row["基准下界"] <= row["基准值"] <= row["基准上界"]
        assert row["对照下界"] <= row["对照值"] <= row["对照上界"]
        assert row["差值下界"] <= row["差值"] <= row["差值上界"]
        assert row["差值"] == pytest.approx(row["对照值"] - row["基准值"])

    auc_row = summary.loc["排序曲线下面积"]
    assert auc_row["基准值"] > 0.5


def test_draws_and_intervals_stay_consistent(step_10_output):
    _, out_dir = step_10_output
    summary = pd.read_csv(
        out_dir / "paired_metric_intervals.csv", index_col=0
    )
    draws = pd.read_csv(out_dir / "bootstrap_draws.csv")

    assert len(draws) == 200
    assert {f"差值_{metric}" for metric in METRIC_NAMES} <= set(draws.columns)

    for metric in METRIC_NAMES:
        quantiles = np.quantile(draws[f"差值_{metric}"], [0.025, 0.975])
        assert summary.loc[metric, "差值下界"] == pytest.approx(quantiles[0])
        assert summary.loc[metric, "差值上界"] == pytest.approx(quantiles[1])


def test_protocol_records_what_is_not_covered(step_10_output):
    table, out_dir = step_10_output
    protocol = json.loads(
        (out_dir / "validation_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["基准模型"] == "第八步固定配置的逻辑回归流程"
    assert protocol["对照模型"] == "第八步固定配置的梯度提升树流程"
    assert protocol["预测来源"] == "未舍入的风险概率"
    assert protocol["是否使用最终测试集"] is False
    assert protocol["是否包含重新训练的不确定性"] is False
    assert protocol["是否处理反复查看验证集的选择偏差"] is False
    assert protocol["多指标说明"] == "分别报告边际区间，不提供联合覆盖保证"
    assert protocol["模型是否重新训练"] is False
    assert protocol["划分方式"] == "stratified_holdout"
    assert protocol["输入数据"]["SHA256"]
    assert protocol["环境"]["梯度提升框架"]
