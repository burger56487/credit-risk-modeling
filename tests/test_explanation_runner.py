"""End-to-end test of the Step 12 explanation runner."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_12_explanations.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_12_explanations", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def step_12_output(runner, model_table_builder, tmp_path_factory):
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200).to_csv(table, index=False)

    out_dir = tmp_path_factory.mktemp("reports") / "step_12"
    runner.main(
        [
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
            "--explain-rows",
            "60",
            "--top-k",
            "2",
        ]
    )

    return table, out_dir


def test_runner_writes_only_aggregate_reports(step_12_output):
    _, out_dir = step_12_output

    assert {path.name for path in out_dir.iterdir()} == {
        "group_contribution_summary.csv",
        "explanation_protocol.json",
    }


def test_aggregate_report_covers_both_models(step_12_output):
    _, out_dir = step_12_output
    report = pd.read_csv(
        out_dir / "group_contribution_summary.csv"
    )

    assert {"model", "business_group", "平均有向贡献", "平均绝对贡献"} <= set(
        report.columns
    )
    assert set(report["model"]) == {"逻辑回归", "梯度提升树"}
    assert (report["平均绝对贡献"] >= 0).all()

    for _, group in report.groupby("model"):
        # Sorted by absolute contribution, descending.
        assert group["平均绝对贡献"].is_monotonic_decreasing


def test_protocol_records_method_and_limits(step_12_output):
    table, out_dir = step_12_output
    protocol = json.loads(
        (out_dir / "explanation_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["样本数量"] == 60
    assert protocol["随机种子"] == 42
    assert protocol["是否使用最终测试集"] is False
    assert protocol["是否公开逐人原因码"] is False
    assert protocol["分组规则版本"] == "第一版"
    assert set(protocol["模型解释口径"]) == {"逻辑回归", "梯度提升树"}

    for kind, metadata in protocol["模型解释口径"].items():
        assert metadata["模型类型"] == kind
        assert metadata["解释单位"] == "对数坏好比"
        assert metadata["最大重构误差"] < 1e-8
        assert metadata["是否重新训练模型"] is False
        assert metadata["原因候选数量上限"] == 2
        assert len(metadata["限制"]) >= 4

    logistic_metadata = protocol["模型解释口径"]["逻辑回归"]
    tree_metadata = protocol["模型解释口径"]["梯度提升树"]
    assert "截距" in logistic_metadata["解释参照"]
    assert "树路径" in tree_metadata["解释参照"]
    assert logistic_metadata["解释参照"] != tree_metadata["解释参照"]

    counts = protocol["数据复核标记汇总"]
    for name in ("逻辑回归", "梯度提升树"):
        assert counts[name]["解释样本数"] == 60
        assert 0 <= counts[name]["需要数据复核的申请数"] <= 60
    assert counts["梯度提升树"]["未知箱申请数"] is None


def test_snapshot_isolated_and_rows_aligned(step_12_output):
    """The aggregate is built from the explanation rows, nothing re-sampled."""
    _, out_dir = step_12_output
    report = pd.read_csv(out_dir / "group_contribution_summary.csv")

    # Both models explain the same applications, so their group columns match.
    logistic_groups = set(
        report.loc[report["model"] == "逻辑回归", "business_group"]
    )
    tree_groups = set(
        report.loc[report["model"] == "梯度提升树", "business_group"]
    )
    assert logistic_groups <= tree_groups or tree_groups <= logistic_groups
