"""End-to-end test of the Step 13 policy-simulation runner."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_13_policy_simulation.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_13_policy_simulation", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_runner(runner, table: Path, out_dir: Path, *extra: str):
    runner.main(
        [
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def relaxed_output(runner, model_table_builder, tmp_path_factory):
    """Constraints loose enough that both models yield a candidate rule."""
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200, separation=3.0).to_csv(table, index=False)

    out_dir = tmp_path_factory.mktemp("reports") / "step_13_relaxed"
    run_runner(
        runner,
        table,
        out_dir,
        "--min-approved-count",
        "20",
        "--min-approval-rate",
        "0.05",
        "--max-approved-event-rate",
        "0.6",
        "--max-threshold",
        "0.6",
    )

    return table, out_dir


@pytest.fixture(scope="module")
def infeasible_output(runner, model_table_builder, tmp_path_factory):
    """Constraints that no candidate can satisfy."""
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200, seed=9).to_csv(table, index=False)

    out_dir = tmp_path_factory.mktemp("reports") / "step_13_infeasible"
    run_runner(
        runner,
        table,
        out_dir,
        "--min-approved-count",
        "10000",
    )

    return table, out_dir


def test_relaxed_run_writes_every_report(relaxed_output):
    _, out_dir = relaxed_output

    assert {path.name for path in out_dir.iterdir()} == {
        "development_policy_curves.csv",
        "selection_status.csv",
        "selected_development_policies.csv",
        "fixed_policy_cost_sensitivity.csv",
        "strategy_protocol.json",
    }


def test_selected_policy_matches_its_curve_row(relaxed_output):
    _, out_dir = relaxed_output

    curves = pd.read_csv(
        out_dir / "development_policy_curves.csv", index_col=[0, 1]
    )
    selected = pd.read_csv(
        out_dir / "selected_development_policies.csv", index_col=0
    )

    assert set(selected.index) == {"逻辑回归", "梯度提升树"}
    assert {"threshold", "approved_count", "scenario_utility"} <= set(
        selected.columns
    )
    assert (selected["scenario_utility"] > 0).all()

    for model, row in selected.iterrows():
        curve = curves.xs(model, level="model")
        match = curve.loc[
            curve["rule_mode"].eq("概率阈值")
            & curve["threshold"].eq(row["threshold"])
        ]
        assert len(match) == 1
        assert match.iloc[0]["approved_count"] == row["approved_count"]
        assert match.iloc[0]["scenario_utility"] == pytest.approx(
            row["scenario_utility"]
        )


def test_cost_sensitivity_keeps_thresholds_and_counts(relaxed_output):
    _, out_dir = relaxed_output
    sensitivity = pd.read_csv(
        out_dir / "fixed_policy_cost_sensitivity.csv"
    )

    assert set(sensitivity["标签一情景损失"]) == {500, 1000, 2000}
    assert len(sensitivity) == 2 * 3

    for model, group in sensitivity.groupby("模型"):
        group = group.sort_values("标签一情景损失")
        # Same rule throughout: the threshold and pass rate cannot move.
        assert group["固定阈值"].nunique() == 1
        assert group["模拟通过率"].nunique() == 1
        assert group["入选标签一比例"].nunique() == 1

        # With approved events in the sample, a higher assumed loss must lower
        # the labelled utility; the pass rate itself cannot respond to cost.
        if group["入选标签一比例"].iloc[0] > 0:
            changes = group["标签情景净效用"].diff().dropna()
            assert (changes < 0).all()


def test_infeasible_run_reports_status_without_inventing_a_rule(
    infeasible_output,
):
    _, out_dir = infeasible_output

    written = {path.name for path in out_dir.iterdir()}
    assert "selection_status.csv" in written
    assert "development_policy_curves.csv" in written
    # No rule was found, so no selected-rule or sensitivity file is written.
    assert "selected_development_policies.csv" not in written
    assert "fixed_policy_cost_sensitivity.csv" not in written

    status = pd.read_csv(out_dir / "selection_status.csv")
    assert set(status["模型"]) == {"逻辑回归", "梯度提升树"}
    assert (status["状态"] == "无可行策略").all()
    assert status["说明"].str.contains("不自动放宽").all()


def test_protocol_states_utility_unit_and_limits(relaxed_output):
    table, out_dir = relaxed_output
    protocol = json.loads(
        (out_dir / "strategy_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["效用单位"] == "假设效用点，不是人民币或真实利润"
    assert protocol["效用假设"] == {
        "non_event_gain": 100.0,
        "event_loss": 1000.0,
        "approval_cost": 5.0,
    }
    assert protocol["规则边界"] == "预测概率小于等于阈值时模拟通过"
    assert protocol["额外基准"] == "全部拒绝"
    assert protocol["是否使用验证标签选择阈值"] is True
    assert protocol["是否使用最终测试集"] is False
    assert protocol["成本敏感性是否重新选择阈值"] is False
    assert protocol["是否导出逐人通过拒绝结果"] is False
    assert 1.0 in protocol["候选概率阈值"]
    assert len(protocol["限制"]) >= 5
    assert protocol["输入数据"]["SHA256"]
