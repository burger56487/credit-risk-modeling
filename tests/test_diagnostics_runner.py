"""End-to-end test of the Step 11 calibration/stability runner."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_11_diagnostics.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_11_diagnostics", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def step_11_output(runner, model_table_builder, tmp_path_factory):
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200).to_csv(table, index=False)

    out_dir = tmp_path_factory.mktemp("reports") / "step_11"
    runner.main(
        [
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
            "--n-bins",
            "5",
        ]
    )

    return table, out_dir


def test_runner_writes_all_reports(step_11_output):
    _, out_dir = step_11_output

    assert {path.name for path in out_dir.iterdir()} == {
        "calibration_summary.csv",
        "calibration_bins.csv",
        "feature_stability_summary.csv",
        "feature_stability_bins.csv",
        "prediction_stability_summary.csv",
        "prediction_stability_bins.csv",
        "diagnostic_protocol.json",
    }


def test_calibration_summary_covers_both_models(step_11_output):
    _, out_dir = step_11_output
    summary = pd.read_csv(out_dir / "calibration_summary.csv", index_col=0)

    assert list(summary.index) == ["逻辑回归", "梯度提升树"]
    assert {
        "样本数",
        "实际标签一比例",
        "平均预测概率",
        "整体预测偏差",
        "布里尔分数",
        "对数损失",
        "分箱加权绝对校准误差",
        "空箱数",
        "小样本箱数",
        "端点概率样本数",
    } <= set(summary.columns)

    # Both models see the same validation sample.
    assert summary["样本数"].nunique() == 1
    assert summary["实际标签一比例"].nunique() == 1

    bins = pd.read_csv(out_dir / "calibration_bins.csv")
    assert {"model", "bin", "n_samples", "mean_probability", "observed_rate"} <= set(
        bins.columns
    )
    assert set(bins["model"]) == {"逻辑回归", "梯度提升树"}
    empty = bins.loc[bins["empty_bin"], "observed_rate"]
    assert empty.isna().all()


def test_stability_reports_have_expected_shape(step_11_output):
    _, out_dir = step_11_output

    features = pd.read_csv(
        out_dir / "feature_stability_summary.csv", index_col=0
    )
    assert {"稳定性指数", "参考缺失率", "当前缺失率", "新出现分箱占比"} <= set(
        features.columns
    )
    assert (features["稳定性指数"] >= 0).all()

    predictions = pd.read_csv(
        out_dir / "prediction_stability_summary.csv", index_col=0
    )
    # Rows are ordered by drift, so only the membership is fixed.
    assert set(predictions.index) == {"逻辑回归风险概率", "梯度提升树风险概率"}
    # Reference and current come from the same split, so drift stays small.
    assert (predictions["稳定性指数"] < 0.05).all()

    prediction_bins = pd.read_csv(out_dir / "prediction_stability_bins.csv")
    assert {"feature", "bin", "reference_share", "current_share"} <= set(
        prediction_bins.columns
    )


def test_protocol_states_the_limits(step_11_output):
    table, out_dir = step_11_output
    protocol = json.loads(
        (out_dir / "diagnostic_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["参考人群"] == "训练集"
    assert protocol["当前人群"] == "随机留出验证集"
    assert protocol["是否真实时间稳定性验证"] is False
    assert protocol["是否拟合概率校准器"] is False
    assert protocol["是否使用最终测试集"] is False
    assert protocol["特征稳定性配置"]["当前数据是否重新拟合边界"] is False
    assert protocol["预测稳定性配置"]["缺失是否单独成箱"] is True
    assert protocol["校准配置"]["逻辑回归"]["分箱数"] == 5
    assert protocol["校准配置"]["梯度提升树"]["是否重新拟合概率"] is False
    assert protocol["输入数据"]["SHA256"]
    assert len(protocol["限制"]) >= 4
