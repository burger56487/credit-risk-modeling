"""End-to-end test of the Step 9 scorecard runner."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_09_scorecard.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_09_scorecard", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def step_09_output(runner, model_table_builder, tmp_path_factory):
    """Run the script once and share its output across the assertions."""
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200).to_csv(table, index=False)

    out_dir = tmp_path_factory.mktemp("reports") / "step_09"
    runner.main(
        [
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
        ]
    )

    return table, out_dir


def test_runner_writes_scorecard_reports(step_09_output):
    _, out_dir = step_09_output

    assert {path.name for path in out_dir.iterdir()} == {
        "scorecard_bins.csv",
        "validation_unknown_bins.csv",
        "validation_score_summary.csv",
        "scorecard_audit.json",
    }


def test_audit_records_the_scale_and_the_error_margins(step_09_output):
    _, out_dir = step_09_output
    audit = json.loads(
        (out_dir / "scorecard_audit.json").read_text(encoding="utf-8")
    )

    scale = audit["评分刻度"]
    assert scale["base_score"] == pytest.approx(600.0)
    assert scale["base_bad_good_odds"] == pytest.approx(1.0 / 50.0)
    assert scale["factor"] == pytest.approx(20.0 / np.log(2.0))
    assert scale["offset"] == pytest.approx(
        scale["base_score"] + scale["factor"] * np.log(scale["base_bad_good_odds"])
    )

    # The base points include the model intercept, so they are not the anchor.
    assert audit["基础分"] == pytest.approx(
        scale["offset"] - scale["factor"] * audit["模型截距"]
    )
    assert audit["基础分"] != pytest.approx(scale["base_score"])

    tolerances = audit["误差验收容差"]
    assert audit["最大分项重构误差"] <= tolerances["分项重构"]
    assert audit["最大概率反算误差"] <= tolerances["概率反算"]
    assert audit["是否使用最终测试集"] is False
    assert audit["是否用验证标签调整刻度"] is False
    assert audit["概率来源"] == "未加权、未经后续校准的逻辑回归"


def test_bin_table_and_summary_are_written(step_09_output):
    _, out_dir = step_09_output

    bins = pd.read_csv(out_dir / "scorecard_bins.csv")
    assert {"feature", "bin", "woe", "coefficient", "points"} <= set(bins.columns)
    assert (bins["points"].notna()).all()

    unknown = pd.read_csv(
        out_dir / "validation_unknown_bins.csv", index_col=0
    )
    assert {"n_samples", "unknown_count", "unknown_rate", "is_selected"} <= set(
        unknown.columns
    )
    assert bool(unknown["is_selected"].any())

    summary = pd.read_csv(
        out_dir / "validation_score_summary.csv", index_col=0
    )
    assert {"count", "mean", "min", "max"} <= set(summary.index)
    assert summary.loc["mean", "score_raw"] > summary.loc["min", "score_raw"]
