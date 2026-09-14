"""Tests for the Step 8 comparison runner and its input guards."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_08_comparison.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_08_comparison", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_model_table(path: Path, n: int = 900, seed: int = 5) -> Path:
    """Minimal table with the raw columns the runner requires."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.0, 1.0, size=n)
    leverage = rng.uniform(0.02, 0.5, size=n)
    logit = 1.0 - 5.0 * score + 3.0 * leverage
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype("int64")

    frame = pd.DataFrame(
        {
            "sk_id_curr": np.arange(5000, 5000 + n),
            "target": target,
            "amt_income_total": rng.uniform(50000, 200000, size=n),
            "amt_credit": rng.uniform(20000, 400000, size=n),
            "amt_annuity": rng.uniform(5000, 30000, size=n),
            "days_birth": -rng.integers(7000, 25000, size=n).astype("float64"),
            "days_employed": -rng.integers(0, 14000, size=n).astype("float64"),
            "ext_source_1": score,
            "ext_source_2": np.clip(score + rng.normal(0, 0.1, n), 0.01, 0.99),
            "ext_source_3": rng.uniform(0.0, 1.0, size=n),
            "bureau_cnt": rng.integers(0, 10, size=n).astype("float64"),
            "bureau_debt_total": rng.uniform(0, 300000, size=n),
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_missing_file_and_columns_rejected(runner, tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到建模表"):
        runner.load_model_table(tmp_path / "absent.csv")

    table = write_model_table(tmp_path / "partial.csv", n=40)
    partial = pd.read_csv(table).drop(columns=["amt_credit"])
    partial_path = tmp_path / "partial_dropped.csv"
    partial.to_csv(partial_path, index=False)

    with pytest.raises(ValueError, match="缺少字段"):
        runner.load_model_table(partial_path)


def test_duplicate_application_ids_rejected(runner, tmp_path):
    table = write_model_table(tmp_path / "table.csv", n=40)
    frame = pd.read_csv(table)
    frame.loc[1, "sk_id_curr"] = frame.loc[0, "sk_id_curr"]
    broken = tmp_path / "duplicate_ids.csv"
    frame.to_csv(broken, index=False)

    with pytest.raises(ValueError, match="申请编号"):
        runner.load_model_table(broken)


def test_overlapping_split_ids_rejected(runner):
    shared = pd.DataFrame({"sk_id_curr": [1, 2, 3]})
    split = SimpleNamespace(
        X_train=shared,
        X_valid=shared.iloc[:1],
    )

    with pytest.raises(ValueError, match="重复申请编号"):
        runner.check_split_ids(split)

    disjoint = SimpleNamespace(
        X_train=shared,
        X_valid=pd.DataFrame({"sk_id_curr": [4, 5]}),
    )
    runner.check_split_ids(disjoint)  # must not raise


def test_file_digest_matches_hashlib(runner, tmp_path):
    table = write_model_table(tmp_path / "digest.csv", n=30)

    expected = hashlib.sha256(table.read_bytes()).hexdigest()
    assert runner.file_digest(table) == expected


def test_main_writes_reports_and_metadata(runner, tmp_path, monkeypatch):
    table = write_model_table(tmp_path / "model_table.csv", n=900)
    out_dir = tmp_path / "step_08"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_step_08_comparison.py",
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
        ],
    )
    runner.main()

    expected_files = {
        "validation_comparison.csv",
        "training_diagnostics.csv",
        "logistic_feature_selection.csv",
        "logistic_coefficients.csv",
        "tree_feature_selection.csv",
        "tree_training_gain.csv",
        "experiment_metadata.json",
    }
    written = {path.name for path in out_dir.iterdir()}
    assert written == expected_files

    comparison = pd.read_csv(
        out_dir / "validation_comparison.csv", index_col=0
    )
    assert list(comparison.index) == [
        "逻辑回归流程",
        "梯度提升树流程",
        "训练比例常数基准",
    ]
    assert comparison.loc["训练比例常数基准", "排序曲线下面积"] == pytest.approx(0.5)

    metadata = json.loads(
        (out_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["是否使用最终测试集"] is False
    assert metadata["是否使用验证集提前停止"] is False
    assert metadata["树模型实际训练轮数"] >= 1
    assert metadata["输入数据"]["SHA256"] == runner.file_digest(table)
