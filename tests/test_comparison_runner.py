"""Tests for the Step 8 comparison runner and the shared input guards."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_layer.model_table import (
    check_split_ids,
    file_digest,
    load_model_table,
)


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


def test_missing_file_and_columns_rejected(write_model_table):
    with pytest.raises(FileNotFoundError, match="找不到建模表"):
        load_model_table(Path("absent") / "absent.csv")

    table = write_model_table("partial_dropped.csv", n=40)
    partial = pd.read_csv(table).drop(columns=["amt_credit"])
    partial.to_csv(table, index=False)

    with pytest.raises(ValueError, match="缺少字段"):
        load_model_table(table)


def test_duplicate_application_ids_rejected(write_model_table):
    table = write_model_table("duplicate_ids.csv", n=40)
    frame = pd.read_csv(table)
    frame.loc[1, "sk_id_curr"] = frame.loc[0, "sk_id_curr"]
    frame.to_csv(table, index=False)

    with pytest.raises(ValueError, match="申请编号"):
        load_model_table(table)


def test_overlapping_split_ids_rejected():
    shared = pd.DataFrame({"sk_id_curr": [1, 2, 3]})

    with pytest.raises(ValueError, match="重复申请编号"):
        check_split_ids(
            SimpleNamespace(X_train=shared, X_valid=shared.iloc[:1])
        )

    check_split_ids(
        SimpleNamespace(
            X_train=shared,
            X_valid=pd.DataFrame({"sk_id_curr": [4, 5]}),
        )
    )  # must not raise


def test_file_digest_matches_hashlib(write_model_table):
    table = write_model_table("digest.csv", n=30)

    expected = hashlib.sha256(table.read_bytes()).hexdigest()
    assert file_digest(table) == expected


def test_main_writes_reports_and_metadata(
    runner, write_model_table, tmp_path
):
    table = write_model_table("model_table.csv", n=900)
    out_dir = tmp_path / "step_08"

    runner.main(
        [
            "--model-table",
            str(table),
            "--out-dir",
            str(out_dir),
        ]
    )

    expected_files = {
        "validation_comparison.csv",
        "training_diagnostics.csv",
        "logistic_feature_selection.csv",
        "logistic_coefficients.csv",
        "tree_feature_selection.csv",
        "tree_training_gain.csv",
        "experiment_metadata.json",
    }
    assert {path.name for path in out_dir.iterdir()} == expected_files

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
    assert metadata["输入数据"]["SHA256"] == file_digest(table)
