"""Run the Step 7 vs Step 8 comparison on one shared split.

Both pipelines are fitted on the same train split and scored on the same
validation split; the test split stays sealed.

    python scripts/run_step_08_comparison.py --model-table data/processed/model_table.csv

The model table must contain ``sk_id_curr``, ``target`` and the raw application
columns produced by the Step 2 SQL layer.
"""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

# Running a file inside scripts/ puts scripts/ on the import path, not the
# project root, so the src package is added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_layer.split_data import stratified_split
from src.data_layer.model_table import (
    check_split_ids,
    file_digest,
    load_model_table,
)
from src.models.configs import build_boosting_model, build_logistic_model
from src.models.logistic import evaluate_probabilities


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-table",
        type=Path,
        default=Path("data/processed/model_table.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/step_08"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--oot-size", type=float, default=0.2)
    arguments = parser.parse_args(argv)

    model_table = load_model_table(arguments.model_table)
    split = stratified_split(
        model_table,
        random_state=arguments.random_state,
        valid_size=arguments.valid_size,
        oot_size=arguments.oot_size,
    )
    check_split_ids(split)

    train_raw = split.X_train.copy()
    valid_raw = split.X_valid.copy()

    logistic_model = build_logistic_model()
    boosting_model = build_boosting_model(random_state=arguments.random_state)

    logistic_model.fit(train_raw, split.y_train)
    boosting_model.fit(train_raw, split.y_train)

    logistic_valid = logistic_model.predict_bad_probability(valid_raw)
    boosting_valid = boosting_model.predict_bad_probability(valid_raw)

    constant_valid = pd.Series(
        float(split.y_train.mean()),
        index=split.y_valid.index,
    )

    validation_comparison = pd.DataFrame(
        {
            "逻辑回归流程": evaluate_probabilities(split.y_valid, logistic_valid),
            "梯度提升树流程": evaluate_probabilities(split.y_valid, boosting_valid),
            "训练比例常数基准": evaluate_probabilities(
                split.y_valid, constant_valid
            ),
        }
    ).T

    training_diagnostics = pd.DataFrame(
        {
            "逻辑回归回代诊断": evaluate_probabilities(
                split.y_train,
                logistic_model.predict_bad_probability(train_raw),
            ),
            "梯度提升树回代诊断": evaluate_probabilities(
                split.y_train,
                boosting_model.predict_bad_probability(train_raw),
            ),
        }
    ).T

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logistic_model.selection_report_.to_csv(
        out_dir / "logistic_feature_selection.csv", encoding="utf-8-sig"
    )
    logistic_model.coefficient_report_.to_csv(
        out_dir / "logistic_coefficients.csv", encoding="utf-8-sig"
    )
    boosting_model.selection_report_.to_csv(
        out_dir / "tree_feature_selection.csv", encoding="utf-8-sig"
    )
    boosting_model.gain_importance_.to_csv(
        out_dir / "tree_training_gain.csv", encoding="utf-8-sig"
    )
    validation_comparison.to_csv(
        out_dir / "validation_comparison.csv", encoding="utf-8-sig"
    )
    training_diagnostics.to_csv(
        out_dir / "training_diagnostics.csv", encoding="utf-8-sig"
    )

    metadata = {
        "实验说明": "同一随机划分下的两条建模流程对照",
        "是否使用最终测试集": False,
        "是否使用验证集提前停止": False,
        "划分方式": split.meta["mode"],
        "划分说明": split.meta["limitation"],
        "逻辑回归配置": logistic_model.get_params(deep=False),
        "梯度提升树配置": boosting_model.get_params(deep=False),
        "训练样本数": boosting_model.training_samples_,
        "验证样本数": len(split.y_valid),
        "封存测试样本数": len(split.y_test),
        "逻辑回归保留变量数": len(logistic_model.selected_features_),
        "树模型保留变量数": len(boosting_model.selected_features_),
        "树模型实际训练轮数": boosting_model.actual_iterations_,
        "输入数据": {
            "文件": arguments.model_table.name,
            "字节数": arguments.model_table.stat().st_size,
            "SHA256": file_digest(arguments.model_table),
        },
        "环境": {
            "解释器版本": platform.python_version(),
            "操作系统": platform.platform(),
            "数值计算库": version("numpy"),
            "数据处理库": version("pandas"),
            "机器学习库": version("scikit-learn"),
            "梯度提升框架": version("lightgbm"),
        },
    }

    with (out_dir / "experiment_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("验证集对照：")
    print(validation_comparison)
    print("训练集回代诊断，不作为泛化结论：")
    print(training_diagnostics)
    print("树模型保留变量数：", len(boosting_model.selected_features_))
    print("树模型实际训练轮数：", boosting_model.actual_iterations_)
    print("报告目录：", out_dir)


if __name__ == "__main__":
    main()
