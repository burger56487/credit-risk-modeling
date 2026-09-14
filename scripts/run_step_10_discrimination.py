"""Run the Step 10 paired comparison of the logistic and boosting pipelines.

    python scripts/run_step_10_discrimination.py --model-table data/processed/model_table.csv

Both models are refitted with the frozen Step 8 configuration on the same train
split, then their validation probabilities are compared with a paired
stratified bootstrap. The sealed test split is not opened.

Per-applicant predictions are deliberately not published, so the script
regenerates them instead of reading a stored file. Everything else (config,
split, seeds) is fixed and recorded.
"""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

# Running a file inside scripts/ puts scripts/ on the import path, not the
# project root, so the src package is added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_layer.model_table import (
    check_split_ids,
    file_digest,
    load_model_table,
)
from src.data_layer.split_data import stratified_split
from src.evaluation.discrimination import paired_stratified_bootstrap
from src.models.configs import build_boosting_model, build_logistic_model


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
        default=Path("reports/step_10"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    arguments = parser.parse_args(argv)

    model_table = load_model_table(arguments.model_table)
    split = stratified_split(
        model_table,
        random_state=arguments.random_state,
        valid_size=arguments.valid_size,
        test_size=arguments.test_size,
    )
    check_split_ids(split)

    train_raw = split.X_train.copy()
    valid_raw = split.X_valid.copy()

    logistic_model = build_logistic_model().fit(train_raw, split.y_train)
    boosting_model = build_boosting_model(
        random_state=arguments.random_state
    ).fit(train_raw, split.y_train)

    logistic_valid_probability = logistic_model.predict_bad_probability(
        valid_raw
    )
    boosting_valid_probability = boosting_model.predict_bad_probability(
        valid_raw
    )

    comparison = paired_stratified_bootstrap(
        y=split.y_valid,
        baseline_probability=logistic_valid_probability,
        challenger_probability=boosting_valid_probability,
        n_bootstrap=arguments.n_bootstrap,
        confidence_level=arguments.confidence_level,
        random_state=arguments.random_state,
    )

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison.summary.to_csv(
        out_dir / "paired_metric_intervals.csv", encoding="utf-8-sig"
    )
    comparison.draws.to_csv(
        out_dir / "bootstrap_draws.csv", index=False, encoding="utf-8-sig"
    )

    protocol = {
        **comparison.metadata,
        "基准模型": "第八步固定配置的逻辑回归流程",
        "对照模型": "第八步固定配置的梯度提升树流程",
        "预测来源": "未舍入的风险概率",
        "是否使用最终测试集": False,
        "是否包含重新训练的不确定性": False,
        "是否处理反复查看验证集的选择偏差": False,
        "多指标说明": "分别报告边际区间，不提供联合覆盖保证",
        "划分方式": "分层随机留出（非时间外）",
        "留出说明": "数据没有可靠的申请时间，因此这是分层随机留出，不是时间外验证。",
        "封存测试样本数": len(split.y_test),
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

    with (out_dir / "validation_protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("基准模型：逻辑回归流程")
    print("对照模型：梯度提升树流程")
    print("差值方向：梯度提升树减逻辑回归")
    print(comparison.summary)
    print("评估口径：")
    print(protocol)
    print("报告目录：", out_dir)


if __name__ == "__main__":
    main()
