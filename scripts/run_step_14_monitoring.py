"""Run the Step 14 batch monitor and append its records.

    python scripts/run_step_14_monitoring.py --model-table data/processed/model_table.csv \
        --threshold 0.08

The reference distribution is the training split; the monitored batch is the
validation split, first without labels and then with the public research labels.
The tier is fixed: this script never re-fits a model and never searches for a
threshold. A threshold has to be supplied explicitly — normally the one frozen
by Step 13 — otherwise the strategy module is skipped rather than improvised.

Re-running with the same run id fails on purpose; new research needs a new id.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Running a file inside scripts/ puts scripts/ on the import path, not the
# project root, so the src package is added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_layer.model_table import (
    check_split_ids,
    file_digest,
    load_model_table,
)
from src.data_layer.split_data import stratified_split
from src.models.configs import build_boosting_model, build_logistic_model
from src.monitoring.runner import (
    BatchMonitor,
    MonitoringConfig,
    run_and_record,
)
from src.strategy.approval import ApprovalRule, UtilityAssumptions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-table",
        type=Path,
        default=Path("data/processed/model_table.csv"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts/monitoring/runs.sqlite3"),
    )
    parser.add_argument(
        "--model",
        choices=["逻辑回归", "梯度提升树"],
        default="逻辑回归",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="已冻结的审批阈值；不提供则跳过策略监控。",
    )
    parser.add_argument("--run-tag", default="第一版")
    parser.add_argument("--batch-id", default="随机验证批次一")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--oot-size", type=float, default=0.2)
    parser.add_argument("--min-batch-size", type=int, default=200)
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

    if arguments.model == "逻辑回归":
        model = build_logistic_model().fit(train_raw, split.y_train)
        model_version = "逻辑回归固定配置第一版"
    else:
        model = build_boosting_model(
            random_state=arguments.random_state
        ).fit(train_raw, split.y_train)
        model_version = "梯度提升树固定配置第一版"

    rule = (
        None
        if arguments.threshold is None
        else ApprovalRule(mode="概率阈值", threshold=arguments.threshold)
    )

    monitor = BatchMonitor(
        model=model,
        reference_raw=train_raw,
        versions={
            "模型版本": model_version,
            "参考分布版本": (
                f"训练样本参考_{arguments.run_tag}"
                f"_摘要_{file_digest(arguments.model_table)[:12]}"
            ),
            "策略版本": (
                "未配置" if rule is None else f"开发候选规则_{arguments.run_tag}"
            ),
            "监控配置版本": f"研究监控_{arguments.run_tag}",
        },
        rule=rule,
        assumptions=UtilityAssumptions(),
        config=MonitoringConfig(min_batch_size=arguments.min_batch_size),
    )

    # Without labels the run cannot produce an observed risk rate.
    unlabeled = run_and_record(
        monitor,
        valid_raw,
        database_path=arguments.database,
        run_id=f"{arguments.batch_id}_无标签诊断_{arguments.run_tag}",
        batch_id=arguments.batch_id,
    )

    # With the complete public research labels, supervised diagnostics run too.
    labeled = run_and_record(
        monitor,
        valid_raw,
        database_path=arguments.database,
        run_id=f"{arguments.batch_id}_公开标签诊断_{arguments.run_tag}",
        batch_id=arguments.batch_id,
        y=split.y_valid,
        label_scope="公开研究标签",
    )

    for label, record in (
        ("无标签诊断", unlabeled),
        ("公开标签诊断", labeled),
    ):
        print(f"【{label}】运行状态：", record["运行状态"])
        print("  复核提示：")
        print(pd.DataFrame(record["复核提示"]).to_string(index=False))
        print("  是否存在复核提示：", record["是否存在复核提示"])
        print("  是否自动采取业务行动：", record["是否自动采取业务行动"])

    print("运行记录库：", arguments.database)
    print("监督诊断：", "已计算" if labeled["有标签诊断"] else "未计算")


if __name__ == "__main__":
    main()
