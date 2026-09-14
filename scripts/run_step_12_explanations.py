"""Run the Step 12 internal explanation prototype.

    python scripts/run_step_12_explanations.py --model-table data/processed/model_table.csv

Both models are refitted with the frozen configuration on the same train split
and explained on a fixed random sample of validation applications. The sample is
drawn without using labels and without picking cases whose explanation looks
nice.

Only aggregate reports are written. Per-application contributions and reason
codes stay in memory for local review and are never published by this
repository; the sealed test split is not opened.
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

from src.data_layer.model_table import (
    check_split_ids,
    file_digest,
    load_model_table,
)
from src.data_layer.split_data import stratified_split
from src.explain.local import InternalModelExplainer
from src.models.configs import build_boosting_model, build_logistic_model


def aggregate_contributions(result) -> pd.DataFrame:
    """Mean signed and mean absolute group contribution for one model."""
    grouped = result.grouped_contributions

    return pd.DataFrame(
        {
            "平均有向贡献": grouped.mean(),
            "平均绝对贡献": grouped.abs().mean(),
        }
    ).sort_values("平均绝对贡献", ascending=False, kind="stable")


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
        default=Path("reports/step_12"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--explain-rows", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=3)
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

    # Labels are not consulted when the explanation sample is drawn.
    explain_raw = valid_raw.sample(
        n=min(arguments.explain_rows, len(valid_raw)),
        random_state=arguments.random_state,
    ).copy()

    explained = {
        "逻辑回归": InternalModelExplainer(
            logistic_model, max_rows=arguments.max_rows
        ).explain(explain_raw, top_k=arguments.top_k),
        "梯度提升树": InternalModelExplainer(
            boosting_model, max_rows=arguments.max_rows
        ).explain(explain_raw, top_k=arguments.top_k),
    }

    aggregate_report = pd.concat(
        {
            name: aggregate_contributions(result)
            for name, result in explained.items()
        },
        names=["model", "business_group"],
    )

    review_counts = {
        name: {
            "需要数据复核的申请数": int(
                result.diagnostics["data_review_required"].sum()
            ),
            "解释样本数": len(result.diagnostics),
            "含缺失标记的申请数": int(
                result.diagnostics["raw_missing_count"].gt(0).sum()
            ),
            "含特殊编码的申请数": int(
                result.diagnostics["sentinel_count"].gt(0).sum()
            ),
            "未知箱申请数": (
                int(result.diagnostics["unknown_selected_bins"].fillna(0).gt(0).sum())
                if result.metadata["模型类型"] == "逻辑回归"
                else None
            ),
        }
        for name, result in explained.items()
    }

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_report.to_csv(
        out_dir / "group_contribution_summary.csv", encoding="utf-8-sig"
    )

    protocol = {
        "解释样本选择": "固定随机种子抽取验证申请，不使用标签挑选案例",
        "样本数量": len(explain_raw),
        "随机种子": arguments.random_state,
        "是否使用最终测试集": False,
        "是否公开逐人原因码": False,
        "分组规则版本": "第一版",
        "模型解释口径": {
            name: result.metadata for name, result in explained.items()
        },
        "数据复核标记汇总": review_counts,
        "解释限制": [
            "聚合排序仅描述本次解释样本，不自动代表完整业务人群。",
            "平均绝对贡献不是业务收益，也不能直接跨模型比较。",
            "原因码仅为内部原型，未经过正式授信政策与合规审核。",
            "贡献是模型归因而非因果影响，逐人明细不出本机报告目录。",
        ],
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

    with (out_dir / "explanation_protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("业务组贡献聚合（按平均绝对贡献排序）：")
    print(aggregate_report)
    print("数据复核标记汇总（仅聚合）：")
    print(json.dumps(review_counts, ensure_ascii=False, indent=2))
    print("报告目录：", out_dir)


if __name__ == "__main__":
    main()
