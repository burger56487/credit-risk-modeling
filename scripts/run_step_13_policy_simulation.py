"""Run the Step 13 offline approval-threshold simulation.

    python scripts/run_step_13_policy_simulation.py --model-table data/processed/model_table.csv

Both models are refitted with the frozen configuration on the same train split
and simulated on the same validation applications. The utility unit is an
assumed "utility point", not currency, and no real credit decision is made. The
sealed test split is not opened, and per-application decisions are not exported.

Cost sensitivity changes the loss assumption while keeping the selected
threshold fixed: re-searching a threshold per cost would make any rule look
robust.
"""
import argparse
import json
import platform
import sys
from dataclasses import asdict
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
from src.models.configs import build_boosting_model, build_logistic_model
from src.strategy.approval import (
    NoFeasiblePolicyError,
    PolicyRequirements,
    UtilityAssumptions,
    build_policy_curve,
    evaluate_fixed_rule,
    select_policy,
)


def parse_losses(text: str) -> list[float]:
    losses = [float(part) for part in text.split(",") if part.strip()]
    if not losses:
        raise ValueError("敏感性分析的损失假设不能为空。")
    return losses


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
        default=Path("reports/step_13"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--non-event-gain", type=float, default=100.0)
    parser.add_argument("--event-loss", type=float, default=1000.0)
    parser.add_argument("--approval-cost", type=float, default=5.0)
    parser.add_argument("--min-approval-rate", type=float, default=0.2)
    parser.add_argument("--max-approved-event-rate", type=float, default=0.06)
    parser.add_argument("--min-approved-count", type=int, default=100)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--max-threshold", type=float, default=0.30)
    parser.add_argument("--sensitivity-losses", default="500,1000,2000")
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

    model_probabilities = {
        "逻辑回归": logistic_model.predict_bad_probability(valid_raw),
        "梯度提升树": boosting_model.predict_bad_probability(valid_raw),
    }

    assumptions = UtilityAssumptions(
        non_event_gain=arguments.non_event_gain,
        event_loss=arguments.event_loss,
        approval_cost=arguments.approval_cost,
    )
    requirements = PolicyRequirements(
        min_approval_rate=arguments.min_approval_rate,
        max_approved_event_rate=arguments.max_approved_event_rate,
        min_approved_count=arguments.min_approved_count,
        require_positive_utility=True,
    )

    # The grid is registered here; it is not claimed to pre-date earlier looks
    # at the validation results.
    steps = int(round(arguments.max_threshold / arguments.threshold_step))
    thresholds = [
        index * arguments.threshold_step for index in range(steps + 1)
    ]

    policy_curves = {}
    selected_policies = {}
    selection_status = []

    for name, probability in model_probabilities.items():
        curve = build_policy_curve(
            split.y_valid, probability, thresholds, assumptions
        )
        policy_curves[name] = curve

        try:
            selected = select_policy(curve, requirements)
        except NoFeasiblePolicyError as exc:
            selection_status.append(
                {"模型": name, "状态": "无可行策略", "说明": str(exc)}
            )
            continue

        selected_policies[name] = selected
        selection_status.append(
            {
                "模型": name,
                "状态": "获得开发候选规则",
                "说明": "仍需冻结规则并进行后续独立评估",
            }
        )

        # Applying the rule reads probabilities only, never labels.
        decisions = selected.rule.decide(probability)

        if int(decisions.sum()) != int(
            selected.development_result["approved_count"]
        ):
            raise RuntimeError("规则应用结果与策略曲线计数不一致。")

    scenario_rows = []

    for name, selected in selected_policies.items():
        probability = model_probabilities[name]

        for event_loss in parse_losses(arguments.sensitivity_losses):
            changed = UtilityAssumptions(
                non_event_gain=arguments.non_event_gain,
                event_loss=event_loss,
                approval_cost=arguments.approval_cost,
            )

            result = evaluate_fixed_rule(
                split.y_valid, probability, selected.rule, changed
            )

            scenario_rows.append(
                {
                    "模型": name,
                    "固定阈值": selected.rule.threshold,
                    "标签一情景损失": event_loss,
                    "模拟通过率": result["approval_rate"],
                    "入选标签一比例": result["approved_event_rate"],
                    "标签情景净效用": result["scenario_utility"],
                    "预测情景净效用": result["predicted_scenario_utility"],
                }
            )

    sensitivity_report = pd.DataFrame(scenario_rows)

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(
        policy_curves, names=["model", "candidate"]
    ).to_csv(
        out_dir / "development_policy_curves.csv", encoding="utf-8-sig"
    )

    pd.DataFrame(selection_status).to_csv(
        out_dir / "selection_status.csv", index=False, encoding="utf-8-sig"
    )

    if selected_policies:
        pd.DataFrame(
            {
                name: selected.development_result
                for name, selected in selected_policies.items()
            }
        ).T.to_csv(
            out_dir / "selected_development_policies.csv",
            encoding="utf-8-sig",
        )

    if not sensitivity_report.empty:
        sensitivity_report.to_csv(
            out_dir / "fixed_policy_cost_sensitivity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    protocol = {
        "数据用途": "已标记公开样本上的开发阶段策略研究",
        "效用单位": "假设效用点，不是人民币或真实利润",
        "效用假设": asdict(assumptions),
        "样本约束": asdict(requirements),
        "候选概率阈值": sorted(set(thresholds + [1.0])),
        "额外基准": "全部拒绝",
        "规则边界": "预测概率小于等于阈值时模拟通过",
        "已选择规则": {
            name: asdict(selected.rule)
            for name, selected in selected_policies.items()
        },
        "是否使用验证标签选择阈值": True,
        "是否使用最终测试集": False,
        "成本敏感性是否重新选择阈值": False,
        "是否导出逐人通过拒绝结果": False,
        "模型配置": {
            "逻辑回归": logistic_model.get_params(deep=False),
            "梯度提升树": boosting_model.get_params(deep=False),
        },
        "验证样本数": len(split.y_valid),
        "验证实际标签一比例": float(split.y_valid.mean()),
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
        "限制": [
            "没有完整申请人群及原拒绝人群的反事实表现。",
            "样本标签不直接等于真实违约损失。",
            "所选规则的开发表现包含策略选择偏差。",
            "样本风险约束不构成未来人群的风险保证。",
            "尚未执行真实审批，也未完成正式合规评估。",
        ],
    }

    with (out_dir / "strategy_protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("选择状态：")
    print(pd.DataFrame(selection_status))
    for name, selected in selected_policies.items():
        print(f"{name} 候选阈值：", selected.rule.threshold)
        print(
            selected.development_result[
                [
                    "approved_count",
                    "approval_rate",
                    "approved_event_rate",
                    "scenario_utility",
                    "predicted_scenario_utility",
                ]
            ]
        )
    print("固定策略成本敏感性：")
    print(sensitivity_report)
    print("报告目录：", out_dir)


if __name__ == "__main__":
    main()
