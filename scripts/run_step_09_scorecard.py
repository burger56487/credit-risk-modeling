"""Run the Step 9 score scaling on the same holdout split as Step 8.

    python scripts/run_step_09_scorecard.py --model-table data/processed/model_table.csv

The script never opens the sealed test split and never uses validation labels to
adjust the scale. It checks three identities before writing any report:

    base points + sum(variable points) == raw score
    scale(probabilities_from_scores(raw score)) == model probability

Raw scores are used for both checks; the display score is a separate rounding
for presentation only.
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
from src.models.configs import build_logistic_model
from src.models.scorecard import LogisticScorecard, ScoreScale

# Engineering acceptance tolerances for the current scale. They are not a
# mathematical guarantee, and extreme parameters still need their own checks.
DECOMPOSITION_TOLERANCE = 1e-9
PROBABILITY_TOLERANCE = 1e-12


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
        default=Path("reports/step_09"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--oot-size", type=float, default=0.2)
    parser.add_argument("--base-score", type=float, default=600.0)
    parser.add_argument("--base-bad-good-odds", type=float, default=1.0 / 50.0)
    parser.add_argument("--points-to-double-odds", type=float, default=20.0)
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

    logistic_model = build_logistic_model().fit(train_raw, split.y_train)

    scale = ScoreScale(
        base_score=arguments.base_score,
        base_bad_good_odds=arguments.base_bad_good_odds,
        points_to_double_odds=arguments.points_to_double_odds,
    )
    scorecard = LogisticScorecard(model=logistic_model, scale=scale)

    valid_scores = scorecard.score(valid_raw)
    valid_parts = scorecard.contributions(valid_raw)

    reconstructed_scores = valid_parts.sum(axis=1) + scorecard.base_points
    decomposition_error = (
        reconstructed_scores - valid_scores["score_raw"]
    ).abs()

    if decomposition_error.max() > DECOMPOSITION_TOLERANCE:
        raise RuntimeError("总分与分项重构不一致，请检查实现。")

    recovered_probability = scale.probabilities_from_scores(
        valid_scores["score_raw"]
    )
    probability_error = (
        recovered_probability - valid_scores["predicted_bad_probability"]
    ).abs()

    if probability_error.max() > PROBABILITY_TOLERANCE:
        raise RuntimeError("分数反算概率与模型概率不一致。")

    bin_table = scorecard.training_bin_table()
    unknown_report = scorecard.unknown_bin_report(valid_raw)

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_table.to_csv(out_dir / "scorecard_bins.csv", encoding="utf-8-sig")
    unknown_report.to_csv(
        out_dir / "validation_unknown_bins.csv", encoding="utf-8-sig"
    )
    valid_scores["score_raw"].describe().to_csv(
        out_dir / "validation_score_summary.csv", encoding="utf-8-sig"
    )

    audit = {
        "评分刻度": asdict(scale),
        "基础分": scorecard.base_points,
        "模型截距": scorecard.intercept,
        "保留变量数": len(logistic_model.selected_features_),
        "保留变量": list(logistic_model.selected_features_),
        "逻辑回归配置": logistic_model.get_params(deep=False),
        "评分方向": "分数越高，模型估计风险越低",
        "概率来源": "未加权、未经后续校准的逻辑回归",
        "未知箱策略": "证据权重为零，分项得分为零",
        "展示舍入": "最接近整数，半分时取最接近偶数",
        "验收样本数": len(valid_scores),
        "最大分项重构误差": float(decomposition_error.max()),
        "最大概率反算误差": float(probability_error.max()),
        "误差验收容差": {
            "分项重构": DECOMPOSITION_TOLERANCE,
            "概率反算": PROBABILITY_TOLERANCE,
        },
        "是否使用最终测试集": False,
        "是否用验证标签调整刻度": False,
        "划分方式": split.meta["mode"],
        "留出说明": split.meta["limitation"],
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
        },
    }

    with (out_dir / "scorecard_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("基础分：", scorecard.base_points)
    print("最大分项重构误差：", float(decomposition_error.max()))
    print("最大概率反算误差：", float(probability_error.max()))
    print("总分描述统计：")
    print(valid_scores["score_raw"].describe())
    print("评分使用变量中的未知箱：")
    print(
        unknown_report.loc[
            unknown_report["is_selected"]
            & unknown_report["unknown_count"].gt(0)
        ]
    )
    print("报告目录：", out_dir)


if __name__ == "__main__":
    main()
