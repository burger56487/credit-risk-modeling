"""Run the Step 11 calibration and distribution-stability diagnostics.

    python scripts/run_step_11_diagnostics.py --model-table data/processed/model_table.csv

Three reports are produced from one shared split:

* probability calibration of both models on the validation split;
* drift of the derived business features, reference = train, current = validation;
* drift of the two models' own risk probabilities over the same two samples.

No calibrator is fitted, the reference binning is never re-fitted on current
data, and the sealed test split is not opened.
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
from src.evaluation.calibration_stability import (
    NumericPSIMonitor,
    calibration_diagnostics,
)
from src.features.engineering import build_business_features
from src.models.configs import build_boosting_model, build_logistic_model


def probabilities_for(model, train_raw: pd.DataFrame, valid_raw: pd.DataFrame):
    """Train and validation probabilities from one frozen model."""
    train_probability = model.predict_bad_probability(train_raw)
    valid_probability = model.predict_bad_probability(valid_raw)

    pd.testing.assert_index_equal(train_probability.index, train_raw.index)
    pd.testing.assert_index_equal(valid_probability.index, valid_raw.index)

    return train_probability, valid_probability


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
        default=Path("reports/step_11"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--psi-bins", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-bin-samples", type=int, default=30)
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

    logistic_train, logistic_valid = probabilities_for(
        logistic_model, train_raw, valid_raw
    )
    boosting_train, boosting_valid = probabilities_for(
        boosting_model, train_raw, valid_raw
    )

    # 1. Probability calibration on the validation split.
    calibration_reports = {
        "逻辑回归": calibration_diagnostics(
            split.y_valid,
            logistic_valid,
            n_bins=arguments.n_bins,
            confidence_level=arguments.confidence_level,
            min_bin_samples=arguments.min_bin_samples,
        ),
        "梯度提升树": calibration_diagnostics(
            split.y_valid,
            boosting_valid,
            n_bins=arguments.n_bins,
            confidence_level=arguments.confidence_level,
            min_bin_samples=arguments.min_bin_samples,
        ),
    }

    calibration_summary = pd.DataFrame(
        {
            name: report.summary
            for name, report in calibration_reports.items()
        }
    ).T

    # 2. Business-feature drift: train reference, validation current.
    feature_monitor = NumericPSIMonitor(
        n_bins=arguments.psi_bins, epsilon=arguments.epsilon
    ).fit(build_business_features(train_raw))

    feature_stability = feature_monitor.report(
        build_business_features(valid_raw)
    )

    # 3. Prediction drift, aligned by index before anything is combined.
    reference_predictions = pd.DataFrame(
        {
            "逻辑回归风险概率": logistic_train,
            "梯度提升树风险概率": boosting_train,
        }
    )
    current_predictions = pd.DataFrame(
        {
            "逻辑回归风险概率": logistic_valid,
            "梯度提升树风险概率": boosting_valid,
        }
    )

    prediction_monitor = NumericPSIMonitor(
        n_bins=arguments.psi_bins, epsilon=arguments.epsilon
    ).fit(reference_predictions)

    prediction_stability = prediction_monitor.report(current_predictions)

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_summary.to_csv(
        out_dir / "calibration_summary.csv", encoding="utf-8-sig"
    )
    pd.concat(
        {
            name: report.bins
            for name, report in calibration_reports.items()
        },
        names=["model", "bin"],
    ).to_csv(out_dir / "calibration_bins.csv", encoding="utf-8-sig")

    feature_stability.summary.to_csv(
        out_dir / "feature_stability_summary.csv", encoding="utf-8-sig"
    )
    feature_stability.bins.to_csv(
        out_dir / "feature_stability_bins.csv", encoding="utf-8-sig"
    )

    prediction_stability.summary.to_csv(
        out_dir / "prediction_stability_summary.csv", encoding="utf-8-sig"
    )
    prediction_stability.bins.to_csv(
        out_dir / "prediction_stability_bins.csv", encoding="utf-8-sig"
    )

    protocol = {
        "参考人群": "训练集",
        "当前人群": "随机留出验证集",
        "是否真实时间稳定性验证": False,
        "是否拟合概率校准器": False,
        "是否使用最终测试集": False,
        "校准配置": {
            name: report.settings
            for name, report in calibration_reports.items()
        },
        "特征稳定性配置": feature_stability.settings,
        "预测稳定性配置": prediction_stability.settings,
        "限制": [
            "训练预测属于回代结果，与验证预测的差异可能包含过拟合因素。",
            "稳定性指数不能单独判断模型性能或概率校准。",
            "箱内区间不包含模型重新训练的不确定性。",
            "公开标签不等同于已验证的固定期限监管违约定义。",
            "输入分布可较早监控，但概率校准与效果评估需要等表现标签成熟。",
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

    with (out_dir / "diagnostic_protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("验证集概率诊断：")
    print(calibration_summary)
    print("业务特征分布差异（前 10 行）：")
    print(feature_stability.summary.head(10))
    print("预测分布差异：")
    print(prediction_stability.summary)
    print("报告目录：", out_dir)


if __name__ == "__main__":
    main()
