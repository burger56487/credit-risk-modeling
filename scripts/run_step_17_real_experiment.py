"""Step 17: real-data training and validation on the frozen split.

Reads the exported modelling table and the frozen membership list, fits the
existing baseline (logistic scorecard) and main model (gradient boosting) on the
training partition only, fits a Platt calibrator per model on the calibration
half of the validation partition, compares four model/probability combinations on
the selection half, picks a scheme by the pre-registered rule and derives a
capacity-based review threshold. Nothing here looks at the final test partition.

The frozen artefacts, the decision and the threshold are written to
``freeze_manifest.json``. ``--smoke-rows`` runs the same code on a subsample for a
connectivity self-test and refuses to write the manifest.

Pre-registered decision rule (fixed before any result was seen):
  * base model by AUC, first by point estimate on the selection subset;
  * probability version by log loss;
  * ties or negligible differences are reported as "no material advantage".

Review capacity is a demo capacity: flag the riskiest 10% of applications for
manual review. It is not an optimal share discovered from the data.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_layer.model_table import file_digest
from src.models.configs import build_boosting_model, build_logistic_model
from src.models.logistic import evaluate_probabilities


EPSILON = 1e-6
PARTITIONS = ("训练", "验证", "测试")


def log_odds(probability: np.ndarray) -> np.ndarray:
    """Log-odds with an explicit clipping rule for the 0/1 endpoints."""
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def fit_platt(log_odds_values: np.ndarray, target: np.ndarray):
    """Platt scaling: fit a logistic regression on the log-odds only."""
    from sklearn.linear_model import LogisticRegression

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(log_odds_values.reshape(-1, 1), target)
    return calibrator


def apply_platt(calibrator, log_odds_values: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(log_odds_values.reshape(-1, 1))[:, 1]


def capacity_threshold(probability: pd.Series, capacity: float) -> dict:
    """Threshold for a fixed review capacity; ties are kept together."""
    order = probability.sort_values(ascending=False)
    count = max(1, int(np.ceil(capacity * len(order))))
    threshold = float(order.iloc[count - 1])
    flagged = probability.ge(threshold)

    return {
        "目标复核容量": float(capacity),
        "目标标记数": int(count),
        "实际标记数": int(flagged.sum()),
        "实际标记比例": float(flagged.mean()),
        "冻结阈值": threshold,
        "边界规则": "预测概率大于等于冻结阈值即标记",
    }


def strategy_metrics(target: pd.Series, flagged: pd.Series) -> dict:
    positive = target.eq(1)
    return {
        "样本数": int(len(target)),
        "正类比例": float(positive.mean()),
        "标记复核比例": float(flagged.mean()),
        "被标记人群正类比例": float(target[flagged].mean()) if flagged.any() else None,
        "正类覆盖比例": float((flagged & positive).sum() / positive.sum()),
        "负类标记比例": float((flagged & ~positive).sum() / (~positive).sum()),
        "未标记正类数量": int((positive & ~flagged).sum()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-table", type=Path, default=Path("data/processed/model_table_real.csv"))
    parser.add_argument("--membership", type=Path, default=Path("artifacts/partitions/real_v1/membership.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/experiments/real_v1"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-calibration-share", type=float, default=0.5)
    parser.add_argument("--review-capacity", type=float, default=0.10)
    parser.add_argument("--smoke-rows", type=int, default=0, help="仅连通性自测；不为真时不会写冻结清单")
    arguments = parser.parse_args(argv)

    timings = {}
    started = time.perf_counter()

    table = pd.read_csv(arguments.model_table)
    membership = pd.read_csv(arguments.membership)
    table["sk_id_curr"] = table["sk_id_curr"].astype("int64")
    membership["application_id"] = membership["application_id"].astype("int64")

    if set(table["sk_id_curr"]) != set(membership["application_id"]):
        raise SystemExit("宽表与成员清单的申请编号集合不一致")
    if not membership["application_id"].is_unique:
        raise SystemExit("成员清单存在重复申请")

    if arguments.smoke_rows:
        keep = membership.groupby("partition", group_keys=False).head(
            max(1, arguments.smoke_rows // len(PARTITIONS))
        )
        membership = keep
        table = table[table["sk_id_curr"].isin(set(membership["application_id"]))]

    frames = {}
    for name in PARTITIONS:
        ids = set(membership.loc[membership["partition"].eq(name), "application_id"])
        frames[name] = table[table["sk_id_curr"].isin(ids)].set_index("sk_id_curr")
        if frames[name].empty:
            raise SystemExit(f"分区 {name} 为空")

    print("分区规模:", {name: len(frame) for name, frame in frames.items()})
    timings["读取与成员装载"] = round(time.perf_counter() - started, 1)

    # 1. Split the validation partition into calibration and selection halves
    #    (stratified; the outer three-way split is untouched).
    step = time.perf_counter()
    train_frame, valid_frame, test_frame = frames["训练"], frames["验证"], frames["测试"]
    from sklearn.model_selection import train_test_split

    calibration_ids, selection_ids = train_test_split(
        valid_frame.index.to_numpy(),
        test_size=arguments.valid_calibration_share,
        stratify=valid_frame["target"].to_numpy(),
        random_state=arguments.random_state,
    )
    calibration = valid_frame.loc[calibration_ids]
    selection = valid_frame.loc[selection_ids]
    inner_membership = pd.DataFrame(
        {
            "application_id": list(calibration.index) + list(selection.index),
            "inner_partition": ["校准"] * len(calibration) + ["选择"] * len(selection),
        }
    ).sort_values("application_id")
    inner_digest = hashlib.sha256(
        inner_membership.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    print(f"验证集内部：校准 {len(calibration):,} / 选择 {len(selection):,}（摘要 {inner_digest[:12]}）")
    timings["验证集内部分区"] = round(time.perf_counter() - step, 1)

    # 2. Fit both models on the training partition only.
    models = {}
    step = time.perf_counter()
    logistic = build_logistic_model().fit(train_frame, train_frame["target"])
    timings["基线逻辑回归训练"] = round(time.perf_counter() - step, 1)
    models["基线逻辑回归"] = logistic

    step = time.perf_counter()
    boosting = build_boosting_model(random_state=arguments.random_state).fit(train_frame, train_frame["target"])
    timings["主模型梯度提升树训练"] = round(time.perf_counter() - step, 1)
    models["主模型梯度提升树"] = boosting

    # 3. Calibrate on the calibration half, compare on the selection half.
    step = time.perf_counter()
    raw = {}
    calibrated = {}
    calibrators = {}
    for name, model in models.items():
        raw[name] = {
            "校准": model.predict_bad_probability(calibration),
            "选择": model.predict_bad_probability(selection),
        }
        calibrator = fit_platt(log_odds(raw[name]["校准"].to_numpy()), calibration["target"].to_numpy())
        calibrators[name] = calibrator
        calibrated[name] = {
            "选择": pd.Series(
                apply_platt(calibrator, log_odds(raw[name]["选择"].to_numpy())), index=selection.index
            ),
        }
    timings["校准器拟合与选择集预测"] = round(time.perf_counter() - step, 1)

    rows = []
    for name in models:
        for version, series in (("原始概率", raw[name]["选择"]), ("校准概率", calibrated[name]["选择"])):
            metrics = evaluate_probabilities(selection["target"], series)
            rows.append(
                {
                    "模型与概率版本": f"{name}·{version}",
                    "模型": name,
                    "概率版本": version,
                    "排序曲线下面积": metrics["排序曲线下面积"],
                    "平均精确率": metrics["平均精确率"],
                    "对数损失": metrics["对数损失"],
                    "布里尔分数": metrics["布里尔分数"],
                }
            )
    comparison = pd.DataFrame(rows).set_index("模型与概率版本")
    print("\n选择子集对比：")
    print(comparison.round(6).to_string())

    # 4. Pre-registered selection: model by AUC, probability version by log loss.
    best_model = comparison.groupby("模型")["排序曲线下面积"].max().idxmax()
    subset = comparison.loc[comparison["模型"].eq(best_model)]
    best_version = subset["对数损失"].idxmin().split("·")[-1]
    auc_gap = float(subset["排序曲线下面积"].max() - subset["排序曲线下面积"].min())
    logloss_gap = float(subset["对数损失"].max() - subset["对数损失"].min())
    chosen_probability = (
        calibrated[best_model]["选择"] if best_version == "校准概率" else raw[best_model]["选择"]
    )

    selected_row = comparison.loc[f"{best_model}·{best_version}"]

    # 5. Capacity-based threshold on the selection subset.
    capacity = capacity_threshold(chosen_probability, arguments.review_capacity)
    flagged_selection = chosen_probability.ge(capacity["冻结阈值"])
    strategy_selection = strategy_metrics(selection["target"], flagged_selection)

    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "selection_comparison.csv", encoding="utf-8-sig")
    inner_membership.to_csv(out_dir / "validation_inner_membership.csv", index=False)

    import joblib

    artifacts = {}
    for name, model in models.items():
        path = out_dir / f"model_{'baseline' if name.startswith('基线') else 'main'}.joblib"
        joblib.dump(model, path)
        artifacts[f"模型_{name}"] = {"文件": path.name, "sha256": file_digest(path)}
    for name, calibrator in calibrators.items():
        path = out_dir / f"calibrator_{'baseline' if name.startswith('基线') else 'main'}.joblib"
        joblib.dump(calibrator, path)
        artifacts[f"校准器_{name}"] = {"文件": path.name, "sha256": file_digest(path)}

    manifest = {
        "状态": "烟雾自测，不作为结果" if arguments.smoke_rows else "本研究版本：方案与阈值已冻结，最终测试集尚未评价",
        "数据版本": {
            "建模宽表文件": arguments.model_table.name,
            "建模宽表摘要": file_digest(arguments.model_table),
            "成员清单摘要": hashlib.sha256(arguments.membership.read_bytes()).hexdigest(),
            "验证集内部成员摘要": inner_digest,
        },
        "特征范围": "现有契约：10 个原始字段经 build_business_features 派生；不含申请编号、标签、分区标记、性别与姓名类字段",
        "样本规模": {name: len(frame) for name, frame in frames.items()},
        "验证集内部规模": {"校准": len(calibration), "选择": len(selection)},
        "模型配置": {"基线": logistic.get_params(deep=False), "主模型": boosting.get_params(deep=False)},
        "概率处理": {"候选": ["原始概率", "校准概率"], "校准方法": "Platt（对 log-odds 做逻辑回归，端点按 eps=1e-6 裁剪）"},
        "选择结果": comparison.round(10).to_dict(orient="index"),
        "选定方案": {
            "模型": best_model,
            "概率版本": best_version,
            "依据": "模型按选择子集排序面积最大者；概率版本按对数损失最小者（事先登记规则）",
            "排序面积差": auc_gap,
            "对数损失差": logloss_gap,
            "是否宣称显著优势": bool(auc_gap > 0.005),
        },
        "策略": {**capacity, "选择子集表现": strategy_selection, "容量性质": "演示容量，非最优比例"},
        "选定方案的指标": {k: float(v) for k, v in selected_row.items() if isinstance(v, (int, float, np.floating))},
        "工件": artifacts,
        "阶段耗时秒": {**timings, "总计": round(time.perf_counter() - started, 1)},
        "环境": {
            "解释器": platform.python_version(),
            "数据库版本": "PostgreSQL 16.14（隔离研究库）",
            "说明": "本版本基于第三方上传副本的数据内容；官方来源字节一致性尚未验证",
        },
    }

    manifest_path = out_dir / "freeze_manifest.json" if not arguments.smoke_rows else out_dir / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n选定方案:", best_model, best_version)
    print("冻结阈值:", capacity["冻结阈值"], "| 目标容量", capacity["目标复核容量"], "| 实际标记比例", round(capacity["实际标记比例"], 4))
    print("选择子集策略表现:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in strategy_selection.items()})
    print("清单:", manifest_path)
    print("阶段耗时:", {**timings, "总计": round(time.perf_counter() - started, 1)})


if __name__ == "__main__":
    main()
