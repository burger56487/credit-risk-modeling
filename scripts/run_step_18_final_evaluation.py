"""Step 18: one independent evaluation of the frozen scheme on the final test set.

This script opens the sealed test partition. It verifies the frozen artefacts by
digest, transforms (never re-fits) the test rows, applies the frozen threshold and
reports the pre-registered metrics. Re-running it is possible for reproduction,
but every opening is recorded, so a second opening is visible rather than silent.
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
from src.models.logistic import evaluate_probabilities


EPSILON = 1e-6


def log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def strategy_metrics(target: pd.Series, flagged: pd.Series) -> dict:
    positive = target.eq(1)
    return {
        "测试样本数": int(len(target)),
        "正类比例": float(positive.mean()),
        "标记复核比例": float(flagged.mean()),
        "被标记人群正类比例": float(target[flagged].mean()) if flagged.any() else None,
        "正类覆盖比例": float((flagged & positive).sum() / positive.sum()),
        "负类标记比例": float((flagged & ~positive).sum() / (~positive).sum()),
        "未标记正类数量": int((positive & ~flagged).sum()),
    }


def reliability_table(target: pd.Series, probability: pd.Series, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"target": target, "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(
        样本数=("target", "size"),
        平均预测概率=("probability", "mean"),
        实际正类比例=("target", "mean"),
    )
    lower = (frame.groupby("bin", observed=True)["probability"].min()).rename("区间下限")
    upper = (frame.groupby("bin", observed=True)["probability"].max()).rename("区间上限")
    return pd.concat([lower, upper, table], axis=1).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-table", type=Path, default=Path("data/processed/model_table_real.csv"))
    parser.add_argument("--membership", type=Path, default=Path("artifacts/partitions/real_v1/membership.csv"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("artifacts/experiments/real_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/final"))
    parser.add_argument("--local-dir", type=Path, default=Path("artifacts/final"))
    parser.add_argument("--openings-log", type=Path, default=Path("artifacts/final/test_set_openings.json"))
    arguments = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = json.loads((arguments.experiment_dir / "freeze_manifest.json").read_text(encoding="utf-8"))

    # 1. Verify the frozen artefacts before anything is opened.
    checks = {}
    checks["建模宽表摘要一致"] = file_digest(arguments.model_table) == manifest["数据版本"]["建模宽表摘要"]
    checks["成员清单摘要一致"] = (
        hashlib.sha256(arguments.membership.read_bytes()).hexdigest() == manifest["数据版本"]["成员清单摘要"]
    )
    for name, info in manifest["工件"].items():
        path = arguments.experiment_dir / info["文件"]
        checks[f"工件摘要一致：{name}"] = file_digest(path) == info["sha256"]
    if not all(checks.values()):
        raise SystemExit(f"冻结工件校验失败：{checks}")

    # 2. Read only the test partition, by frozen membership.
    table = pd.read_csv(arguments.model_table)
    membership = pd.read_csv(arguments.membership)
    table["sk_id_curr"] = table["sk_id_curr"].astype("int64")
    test_ids = set(membership.loc[membership["partition"].eq("测试"), "application_id"])
    other_ids = set(membership.loc[~membership["partition"].eq("测试"), "application_id"])
    if test_ids & other_ids:
        raise SystemExit("测试集与其他分区存在重叠")
    test_frame = table[table["sk_id_curr"].isin(test_ids)].set_index("sk_id_curr")

    # 3. Apply the frozen chain: transform only.
    import joblib

    chosen_model = manifest["选定方案"]["模型"]
    chosen_version = manifest["选定方案"]["概率版本"]
    model_key = "模型_基线逻辑回归" if chosen_model.startswith("基线") else "模型_主模型梯度提升树"
    cal_key = "校准器_基线逻辑回归" if chosen_model.startswith("基线") else "校准器_主模型梯度提升树"

    model = joblib.load(arguments.experiment_dir / manifest["工件"][model_key]["文件"])
    calibrator = joblib.load(arguments.experiment_dir / manifest["工件"][cal_key]["文件"])
    probabilities = {chosen_model: model.predict_bad_probability(test_frame)}
    if chosen_version == "校准概率":
        probabilities[chosen_model] = pd.Series(
            calibrator.predict_proba(log_odds(probabilities[chosen_model].to_numpy()).reshape(-1, 1))[:, 1],
            index=test_frame.index,
        )

    baseline_model = joblib.load(arguments.experiment_dir / manifest["工件"]["模型_基线逻辑回归"]["文件"])
    probabilities["基线逻辑回归"] = baseline_model.predict_bad_probability(test_frame)

    # 4. Pre-registered metrics and the frozen threshold (not re-derived).
    capability = pd.DataFrame(
        {name: evaluate_probabilities(test_frame["target"], series) for name, series in probabilities.items()}
    ).T

    threshold = float(manifest["策略"]["冻结阈值"])
    flagged = probabilities[chosen_model].ge(threshold)
    strategy = strategy_metrics(test_frame["target"], flagged)
    strategy["冻结阈值"] = threshold
    strategy["边界规则"] = manifest["策略"]["边界规则"]

    # 5. Write reports; the per-applicant file stays local.
    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    arguments.local_dir.mkdir(parents=True, exist_ok=True)

    capability.to_csv(out_dir / "final_model_capability.csv", encoding="utf-8-sig")
    pd.DataFrame([strategy]).T.rename(columns={0: "最终测试结果"}).to_csv(
        out_dir / "final_strategy.csv", encoding="utf-8-sig"
    )
    reliability = reliability_table(test_frame["target"], probabilities[chosen_model])
    reliability.to_csv(out_dir / "final_reliability.csv", index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame(
        {
            "sk_id_curr": test_frame.index,
            "target": test_frame["target"].to_numpy(),
            "predicted_probability": probabilities[chosen_model].to_numpy(),
            "flagged_for_review": flagged.to_numpy(),
        }
    )
    predictions_path = arguments.local_dir / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        axis.plot(reliability["平均预测概率"], reliability["实际正类比例"], marker="o", label=chosen_model)
        axis.set(xlabel="mean predicted probability", ylabel="observed positive rate",
                 title="Final test set reliability (deciles)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(out_dir / "final_reliability.png", dpi=160)
        plt.close(figure)
        figure_note = "final_reliability.png"
    except ImportError:
        figure_note = "未生成（缺少绘图库）"

    opening = {
        "打开时间": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "冻结清单摘要": hashlib.sha256(
            (arguments.experiment_dir / "freeze_manifest.json").read_bytes()
        ).hexdigest(),
        "测试样本数": len(test_frame),
        "预测文件摘要": file_digest(predictions_path),
        "说明": "最终测试集每次打开都会追加一条记录；如出现第 2 条，说明本集已被多次查看",
    }
    history = json.loads(arguments.openings_log.read_text(encoding="utf-8")) if arguments.openings_log.exists() else []
    history.append(opening)
    arguments.openings_log.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "冻结校验": checks,
        "选定方案": {"模型": chosen_model, "概率版本": chosen_version},
        "最终测试集": strategy,
        "模型能力": capability.round(6).to_dict(orient="index"),
        "校准图": figure_note,
        "逐申请预测（本地）": {"文件": predictions_path.name, "sha256": opening["预测文件摘要"]},
        "测试集打开次数": len(history),
        "环境": {"解释器": platform.python_version(), "耗时秒": round(time.perf_counter() - started, 1)},
    }
    (out_dir / "final_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    print("冻结校验:", checks)
    print("\n模型能力（最终测试集）:")
    print(capability.round(6).to_string())
    print("\n固定阈值策略（最终测试集）:")
    for key, value in strategy.items():
        print(f"   {key}: {round(value, 4) if isinstance(value, float) else value}")
    print("\n可靠度分箱:")
    print(reliability.round(4).to_string(index=False))
    print("\n校准图:", figure_note, "| 测试集打开次数:", len(history))
    print("报告目录:", out_dir, "| 本地预测:", predictions_path)


if __name__ == "__main__":
    main()
