"""Export the Step 15 serving artefact and verify the load round trip.

    python scripts/run_step_15_release.py --model-table data/processed/model_table.csv \
        --release 研究包第一版 --threshold 0.08

The script fits the frozen logistic configuration, packages the scorecard (which
owns the model snapshot), the score scale and the frozen policy into one
artefact, exports it to a brand-new directory and then reloads it using the
digest it just produced. The reload runs the fixed canary rows and compares them
with the offline predictions, and optionally issues one in-process API request to
show that the service and the offline path agree.

Re-exporting to the same directory fails on purpose: a new release needs a new
name. Any later change to src invalidates the package, which is the intended
behaviour and means "test, then release again".
"""
import argparse
import secrets
import sys
from pathlib import Path

import numpy as np
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
from src.serving.artifact import (
    ReleaseInfo,
    ServingBundle,
    canary_frame,
    export_bundle,
    load_bundle,
)
from src.strategy.approval import ApprovalRule


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-table",
        type=Path,
        default=Path("data/processed/model_table.csv"),
    )
    parser.add_argument(
        "--releases-dir",
        type=Path,
        default=Path("artifacts/releases"),
    )
    parser.add_argument("--release", default="研究包第一版")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="已冻结的审批阈值；不提供则发布为未配置策略。",
    )
    parser.add_argument("--base-score", type=float, default=600.0)
    parser.add_argument("--base-bad-good-odds", type=float, default=1.0 / 50.0)
    parser.add_argument("--points-to-double-odds", type=float, default=20.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--oot-size", type=float, default=0.2)
    parser.add_argument(
        "--skip-self-request",
        action="store_true",
        help="跳过进程内接口自检。",
    )
    arguments = parser.parse_args(argv)

    model_table = load_model_table(arguments.model_table)
    split = stratified_split(
        model_table,
        random_state=arguments.random_state,
        valid_size=arguments.valid_size,
        oot_size=arguments.oot_size,
    )
    check_split_ids(split)

    model = build_logistic_model().fit(
        split.X_train.copy(), split.y_train
    )

    rule = (
        None
        if arguments.threshold is None
        else ApprovalRule(mode="概率阈值", threshold=arguments.threshold)
    )

    bundle = ServingBundle(
        release=ReleaseInfo(
            artifact_id=arguments.release,
            model_version=(
                "固定逻辑回归第一版_"
                f"数据摘要_{file_digest(arguments.model_table)[:12]}"
            ),
            strategy_version=(
                "未配置" if rule is None else "开发候选规则第一版"
            ),
        ),
        scorecard=LogisticScorecard(
            model,
            ScoreScale(
                base_score=arguments.base_score,
                base_bad_good_odds=arguments.base_bad_good_odds,
                points_to_double_odds=arguments.points_to_double_odds,
            ),
        ),
        rule=rule,
    )

    release_directory = arguments.releases_dir / arguments.release
    manifest_digest = export_bundle(bundle, release_directory)

    # Reload exactly what was written, using the digest from this export.
    restored = load_bundle(release_directory, manifest_digest)

    offline = bundle.predict_frame(canary_frame())
    online = restored.predict_frame(canary_frame())

    comparison = pd.DataFrame(
        {
            "离线概率": offline["predicted_bad_probability"],
            "重载概率": online["predicted_bad_probability"],
        }
    )
    comparison["绝对差"] = (
        comparison["离线概率"] - comparison["重载概率"]
    ).abs()

    print("发布目录：", release_directory)
    print("待核验并配置的发布清单摘要：", manifest_digest)
    print("释放包内策略版本：", bundle.release.strategy_version)
    print("自检样本离线与重载对比：")
    print(comparison.to_string(index=False))
    print("最大绝对差：", float(comparison["绝对差"].max()))

    if not arguments.skip_self_request:
        from fastapi.testclient import TestClient

        from src.serving.api import create_app

        # Generated here and never printed or stored in the repository.
        token = secrets.token_urlsafe(32)
        app = create_app(release_directory, manifest_digest, token)

        application = canary_frame().to_dict("records")[0]
        application["application_id"] = "人工申请一"

        with TestClient(app) as client:
            health = client.get("/health")
            response = client.post(
                "/v1/score",
                json={
                    "request_id": "本地自检一",
                    "applications": [application],
                },
                headers={"Authorization": "Bearer " + token},
            )

        print("健康检查：", health.status_code, health.json())
        print("接口状态码：", response.status_code)

        payload = response.json()
        print("接口用途标记：", payload["purpose"])
        print("接口版本：", payload["artifact_id"], payload["model_version"])

        item = payload["results"][0]
        difference = abs(
            item["predicted_bad_probability"]
            - float(offline["predicted_bad_probability"].iloc[0])
        )
        print("接口与离线概率绝对差：", difference)
        print("模拟通过结果：", item["simulated_approval"])
        print("数据复核标记：", item["input_review_required"])
        print("未知箱数量：", item["unknown_selected_bins"])
        print("响应是否包含原始申请字段：", "amt_income_total" in response.text)

        if not np.isfinite(difference) or difference > 1e-9:
            raise RuntimeError("接口预测与离线预测不一致。")


if __name__ == "__main__":
    main()
