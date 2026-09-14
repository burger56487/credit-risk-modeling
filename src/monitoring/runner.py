"""Step 14: fixed-model batch monitoring, run records and review notices.

A run never re-trains a model, never re-selects a threshold and never executes a
credit decision. Review notices mean "someone should look", not "the model has
failed".

Labels are optional and are handled strictly:

* without labels, only input-side diagnostics are produced — never an observed
  risk rate, calibration error or labelled utility;
* partial labels are rejected for the whole batch rather than filled with zeros
  or silently reduced to the labelled subset;
* real label maturity is not implemented, so a scope claiming maturity is
  rejected.

The reference distribution is fixed at construction and is never updated by a
current batch. Run records are append-only: the same run id can never overwrite
an earlier result.
"""
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.evaluation.calibration_stability import (
    NumericPSIMonitor,
    calibration_diagnostics,
)
from src.explain.local import InternalModelExplainer
from src.features.engineering import build_business_features
from src.models.boosting import RiskBoostingModel
from src.models.logistic import RiskLogisticModel, evaluate_probabilities
from src.strategy.approval import (
    ApprovalRule,
    UtilityAssumptions,
    evaluate_fixed_rule,
)


@dataclass(frozen=True)
class MonitoringConfig:
    """Engineering screening limits for a research prototype.

    These are not statistical power guarantees and not industry thresholds.
    """

    min_batch_size: int = 200
    min_label_class_count: int = 20
    explanation_sample_size: int = 20
    psi_review_threshold: float = 0.2
    unknown_rate_threshold: float = 0.01
    calibration_gap_threshold: float = 0.03

    def __post_init__(self):
        for name in (
            "min_batch_size",
            "min_label_class_count",
            "explanation_sample_size",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name}必须是正整数。")
            object.__setattr__(self, name, int(value))

        for name in (
            "psi_review_threshold",
            "unknown_rate_threshold",
            "calibration_gap_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name}必须是有限非负数。")

            object.__setattr__(self, name, float(value))

        if self.psi_review_threshold <= 0:
            raise ValueError("分布复核阈值必须大于零。")

        if (
            self.unknown_rate_threshold > 1
            or self.calibration_gap_threshold > 1
        ):
            raise ValueError("比例类提示阈值不能超过一。")


def validate_batch(X: pd.DataFrame) -> None:
    """Check batch structure; the application id never enters the features."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("批次输入必须是数据表。")
    if X.empty:
        raise ValueError("当前批次不能为空。")
    if not X.columns.is_unique:
        raise ValueError("批次字段名不能重复。")
    if not X.index.is_unique:
        raise ValueError("批次行索引不能重复。")
    if "sk_id_curr" not in X.columns:
        raise ValueError("批次缺少申请编号。")

    ids = X["sk_id_curr"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("批次申请编号缺失或重复。")

    # Unique application records do not prove that applicants are independent.


def json_safe(value):
    """Convert undefined values to explicit nulls; infinities stay an error.

    A numerical fault must not be disguised as an ordinary missing value.
    """
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if not np.isfinite(value):
            raise ValueError("监控报告包含无穷值。")
    return value


class BatchMonitor:
    """Batch research monitor that never re-fits a model or a policy."""

    def __init__(
        self,
        model,
        reference_raw: pd.DataFrame,
        versions: dict,
        rule: ApprovalRule | None = None,
        assumptions: UtilityAssumptions | None = None,
        config: MonitoringConfig | None = None,
    ):
        if not isinstance(model, (RiskLogisticModel, RiskBoostingModel)):
            raise TypeError("监控器只支持本项目的两类模型流程。")

        check_is_fitted(model, ["estimator_", "selected_features_"])
        validate_batch(reference_raw)

        required_versions = {
            "模型版本",
            "参考分布版本",
            "策略版本",
            "监控配置版本",
        }
        if (
            not isinstance(versions, dict)
            or set(versions) != required_versions
            or not all(
                isinstance(value, str) and value.strip()
                for value in versions.values()
            )
        ):
            raise ValueError("必须完整登记非空的模型、参考、策略和监控版本。")

        if rule is not None and not isinstance(rule, ApprovalRule):
            raise TypeError("策略必须是固定审批规则，或明确不配置。")
        if assumptions is not None and not isinstance(
            assumptions, UtilityAssumptions
        ):
            raise TypeError("效用假设类型不正确。")
        if config is not None and not isinstance(config, MonitoringConfig):
            raise TypeError("监控配置类型不正确。")

        self._model = deepcopy(model)
        self._versions = dict(versions)
        self._rule = deepcopy(rule)
        self._assumptions = (
            UtilityAssumptions() if assumptions is None else assumptions
        )
        self._config = MonitoringConfig() if config is None else config

        reference_features = build_business_features(reference_raw)
        reference_probability = self._model.predict_bad_probability(
            reference_raw
        )

        pd.testing.assert_index_equal(
            reference_features.index,
            reference_probability.index,
        )

        reference_view = reference_features.copy()
        reference_view["model_probability"] = reference_probability

        # The reference binning is created once, here, and never re-fitted.
        self._distribution_monitor = NumericPSIMonitor(
            n_bins=5, epsilon=1e-6
        ).fit(reference_view)

        self._reference_size = len(reference_view)

        self._explainer = InternalModelExplainer(
            self._model,
            max_rows=self._config.explanation_sample_size,
        )

    def context(self) -> dict:
        return {
            "版本": dict(self._versions),
            "监控配置": asdict(self._config),
            "参考样本数": self._reference_size,
            "是否配置策略": self._rule is not None,
            "策略规则": (
                None if self._rule is None else asdict(self._rule)
            ),
        }

    def _invalid_and_sentinel_counts(self, X: pd.DataFrame) -> tuple[int, int]:
        """Applications carrying invalid values, and the special-code count."""
        features = build_business_features(X)
        invalid_columns = [col for col in features if col.endswith("_invalid")]
        sentinel_columns = [
            col for col in features if col.endswith("_sentinel")
        ]

        invalid_count = int(features[invalid_columns].any(axis=1).sum())
        sentinel_count = int(features[sentinel_columns].any(axis=1).sum())

        return invalid_count, sentinel_count

    def _highest_unknown_bin_rate(self, X: pd.DataFrame) -> float | None:
        """Highest unknown-bin rate over retained variables.

        This is the per-variable rate, not the share of applications with at
        least one unknown value. The tree branch does not use scorecard bins, so
        the rate is not applicable there and is reported as unknown.
        """
        if not isinstance(self._model, RiskLogisticModel):
            return None

        bins = self._model.binner_.transform(build_business_features(X))
        rates = []

        for col in self._model.selected_features_:
            known_bins = self._model.encoder_.mapping_[col]
            rates.append(float((~bins[col].isin(known_bins)).mean()))

        return max(rates) if rates else None

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_scope: str = "未提供",
    ) -> dict:
        """Run every diagnostic that the available inputs allow."""
        validate_batch(X)

        if not isinstance(label_scope, str) or not label_scope.strip():
            raise ValueError("标签口径必须是非空字符串。")
        if "成熟" in label_scope:
            raise ValueError(
                "本版未实现真实业务标签成熟判定，"
                "不能用标签口径声明成熟。"
            )
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("标签必须是带索引的序列。")

        notices = []

        probability = self._model.predict_bad_probability(X)

        current_features = build_business_features(X)
        current_view = current_features.copy()
        current_view["model_probability"] = probability

        stability = self._distribution_monitor.report(current_view)

        invalid_count, sentinel_count = self._invalid_and_sentinel_counts(X)

        if invalid_count > 0:
            notices.append(
                {
                    "等级": "需复核",
                    "项目": "数据无效值",
                    "说明": (
                        "批次中存在不符合字段取值约定的记录，"
                        "请检查数据来源与上游计算；"
                        "不自动删除申请，也不据此推断客户行为。"
                    ),
                }
            )

        if sentinel_count > 0:
            notices.append(
                {
                    "等级": "需复核",
                    "项目": "特殊编码",
                    "说明": (
                        "批次中存在特殊编码记录（例如就业天数的占位取值），"
                        "它是数据可用性问题，不是失业或行为异常的结论。"
                    ),
                }
            )

        unknown_max = self._highest_unknown_bin_rate(X)

        if unknown_max is None:
            notices.append(
                {
                    "等级": "信息",
                    "项目": "未知箱",
                    "说明": "当前模型不使用评分卡分箱，未知箱比例不适用。",
                }
            )
        elif unknown_max > self._config.unknown_rate_threshold:
            notices.append(
                {
                    "等级": "需复核",
                    "项目": "未知箱",
                    "说明": (
                        "至少一个保留变量的未知箱比例超过配置阈值，"
                        "请检查字段变化与模型覆盖范围。"
                    ),
                }
            )

        enough_samples = len(X) >= self._config.min_batch_size

        if not enough_samples:
            notices.append(
                {
                    "等级": "信息",
                    "项目": "统计提示样本量",
                    "说明": (
                        "当前样本量低于配置下限，分布与校准提示未判定；"
                        "未判定不等于正常。"
                    ),
                }
            )
        elif (
            float(stability.summary["稳定性指数"].max())
            > self._config.psi_review_threshold
        ):
            notices.append(
                {
                    "等级": "需复核",
                    "项目": "分布变化",
                    "说明": (
                        "至少一个变量的稳定性指数超过配置阈值，"
                        "请结合分箱明细与用户群体变化复核。"
                    ),
                }
            )

        # The audit sample uses no label and does not pick flattering cases.
        audit_sample = X.sample(
            n=min(len(X), self._config.explanation_sample_size),
            random_state=42,
        )
        explanation = self._explainer.explain(audit_sample)

        strategy = {
            "状态": "未配置" if self._rule is None else "固定规则模拟",
            "模拟通过数": None,
            "模拟通过率": None,
            "标签情景评价": None,
        }

        if self._rule is not None:
            decisions = self._rule.decide(probability)
            strategy["模拟通过数"] = int(decisions.sum())
            strategy["模拟通过率"] = float(decisions.mean())

        supervised = None

        if y is not None:
            # This rejects misaligned, partially missing and illegal labels; it
            # never filters incomplete records on the caller's behalf.
            calibration = calibration_diagnostics(
                y, probability, n_bins=10, min_bin_samples=30
            )

            ranking = None

            if y.nunique() == 2:
                ranking = evaluate_probabilities(y, probability)
            else:
                notices.append(
                    {
                        "等级": "信息",
                        "项目": "排序指标",
                        "说明": "当前标签只有一种类别，排序指标不计算。",
                    }
                )

            class_counts = y.value_counts().reindex([0, 1], fill_value=0)
            enough_labels = (
                enough_samples
                and class_counts.min() >= self._config.min_label_class_count
            )

            gap = abs(calibration.summary["整体预测偏差"])

            if not enough_labels:
                notices.append(
                    {
                        "等级": "信息",
                        "项目": "校准提示",
                        "说明": "有效样本或某类标签数量不足，概率偏移未判定。",
                    }
                )
            elif gap > self._config.calibration_gap_threshold:
                notices.append(
                    {
                        "等级": "需复核",
                        "项目": "整体概率偏移",
                        "说明": (
                            "平均预测概率与实际标签一比例的差距超过配置阈值；"
                            "需结合分箱明细与样本不确定性复核。"
                        ),
                    }
                )

            supervised = {
                "标签口径": label_scope,
                "排序与概率指标": ranking,
                "校准摘要": calibration.summary,
                "校准分箱": calibration.bins.reset_index().to_dict("records"),
            }

            if self._rule is not None:
                fixed_result = evaluate_fixed_rule(
                    y, probability, self._rule, self._assumptions
                )

                if int(fixed_result["approved_count"]) != strategy["模拟通过数"]:
                    raise RuntimeError("策略监控与固定规则计数不一致。")

                strategy["标签情景评价"] = fixed_result.to_dict()

        report = {
            "标签口径": label_scope,
            "当前样本数": len(X),
            "分布提示是否具备最低样本量": enough_samples,
            "无效值申请数": invalid_count,
            "特殊编码申请数": sentinel_count,
            "最高保留变量未知箱比例": unknown_max,
            "分布摘要": stability.summary.reset_index().to_dict("records"),
            "分布分箱明细": stability.bins.reset_index().to_dict("records"),
            "解释审计": {
                "抽查样本数": len(audit_sample),
                "最大重构误差": explanation.metadata["最大重构误差"],
                "是否覆盖整批": len(audit_sample) == len(X),
            },
            "策略监控": strategy,
            "有标签诊断": supervised,
            "复核提示": notices,
            "是否存在复核提示": any(
                item["等级"] == "需复核" for item in notices
            ),
            "是否自动采取业务行动": False,
        }

        return json_safe(report)


def save_run(database_path: Path, record: dict) -> None:
    """Append a run record in a transaction; duplicates are never overwritten."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    record_id = record.get("运行编号")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("运行编号必须是非空字符串。")

    payload = json.dumps(
        json_safe(record), ensure_ascii=False, allow_nan=False
    )

    # The connection context manages the transaction; the closing context
    # guarantees that the connection is released.
    with closing(sqlite3.connect(database_path, timeout=30)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_runs (
                    record_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                INSERT INTO monitoring_runs (record_id, payload)
                VALUES (?, ?)
                """,
                (record_id, payload),
            )


def load_runs(database_path: Path, limit: int = 200) -> dict[str, dict]:
    """Read recorded runs, newest first, opening the database read-only.

    The dashboard uses this so that rendering a page can never create or modify
    a record.
    """
    database_path = Path(database_path)

    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("读取条数上限必须是正整数。")

    if not database_path.is_file():
        raise FileNotFoundError(f"尚未发现运行记录库：{database_path}")

    uri = database_path.resolve().as_uri() + "?mode=ro"

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        try:
            rows = connection.execute(
                """
                SELECT record_id, payload
                FROM monitoring_runs
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise ValueError(
                "运行记录库结构不完整，缺少 monitoring_runs 表。"
            ) from exc

    return {
        record_id: json.loads(payload) for record_id, payload in rows
    }


def run_and_record(
    monitor: BatchMonitor,
    X: pd.DataFrame,
    database_path: Path,
    run_id: str,
    batch_id: str,
    y: pd.Series | None = None,
    label_scope: str = "未提供",
) -> dict:
    """Record a full run; on failure store bounded failure info and re-raise."""
    for name, value in (("运行编号", run_id), ("批次编号", batch_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}必须是非空字符串。")

    context = {
        "运行编号": run_id,
        "批次编号": batch_id,
        "记录结构版本": "第一版",
        "执行时刻": datetime.now(timezone.utc).isoformat(),
        "时间含义": "诊断执行时刻，不是申请日期",
        **monitor.context(),
    }

    try:
        result = monitor.evaluate(X, y=y, label_scope=label_scope)
    except Exception:
        failure = {
            **context,
            "运行状态": "失败",
            "错误说明": (
                "输入校验或诊断计算失败。"
                "请在受控运行环境查看异常，不在报告中保存原始数据。"
            ),
        }
        save_run(database_path, failure)
        raise

    record = {
        **context,
        "运行状态": "完成",
        **result,
    }

    # A failed write is a failure, never a silent "mostly succeeded".
    save_run(database_path, record)
    return record
