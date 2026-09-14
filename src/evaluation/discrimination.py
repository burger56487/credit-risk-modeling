"""Step 10: discrimination metrics and paired stratified resampling.

Scope of the uncertainty reported here: the validation split, with both models
frozen. Binning, encoding, feature list, parameters and predictions are all
fixed; only validation application records are resampled. The intervals
therefore do **not** include the uncertainty of re-splitting or re-training, and
they cannot undo the selection bias created by earlier looks at the
validation set.

Direction convention: a larger input value means the model thinks label 1 is
more likely. Raw probabilities are used, never credit scores (higher score means
lower risk) and never the rounded display score.

Metrics follow the Step 7 definitions: average precision is
``sum((recall_k - recall_{k-1}) * precision_k)``, not a trapezoidal area, so the
two places cannot drift apart.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.models.logistic import validate_binary_target


METRIC_NAMES = (
    "排序曲线下面积",
    "最大分布差异",
    "平均精确率",
)


def _finite_numeric_series(values: pd.Series, description: str) -> pd.Series:
    """Validate an indexed numeric series: no missing, complex or infinite."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{description}必须是带索引的序列。")
    if values.empty:
        raise ValueError(f"{description}不能为空。")
    if not values.index.is_unique:
        raise ValueError(f"{description}索引不能重复。")

    try:
        numeric = pd.to_numeric(values, errors="raise")

        if np.iscomplexobj(numeric.to_numpy()):
            raise ValueError("不接受复数。")

        result = numeric.astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description}包含无效数值。") from exc

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{description}必须全部为有限数值。")

    return result


def _validated_prediction(
    y: pd.Series,
    probability: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Check label/prediction alignment and return independent arrays."""
    values = _finite_numeric_series(probability, "预测概率")
    target = validate_binary_target(y, values.index)

    if ((values < 0) | (values > 1)).any():
        raise ValueError("预测概率必须位于零到一之间。")

    return (
        target.to_numpy(dtype="int64", copy=True),
        values.to_numpy(dtype="float64", copy=True),
    )


class _RankMetricCache:
    """Cache the fixed prediction order; callers validate the inputs."""

    def __init__(self, target: np.ndarray, probability: np.ndarray):
        order = np.argsort(-probability, kind="stable")
        sorted_probability = probability[order]

        # Last position of each group of equal predictions: tied scores must be
        # accumulated as a whole, never split by their row order.
        group_ends = np.r_[
            np.flatnonzero(
                sorted_probability[:-1] != sorted_probability[1:]
            ),
            len(sorted_probability) - 1,
        ]

        self.order = order
        self.sorted_target = target[order]
        self.group_ends = group_ends

    def evaluate(self, weights: np.ndarray) -> np.ndarray:
        """Compute the three ranking metrics from record counts or weights."""
        sorted_weights = weights[self.order]

        cumulative_bad = np.cumsum(
            sorted_weights * self.sorted_target
        )[self.group_ends]

        cumulative_good = np.cumsum(
            sorted_weights * (1 - self.sorted_target)
        )[self.group_ends]

        total_bad = cumulative_bad[-1]
        total_good = cumulative_good[-1]

        if total_bad <= 0 or total_good <= 0:
            raise ValueError("加权后必须同时保留两类样本。")

        true_positive_rate = np.r_[0.0, cumulative_bad / total_bad]
        false_positive_rate = np.r_[0.0, cumulative_good / total_good]

        # Trapezoidal integration; tied predictions are already grouped.
        auc = np.sum(
            np.diff(false_positive_rate)
            * (true_positive_rate[1:] + true_positive_rate[:-1])
            / 2.0
        )

        ks = np.max(np.abs(true_positive_rate - false_positive_rate))

        cumulative_total = cumulative_bad + cumulative_good
        precision = np.divide(
            cumulative_bad,
            cumulative_total,
            out=np.zeros_like(cumulative_bad, dtype="float64"),
            where=cumulative_total > 0,
        )

        # Average precision is recall-increment weighted, not a trapezoid.
        average_precision = np.sum(np.diff(true_positive_rate) * precision)

        return np.array([auc, ks, average_precision], dtype="float64")


def ranking_metrics(
    y: pd.Series,
    probability: pd.Series,
    sample_weight: pd.Series | None = None,
) -> dict:
    """Public metric interface, usable for weighted cross-checks."""
    target, values = _validated_prediction(y, probability)

    if sample_weight is None:
        weights = np.ones(len(target), dtype="float64")
    else:
        validated_weight = _finite_numeric_series(sample_weight, "样本权重")

        if not validated_weight.index.equals(y.index):
            raise ValueError("样本权重与标签索引或顺序不一致。")
        if validated_weight.lt(0).any():
            raise ValueError("样本权重不能为负数。")

        weights = validated_weight.to_numpy(dtype="float64", copy=True)

    result = _RankMetricCache(target, values).evaluate(weights)

    return {
        name: float(value) for name, value in zip(METRIC_NAMES, result)
    }


@dataclass
class ComparisonResult:
    """Comparison summary, resampling draws and the protocol used."""

    summary: pd.DataFrame
    draws: pd.DataFrame
    metadata: dict


def paired_stratified_bootstrap(
    y: pd.Series,
    baseline_probability: pd.Series,
    challenger_probability: pd.Series,
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> ComparisonResult:
    """Resample validation records in pairs, stratified by label.

    Difference direction: challenger minus baseline.
    Interval method: percentile interval of the paired differences.
    Neither model is re-trained and the sealed test split is not touched.
    """
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, Integral)
        or n_bootstrap < 2
    ):
        raise ValueError("重采样次数必须是大于等于二的整数。")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("置信水平必须严格位于零与一之间。")

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or not 0 <= random_state < 2**32
    ):
        raise ValueError("随机种子必须是规定范围内的非负整数。")

    target, baseline = _validated_prediction(y, baseline_probability)
    _, challenger = _validated_prediction(y, challenger_probability)

    good_positions = np.flatnonzero(target == 0)
    bad_positions = np.flatnonzero(target == 1)

    if min(len(good_positions), len(bad_positions)) < 2:
        raise ValueError(
            "每类至少需要两个样本；单样本类别无法支持有效重采样。"
        )

    baseline_cache = _RankMetricCache(target, baseline)
    challenger_cache = _RankMetricCache(target, challenger)

    unit_weights = np.ones(len(target), dtype="float64")
    baseline_point = baseline_cache.evaluate(unit_weights)
    challenger_point = challenger_cache.evaluate(unit_weights)
    difference_point = challenger_point - baseline_point

    repetitions = int(n_bootstrap)
    baseline_draws = np.empty(
        (repetitions, len(METRIC_NAMES)), dtype="float64"
    )
    challenger_draws = np.empty_like(baseline_draws)

    rng = np.random.default_rng(int(random_state))

    for iteration in range(repetitions):
        sampled_good = rng.choice(
            good_positions, size=len(good_positions), replace=True
        )
        sampled_bad = rng.choice(
            bad_positions, size=len(bad_positions), replace=True
        )

        sampled_positions = np.concatenate([sampled_good, sampled_bad])

        # One occurrence count drives both models, which is what makes the
        # resampling paired rather than two independent draws.
        weights = np.bincount(
            sampled_positions, minlength=len(target)
        ).astype("float64")

        baseline_draws[iteration] = baseline_cache.evaluate(weights)
        challenger_draws[iteration] = challenger_cache.evaluate(weights)

    difference_draws = challenger_draws - baseline_draws

    tail = (1.0 - float(confidence_level)) / 2.0
    quantiles = [tail, 1.0 - tail]

    baseline_interval = np.quantile(baseline_draws, quantiles, axis=0)
    challenger_interval = np.quantile(challenger_draws, quantiles, axis=0)
    difference_interval = np.quantile(difference_draws, quantiles, axis=0)

    summary = pd.DataFrame(
        {
            "基准值": baseline_point,
            "基准下界": baseline_interval[0],
            "基准上界": baseline_interval[1],
            "对照值": challenger_point,
            "对照下界": challenger_interval[0],
            "对照上界": challenger_interval[1],
            "差值": difference_point,
            "差值下界": difference_interval[0],
            "差值上界": difference_interval[1],
        },
        index=pd.Index(METRIC_NAMES, name="指标"),
    )

    draw_frames = []
    for group_name, values in (
        ("基准", baseline_draws),
        ("对照", challenger_draws),
        ("差值", difference_draws),
    ):
        draw_frames.append(
            pd.DataFrame(
                values,
                columns=[
                    f"{group_name}_{metric}" for metric in METRIC_NAMES
                ],
            )
        )

    quality_notes = []

    if repetitions < 1000:
        quality_notes.append(
            "重采样次数较少，建议仅用于调试；尾部分位数可能不稳定。"
        )

    if min(len(good_positions), len(bad_positions)) < 30:
        quality_notes.append(
            "至少一个类别样本较少，区间可靠性需要谨慎解释。"
        )

    metadata = {
        "样本数": len(target),
        "标签零样本数": int(len(good_positions)),
        "标签一样本数": int(len(bad_positions)),
        "重采样次数": repetitions,
        "置信水平": float(confidence_level),
        "随机种子": int(random_state),
        "差值方向": "对照模型减基准模型",
        "区间方法": "配对分层重采样的百分位区间",
        "模型是否重新训练": False,
        "标签比例是否固定": True,
        "独立性假设": "以申请记录作为近似独立的抽样单位",
        "用途": "开发阶段验证，不替代封存测试集的最终评估",
        "质量提示": quality_notes,
    }

    return ComparisonResult(
        summary=summary,
        draws=pd.concat(draw_frames, axis=1),
        metadata=metadata,
    )
