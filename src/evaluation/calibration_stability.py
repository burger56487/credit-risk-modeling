"""Step 11: probability calibration diagnostics and reference-binning drift checks.

Three different questions, deliberately kept apart:

* ranking — covered by Step 10;
* are the probabilities accurate — calibration diagnostics, this module;
* did the input or prediction distribution move — stability checks, this module.

Being well ranked does not imply well calibrated, and a stable distribution does
not imply a working model.

The stability index here compares two distributions on one shared binning. It is
not a test of the model, not a value-at-risk coverage test, and no empirical
threshold is treated as "the model has failed": the report shows the number and
where it comes from, and leaves the decision to a human process.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_is_fitted

from src.features.engineering import (
    TrainQuantileBinner,
    validate_numeric_frame,
)


def _positive_integer(value, description: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{description}必须是正整数。")
    return int(value)


def _calibration_inputs(
    y: pd.Series,
    probability: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate calibration inputs.

    Unlike the ranking metrics, calibration statistics still make sense when the
    current sample contains only one label class.
    """
    if not isinstance(y, pd.Series):
        raise TypeError("标签必须是带索引的序列。")
    if not isinstance(probability, pd.Series):
        raise TypeError("预测概率必须是带索引的序列。")
    if probability.empty:
        raise ValueError("预测输入不能为空。")
    if not probability.index.is_unique:
        raise ValueError("样本索引不能重复。")
    if not y.index.equals(probability.index):
        raise ValueError("标签与预测的索引或顺序不一致。")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("标签必须是没有缺失的零或一。")

    try:
        numeric = pd.to_numeric(probability, errors="raise")
        if np.iscomplexobj(numeric.to_numpy()):
            raise ValueError("不接受复数。")
        values = numeric.astype("float64").to_numpy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("概率包含无效数值。") from exc

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("概率必须位于零到一之间且全部有限。")

    return y.to_numpy(dtype="int64", copy=True), values.copy()


@dataclass
class CalibrationResult:
    summary: dict
    bins: pd.DataFrame
    settings: dict


def calibration_diagnostics(
    y: pd.Series,
    probability: pd.Series,
    *,
    n_bins: int = 10,
    confidence_level: float = 0.95,
    min_bin_samples: int = 30,
) -> CalibrationResult:
    """Fixed equal-width probability bins; no calibrator is fitted.

    Bin edges are chosen in advance and never from the validation labels. Bins
    that stay empty keep a missing observed rate instead of being recorded as
    zero risk, and small bins are flagged.
    """
    n_bins = _positive_integer(n_bins, "概率分箱数")
    min_bin_samples = _positive_integer(min_bin_samples, "小样本箱阈值")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("置信水平必须严格位于零与一之间。")

    z = float(norm.ppf((1.0 + confidence_level) / 2.0))
    if not np.isfinite(z):
        raise ValueError("置信水平过于接近端点，无法稳定计算区间。")

    target, values = _calibration_inputs(y, probability)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Right-closed intervals: a value equal to an inner edge joins the lower
    # bin, zero joins the first bin and one joins the last bin.
    positions = np.searchsorted(edges[1:-1], values, side="left")

    counts = np.bincount(positions, minlength=n_bins)
    bad_counts = np.bincount(positions, weights=target, minlength=n_bins)
    probability_sums = np.bincount(
        positions, weights=values, minlength=n_bins
    )

    populated = counts > 0
    mean_probability = np.full(n_bins, np.nan)
    observed_rate = np.full(n_bins, np.nan)

    np.divide(
        probability_sums, counts, out=mean_probability, where=populated
    )
    np.divide(bad_counts, counts, out=observed_rate, where=populated)

    lower = np.full(n_bins, np.nan)
    upper = np.full(n_bins, np.nan)

    n = counts[populated].astype("float64")
    rate = observed_rate[populated]
    denominator = 1.0 + z**2 / n

    center = (rate + z**2 / (2.0 * n)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / n + z**2 / (4.0 * n**2))
        / denominator
    )

    lower[populated] = np.maximum(0.0, center - half_width)
    upper[populated] = np.minimum(1.0, center + half_width)

    # Positive means the bin's mean prediction is above the observed rate.
    gap = mean_probability - observed_rate

    grouped_error = float(
        np.sum(counts[populated] / len(target) * np.abs(gap[populated]))
    )

    table = pd.DataFrame(
        {
            "lower_edge": edges[:-1],
            "upper_edge": edges[1:],
            "n_samples": counts,
            "n_bad": bad_counts.astype("int64"),
            "mean_probability": mean_probability,
            "observed_rate": observed_rate,
            "observed_lower": lower,
            "observed_upper": upper,
            "calibration_gap": gap,
            "empty_bin": ~populated,
            "small_bin": (populated & (counts < min_bin_samples)),
        },
        index=pd.Index(
            [f"概率箱_{i:02d}" for i in range(n_bins)], name="bin"
        ),
    )

    summary = {
        "样本数": len(target),
        "实际标签一比例": float(target.mean()),
        "平均预测概率": float(values.mean()),
        "整体预测偏差": float(values.mean() - target.mean()),
        "布里尔分数": float(np.mean((values - target) ** 2)),
        "对数损失": float(log_loss(target, values, labels=[0, 1])),
        "分箱加权绝对校准误差": grouped_error,
        "空箱数": int((~populated).sum()),
        "小样本箱数": int(table["small_bin"].sum()),
        "端点概率样本数": int(((values == 0) | (values == 1)).sum()),
    }

    settings = {
        "分箱数": n_bins,
        "分箱规则": "固定等宽区间，右闭；边界值归入左侧箱，零归入首箱，一归入末箱",
        "区间方法": "箱内实际比例的威尔逊区间",
        "置信水平": float(confidence_level),
        "小样本箱阈值": min_bin_samples,
        "是否重新拟合概率": False,
    }

    return CalibrationResult(
        summary=summary,
        bins=table,
        settings=settings,
    )


@dataclass
class StabilityResult:
    summary: pd.DataFrame
    bins: pd.DataFrame
    settings: dict


class NumericPSIMonitor(BaseEstimator):
    """Drift monitor on a fixed reference binning.

    Bin edges are fitted on reference data only and then applied unchanged;
    current data never re-fits them. Missing values always have their own bin
    and join the shared bin set, so a share comparison can never silently drop
    them.
    """

    def __init__(
        self,
        n_bins: int = 5,
        epsilon: float = 1e-6,
    ):
        self.n_bins = n_bins
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y=None):
        n_bins = _positive_integer(self.n_bins, "参考分箱数")
        if n_bins < 2:
            raise ValueError("参考分箱数必须至少为二。")

        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, Real)
            or not np.isfinite(self.epsilon)
            or not 0 < self.epsilon < 1
        ):
            raise ValueError("占比平滑参数必须严格位于零与一之间。")

        data = validate_numeric_frame(X)

        binner = TrainQuantileBinner(n_bins=n_bins)
        bins = binner.fit_transform(data)

        reference_counts = {}

        for col in data.columns:
            if binner.all_missing_[col]:
                labels = ["训练外非缺失箱", "缺失箱"]
            else:
                numeric_bin_count = len(binner.edges_[col]) + 1
                labels = [
                    f"数值箱_{i:03d}" for i in range(numeric_bin_count)
                ] + ["缺失箱"]

            counts = (
                bins[col]
                .value_counts()
                .reindex(labels, fill_value=0)
                .astype("int64")
            )

            if int(counts.sum()) != len(data):
                raise RuntimeError("参考分箱计数未覆盖全部样本。")

            reference_counts[col] = counts

        self.binner_ = binner
        self.reference_counts_ = reference_counts
        self.reference_size_ = len(data)
        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.n_features_in_ = data.shape[1]

        # Freeze the fitted configuration so a later change to the constructor
        # arguments cannot silently alter the report.
        self.epsilon_ = float(self.epsilon)
        self.n_bins_fitted_ = n_bins

        return self

    def report(self, X: pd.DataFrame) -> StabilityResult:
        check_is_fitted(
            self, ["binner_", "reference_counts_", "epsilon_"]
        )

        data = validate_numeric_frame(X)
        bins = self.binner_.transform(data)

        rows = []
        details = {}

        for col in self.feature_names_in_:
            reference_count = self.reference_counts_[col].copy()

            current_count = (
                bins[col]
                .value_counts()
                .reindex(reference_count.index, fill_value=0)
                .astype("int64")
            )

            if int(current_count.sum()) != len(data):
                raise RuntimeError("当前分箱计数未覆盖全部样本。")

            reference_share = reference_count / self.reference_size_
            current_share = current_count / len(data)

            k = len(reference_count)
            denominator = 1.0 + k * self.epsilon_

            # Share-level symmetric smoothing: identical proportions stay
            # identical even when the two samples differ in size.
            reference_smoothed = (
                reference_share + self.epsilon_
            ) / denominator
            current_smoothed = (current_share + self.epsilon_) / denominator

            contribution = (current_smoothed - reference_smoothed) * (
                np.log(current_smoothed) - np.log(reference_smoothed)
            )

            table = pd.DataFrame(
                {
                    "reference_count": reference_count,
                    "current_count": current_count,
                    "reference_share": reference_share,
                    "current_share": current_share,
                    "reference_smoothed": reference_smoothed,
                    "current_smoothed": current_smoothed,
                    "psi_component": contribution,
                }
            )
            details[col] = table

            reference_min, reference_max = self.binner_.training_ranges_[col]

            if reference_min is None:
                below_rate = np.nan
                above_rate = np.nan
            else:
                # The denominator is the whole current sample; missing values
                # are not counted as out of range.
                below_rate = float(data[col].lt(reference_min).mean())
                above_rate = float(data[col].gt(reference_max).mean())

            unseen_count = int(
                current_count.loc[reference_count.eq(0)].sum()
            )

            rows.append(
                {
                    "变量": col,
                    "稳定性指数": float(contribution.sum()),
                    "参考样本数": self.reference_size_,
                    "当前样本数": len(data),
                    "参考缺失率": float(reference_share.loc["缺失箱"]),
                    "当前缺失率": float(current_share.loc["缺失箱"]),
                    "新出现分箱占比": unseen_count / len(data),
                    "低于参考最小值占比": below_rate,
                    "高于参考最大值占比": above_rate,
                }
            )

        summary = (
            pd.DataFrame(rows)
            .set_index("变量")
            .sort_values("稳定性指数", ascending=False, kind="stable")
        )

        return StabilityResult(
            summary=summary,
            bins=pd.concat(details, names=["feature", "bin"]),
            settings={
                "参考分箱数配置": self.n_bins_fitted_,
                "占比平滑参数": self.epsilon_,
                "缺失是否单独成箱": True,
                "当前数据是否重新拟合边界": False,
                "参考样本数": self.reference_size_,
            },
        )
