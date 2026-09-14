"""Step 9: score scaling and per-variable score decomposition.

The score is a monotone transform of the model's log-odds, so it adds no
information. Two conventions matter and are easy to state wrongly:

* the quantity that doubles is the **bad-to-good odds**, not the probability;
* the anchor score (600 here) is a scale definition, not an approval cut-off.

Higher score means lower model-estimated risk, so ranking metrics computed from
scores must use the opposite direction from the probability metrics.

Decomposition uses the fitted logistic model:

    logit(p) = intercept + sum_j(coef_j * w_j)
    S = (offset - factor * intercept) + sum_j(-factor * coef_j * w_j)
        \\____________base points_____/   \\____variable points____/

Scoring never re-fits the binning or the encoding, never rounds a probability
before converting it, and never clips a zero or one probability into range.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.utils.validation import check_is_fitted

from src.models.logistic import RiskLogisticModel


def finite_series(values: pd.Series, description: str) -> pd.Series:
    """Check an indexed numeric series: no empties, duplicates or infinities."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{description}必须是带索引的序列。")
    if values.empty:
        raise ValueError(f"{description}不能为空。")
    if not values.index.is_unique:
        raise ValueError(f"{description}的索引不能重复。")

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


@dataclass(frozen=True)
class ScoreScale:
    """Immutable score-scale configuration.

    ``points_to_double_odds`` is the score drop for each doubling of the
    bad-to-good odds.
    """

    base_score: float = 600.0
    base_bad_good_odds: float = 1.0 / 50.0
    points_to_double_odds: float = 20.0

    factor: float = field(init=False)
    offset: float = field(init=False)

    def __post_init__(self):
        parameters = (
            ("基准分", self.base_score),
            ("基准坏好比", self.base_bad_good_odds),
            ("坏好比翻倍下降分数", self.points_to_double_odds),
        )

        for name, value in parameters:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name}必须是有限实数。")

            try:
                converted = float(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{name}超出可处理范围。") from exc

            if not np.isfinite(converted):
                raise ValueError(f"{name}必须是有限实数。")

        if self.base_bad_good_odds <= 0:
            raise ValueError("基准坏好比必须大于零。")
        if self.points_to_double_odds <= 0:
            raise ValueError("坏好比翻倍下降分数必须大于零。")

        with np.errstate(over="ignore", invalid="ignore"):
            factor = np.float64(self.points_to_double_odds) / np.log(2.0)
            offset = np.float64(self.base_score) + factor * np.log(
                np.float64(self.base_bad_good_odds)
            )

        if not np.isfinite([factor, offset]).all():
            raise ValueError("评分刻度计算溢出，请调整参数。")

        object.__setattr__(self, "factor", float(factor))
        object.__setattr__(self, "offset", float(offset))

    def scores_from_log_odds(self, log_odds: pd.Series) -> pd.Series:
        """Score from log bad-to-good odds; preferred for internal use.

        The linear output stays finite even where the probability has already
        saturated at zero or one.
        """
        values = finite_series(log_odds, "对数坏好比")

        with np.errstate(over="ignore", invalid="ignore"):
            scores = self.offset - self.factor * values

        return finite_series(scores, "原始分数").rename("score_raw")

    def scores_from_probabilities(self, probabilities: pd.Series) -> pd.Series:
        """Score from probabilities strictly inside (0, 1)."""
        values = finite_series(probabilities, "预测概率")

        if ((values <= 0) | (values >= 1)).any():
            raise ValueError(
                "概率转分数要求概率严格大于零且严格小于一，不自动裁剪端点。"
            )

        log_odds = np.log(values) - np.log1p(-values)
        return self.scores_from_log_odds(log_odds)

    def probabilities_from_scores(self, scores: pd.Series) -> pd.Series:
        """Invert the scale; extreme inputs may saturate in floating point."""
        values = finite_series(scores, "输入分数")

        with np.errstate(over="ignore", invalid="ignore"):
            log_odds = (self.offset - values) / self.factor

        log_odds = finite_series(log_odds, "反算对数坏好比")

        return pd.Series(
            expit(log_odds.to_numpy()),
            index=values.index,
            name="predicted_bad_probability",
        )


class LogisticScorecard:
    """Score view of a fitted logistic pipeline.

    No new parameter is fitted. The model is deep-copied so later edits to the
    source object cannot silently change the scorecard. That copy isolates the
    object; it is not version management for code, data, dependencies or
    released model artefacts.
    """

    def __init__(
        self,
        model: RiskLogisticModel,
        scale: ScoreScale | None = None,
    ):
        if not isinstance(model, RiskLogisticModel):
            raise TypeError("本评分卡只接受本项目的逻辑回归流程。")

        check_is_fitted(
            model,
            [
                "binner_",
                "encoder_",
                "selected_features_",
                "estimator_",
            ],
        )

        if scale is not None and not isinstance(scale, ScoreScale):
            raise TypeError("评分刻度必须使用指定的刻度配置对象。")

        self.scale = ScoreScale() if scale is None else scale
        self._model = deepcopy(model)

        estimator = self._model.estimator_
        selected = list(self._model.selected_features_)

        if not np.array_equal(estimator.classes_, np.array([0, 1])):
            raise ValueError("当前评分分解要求类别顺序为零、一。")

        if estimator.coef_.shape != (1, len(selected)):
            raise ValueError("回归系数数量与保留变量数量不一致。")

        if list(estimator.feature_names_in_) != selected:
            raise ValueError("模型变量顺序与评分变量顺序不一致。")

        if self._model.encoder_.unknown_policy != "neutral":
            raise ValueError("本版评分卡要求未知箱采用中性回退策略。")

        self._coefficients = pd.Series(
            estimator.coef_[0].copy(),
            index=selected,
            name="coefficient",
        )
        # Intercept and base points are public: the audit trail needs them.
        self.intercept = float(estimator.intercept_[0])

        with np.errstate(over="ignore", invalid="ignore"):
            base_points = self.scale.offset - self.scale.factor * self.intercept

        if (
            not np.isfinite(self._coefficients.to_numpy()).all()
            or not np.isfinite(base_points)
        ):
            raise ValueError("模型参数或基础分包含非有限值。")

        # The base score is the scale anchor plus the model intercept term; it
        # is not automatically equal to the anchor score.
        self.base_points = float(base_points)

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the linear output, model probability, raw and display score."""
        design = self._model.transform_features(X)
        estimator = self._model.estimator_

        log_odds = pd.Series(
            estimator.decision_function(design),
            index=design.index,
            name="log_odds",
        )

        scores = self.scale.scores_from_log_odds(log_odds)
        probabilities = estimator.predict_proba(design)[:, 1]

        return pd.DataFrame(
            {
                "log_odds": log_odds,
                "predicted_bad_probability": probabilities,
                "score_raw": scores,
                # Display only; a value exactly between integers rounds to even.
                "score_display": np.rint(scores.to_numpy()),
            },
            index=design.index,
        )

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Unrounded per-variable points, excluding the base points."""
        design = self._model.transform_features(X)

        with np.errstate(over="ignore", invalid="ignore"):
            points = design.mul(self._coefficients, axis="columns") * (
                -self.scale.factor
            )

        if not np.isfinite(points.to_numpy()).all():
            raise ValueError("分项得分出现非有限值。")

        return points

    def training_bin_table(self) -> pd.DataFrame:
        """Export the training bins and their per-bin points."""
        tables = {}

        for feature, coefficient in self._coefficients.items():
            table = self._model.encoder_.tables_[feature].copy(deep=True)
            table["coefficient"] = coefficient

            with np.errstate(over="ignore", invalid="ignore"):
                table["points"] = (
                    -self.scale.factor * coefficient * table["woe"]
                )

            if not np.isfinite(table["points"].to_numpy()).all():
                raise ValueError("分箱分值出现非有限值。")

            tables[feature] = table

        return pd.concat(tables, names=["feature", "bin"])

    def unknown_bin_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Unknown-bin report, flagging whether the variable enters scoring."""
        report = self._model.unknown_bin_report(X).copy()
        report["is_selected"] = report.index.isin(self._coefficients.index)
        return report
