"""Step 7: end-to-end logistic-regression credit risk baseline.

Fitting learns the whole chain from raw application features:

    business rules -> train-only binning -> train-only WOE -> IV screen
    -> duplicate/constant removal -> regularised logistic regression

Prediction re-applies the fitted chain, so training and scoring cannot drift
apart. Wrapping the chain does not grant data permissions: passing validation
data to ``fit`` would still leak.
"""
import warnings
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.validation import check_is_fitted

from src.features.engineering import (
    TrainQuantileBinner,
    build_business_features,
)
from src.features.woe import WOEEncoder


def validate_binary_target(
    y: pd.Series,
    expected_index: pd.Index,
) -> pd.Series:
    """Check label values, classes and alignment with the feature index."""
    if not isinstance(y, pd.Series):
        raise TypeError("标签必须是带索引的序列。")
    if not y.index.is_unique:
        raise ValueError("标签索引不能重复。")
    if not y.index.equals(expected_index):
        raise ValueError("标签与输入的索引或顺序不一致。")
    if y.isna().any():
        raise ValueError("标签不能缺失。")
    if not y.isin([0, 1]).all():
        raise ValueError("标签只能取零或一。")
    if y.nunique() != 2:
        raise ValueError("当前训练或评估数据必须同时包含两类标签。")

    return y.astype("int64")


class RiskLogisticModel(ClassifierMixin, BaseEstimator):
    """Baseline model from raw application features to a bad-probability score.

    ``fit`` learns the bin edges, the WOE mapping, the candidate feature list
    and the regression parameters. ``predict`` applies those fitted objects
    without re-learning anything.
    """

    def __init__(
        self,
        n_bins: int = 5,
        alpha: float = 0.5,
        iv_threshold: float = 0.02,
        C: float = 1.0,
        max_iter: int = 2000,
    ):
        self.n_bins = n_bins
        self.alpha = alpha
        self.iv_threshold = iv_threshold
        self.C = C
        self.max_iter = max_iter

    def _validate_parameters(self) -> None:
        for name, value in (
            ("信息价值门槛", self.iv_threshold),
            ("正则化倒数参数", self.C),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name}必须是有限数值。")

        if self.iv_threshold < 0:
            raise ValueError("信息价值门槛不能为负数。")
        if self.C <= 0:
            raise ValueError("正则化倒数参数必须大于零。")
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, Integral)
            or self.max_iter < 1
        ):
            raise ValueError("最大迭代次数必须是正整数。")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit on training data only."""
        self._validate_parameters()

        # Fixed business rules do not read the label.
        features = build_business_features(X)
        target = validate_binary_target(y, features.index)

        # Fit the whole preprocessing chain inside the data passed to fit().
        binner = TrainQuantileBinner(n_bins=self.n_bins)
        bins = binner.fit_transform(features)

        encoder = WOEEncoder(
            alpha=self.alpha,
            min_bin_samples=20,
            unknown_policy="neutral",
        )
        encoded = encoder.fit_transform(bins, target)

        selection_report = encoder.iv_report_.copy(deep=True)
        selection_report["selection_reason"] = "待检查"
        selected = []

        # The IV report is already sorted by IV, descending. When two columns
        # are identical, the first one encountered is kept.
        for col in selection_report.index:
            if encoded[col].nunique(dropna=False) <= 1:
                selection_report.loc[
                    col, "selection_reason"
                ] = "删除：训练编码为常数"
                continue

            if selection_report.loc[col, "iv"] < self.iv_threshold:
                selection_report.loc[
                    col, "selection_reason"
                ] = "删除：低于信息价值门槛"
                continue

            duplicate_of = next(
                (
                    kept
                    for kept in selected
                    if np.array_equal(
                        encoded[col].to_numpy(),
                        encoded[kept].to_numpy(),
                    )
                ),
                None,
            )

            if duplicate_of is not None:
                selection_report.loc[
                    col, "selection_reason"
                ] = f"删除：编码与 {duplicate_of} 完全重复"
                continue

            selected.append(col)
            selection_report.loc[col, "selection_reason"] = "保留"

        if not selected:
            raise ValueError(
                "筛选后没有可用特征，请检查数据、分箱和初筛门槛。"
            )

        design = encoded.loc[:, selected]
        if not np.isfinite(design.to_numpy()).all():
            raise ValueError("编码后的模型输入包含非有限值。")

        estimator = LogisticRegression(
            C=float(self.C),
            solver="lbfgs",
            max_iter=int(self.max_iter),
            tol=1e-6,
            class_weight=None,
            fit_intercept=True,
        )

        # Do not ignore a non-converged optimiser and then treat the result as
        # a normal model.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                estimator.fit(design, target)
            except ConvergenceWarning as exc:
                raise RuntimeError(
                    "逻辑回归未收敛。请检查特征、正则化和迭代预算，"
                    "不要直接使用本次结果。"
                ) from exc

        # Store the fitted state only after every step succeeded.
        self.binner_ = binner
        self.encoder_ = encoder
        self.selected_features_ = selected
        self.selection_report_ = selection_report
        self.estimator_ = estimator
        self.classes_ = estimator.classes_.copy()
        self.training_bad_rate_ = float(target.mean())
        self.training_samples_ = len(target)

        self.coefficient_report_ = pd.DataFrame(
            {
                "feature": selected,
                "coefficient": estimator.coef_[0],
            }
        ).set_index("feature")

        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediction-time transformation; nothing is refitted."""
        check_is_fitted(
            self,
            ["binner_", "encoder_", "selected_features_", "estimator_"],
        )

        features = build_business_features(X)
        bins = self.binner_.transform(features)
        encoded = self.encoder_.transform(bins)

        design = encoded.loc[:, self.selected_features_]
        if not np.isfinite(design.to_numpy()).all():
            raise ValueError("预测输入包含非有限值。")

        return design

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return both class probabilities, ordered by the fitted classes."""
        design = self.transform_features(X)
        return self.estimator_.predict_proba(design)

    def predict_bad_probability(self, X: pd.DataFrame) -> pd.Series:
        """Return the probability of label 1, keeping the application index."""
        probabilities = self.predict_proba(X)
        bad_position = int(np.flatnonzero(self.classes_ == 1)[0])

        return pd.Series(
            probabilities[:, bad_position],
            index=X.index,
            name="predicted_bad_probability",
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Standard classification interface.

        The 0.5 threshold is not an approval strategy.
        """
        return (
            self.predict_bad_probability(X).to_numpy() >= 0.5
        ).astype("int64")

    def unknown_bin_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Report unseen bins in scoring data without changing fitted state."""
        check_is_fitted(self, ["binner_", "encoder_"])

        features = build_business_features(X)
        bins = self.binner_.transform(features)

        return self.encoder_.unknown_report(bins)


def evaluate_probabilities(
    y: pd.Series,
    probability: pd.Series,
) -> dict:
    """Evaluate continuous probabilities; no approval threshold is chosen here."""
    if not isinstance(probability, pd.Series):
        raise TypeError("预测概率必须是带索引的序列。")

    target = validate_binary_target(y, probability.index)

    try:
        values = probability.astype("float64").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("预测概率不能转换为数值。") from exc

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("预测概率必须位于零到一之间且不能缺失。")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        target,
        values,
        pos_label=1,
        drop_intermediate=False,
    )

    auc = float(roc_auc_score(target, values))

    return {
        "样本数": len(target),
        "实际标签一占比": float(target.mean()),
        "平均预测概率": float(values.mean()),
        "排序曲线下面积": auc,
        "基尼系数": 2.0 * auc - 1.0,
        "最大分布差异": float(
            np.max(np.abs(true_positive_rate - false_positive_rate))
        ),
        "平均精确率": float(
            average_precision_score(target, values)
        ),
        "布里尔分数": float(
            brier_score_loss(target, values)
        ),
        "对数损失": float(
            log_loss(target, values, labels=[0, 1])
        ),
    }


def save_baseline_reports(
    model: RiskLogisticModel,
    metrics: pd.DataFrame,
    unknown_report: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write the auditable Step 7 reports (metrics, selection, coefficients)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for key, frame, name in (
        ("baseline_metrics", metrics, "baseline_metrics.csv"),
        ("feature_selection", model.selection_report_, "feature_selection.csv"),
        ("coefficients", model.coefficient_report_, "coefficients.csv"),
        ("unknown_bins", unknown_report, "unknown_bins.csv"),
    ):
        path = out_dir / name
        frame.to_csv(path, encoding="utf-8-sig")
        paths[key] = path

    summary = pd.Series(
        {
            "训练样本数": model.training_samples_,
            "训练标签一占比": model.training_bad_rate_,
            "保留变量数": len(model.selected_features_),
            "截距": float(model.estimator_.intercept_[0]),
            "实际迭代次数": int(model.estimator_.n_iter_[0]),
            "正则化倒数参数": model.C,
            "信息价值门槛": model.iv_threshold,
        },
        name="取值",
    )
    path = out_dir / "model_summary.csv"
    summary.to_csv(path, encoding="utf-8-sig", header=True)
    paths["model_summary"] = path

    return paths
