"""Step 8: gradient-boosting comparison model on raw numeric features.

This is a deliberately different modelling chain from the scorecard branch:

    business features -> drop constants and exact duplicates -> LightGBM

There is no binning, no WOE encoding and no IV screen, because the tree finds
its own numeric splits and a variable with low univariate IV may still matter
through interactions. The two pipelines are therefore compared as whole
procedures, not as one classifier swapped inside a fixed feature set.

Missing values are kept and the tree handles them natively. That a model can
accept missing values does not mean an unseen missing pattern will be scored
reliably, so distribution shift still needs monitoring.
"""
from numbers import Integral, Real

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.features.engineering import build_business_features
from src.models.logistic import validate_binary_target


class RiskBoostingModel(ClassifierMixin, BaseEstimator):
    """Gradient-boosted tree pipeline from raw application features to risk.

    Only training-time constants and exact duplicates are dropped. The
    validation set is never passed to ``fit``, so there is no early stopping
    and no tuning on validation data.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 100,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validate_parameters(self) -> None:
        integer_parameters = (
            ("训练轮数", self.n_estimators, 1),
            ("最大叶子数", self.num_leaves, 2),
            ("最小叶子样本量参数", self.min_child_samples, 1),
            ("计算线程数", self.n_jobs, 1),
        )

        for name, value, minimum in integer_parameters:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name}必须是大于等于 {minimum} 的整数。")

        for name, value in (
            ("学习率", self.learning_rate),
            ("正则化参数", self.reg_lambda),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name}必须是有限数值。")

        if not 0 < self.learning_rate <= 1:
            raise ValueError("本项目要求学习率大于零且不超过一。")

        if self.reg_lambda < 0:
            raise ValueError("正则化参数不能为负数。")

        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state < 2**31
        ):
            raise ValueError("随机种子必须是规定范围内的非负整数。")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit on raw training features and training labels only."""
        self._validate_parameters()

        features = build_business_features(X)
        target = validate_binary_target(y, features.index)

        selected = []
        report_rows = []

        # Both checks use training data only.
        for col in features.columns:
            values = features[col]

            if values.nunique(dropna=False) <= 1:
                report_rows.append(
                    {
                        "feature": col,
                        "selection_reason": "删除：训练特征为常数",
                    }
                )
                continue

            duplicate_of = next(
                (
                    kept
                    for kept in selected
                    if np.array_equal(
                        values.to_numpy(),
                        features[kept].to_numpy(),
                        equal_nan=True,
                    )
                ),
                None,
            )

            if duplicate_of is not None:
                report_rows.append(
                    {
                        "feature": col,
                        "selection_reason": f"删除：与 {duplicate_of} 完全重复",
                    }
                )
                continue

            selected.append(col)
            report_rows.append({"feature": col, "selection_reason": "保留"})

        if not selected:
            raise ValueError("删除常数与重复列后，没有可用特征。")

        design = features.loc[:, selected]

        # Missing values are kept; infinities are not allowed into the model.
        if np.isinf(design.to_numpy()).any():
            raise ValueError("树模型训练特征不能包含无穷值。")

        estimator = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            num_leaves=int(self.num_leaves),
            min_child_samples=int(self.min_child_samples),
            reg_lambda=float(self.reg_lambda),
            class_weight=None,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            random_state=int(self.random_state),
            n_jobs=int(self.n_jobs),
            device_type="cpu",
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )

        # No validation set is passed, so there is no early stopping.
        estimator.fit(design, target)

        self.selected_features_ = selected
        self.selection_report_ = pd.DataFrame(report_rows).set_index("feature")
        self.estimator_ = estimator
        self.classes_ = estimator.classes_.copy()
        self.training_bad_rate_ = float(target.mean())
        self.training_samples_ = len(target)
        self.actual_iterations_ = estimator.booster_.current_iteration()

        # Split gain is a training-time statistic, not a causal contribution
        # and not the benefit measured on the validation set.
        self.gain_importance_ = (
            pd.DataFrame(
                {
                    "feature": selected,
                    "training_split_gain": (
                        estimator.booster_.feature_importance(
                            importance_type="gain"
                        )
                    ),
                }
            )
            .set_index("feature")
            .sort_values(
                "training_split_gain",
                ascending=False,
                kind="stable",
            )
        )

        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fixed business rules and the training-time column list."""
        check_is_fitted(self, ["selected_features_", "estimator_"])

        features = build_business_features(X)
        design = features.loc[:, self.selected_features_]

        if np.isinf(design.to_numpy()).any():
            raise ValueError("树模型预测特征不能包含无穷值。")

        return design

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        design = self.transform_features(X)
        return self.estimator_.predict_proba(design)

    def predict_bad_probability(self, X: pd.DataFrame) -> pd.Series:
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
