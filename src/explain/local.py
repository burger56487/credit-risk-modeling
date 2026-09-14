"""Step 12: internal local explanations and reason-code prototypes.

Both pipelines are explained in the log-odds space:

    eta = f0 + sum_j(phi_j),   p = 1 / (1 + exp(-eta))

The two explanations are *not* the same kind of object:

* logistic regression: a linear decomposition ``coefficient * WOE`` relative to
  the model intercept. It is exactly additive, and it is **not** a Shapley value;
  Step 9's scorecard points are the same quantity rescaled by ``-factor``.
* gradient boosting: the framework's native tree-path attribution plus its own
  base term. Nothing here fixes an external background sample or a "standard
  applicant", so its reference is not the training mean probability either.

Because the references differ, the two models' contribution magnitudes are not
comparable. Attributions are model bookkeeping, not causal effects and not
evidence about a customer's circumstances: a missing value, a sentinel code or
an unseen bin is a data-coverage flag, never a fraud or employment finding.

Reason codes produced here are an internal review prototype. This project has no
approval policy, legal review, customer-notification requirement or data-usage
authorisation, so nothing in this module may be presented as an approval or
rejection reason.
"""
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.utils.validation import check_is_fitted

from src.features.engineering import (
    build_business_features,
    validate_numeric_frame,
)
from src.models.boosting import RiskBoostingModel
from src.models.logistic import RiskLogisticModel


# Base feature name (suffix-free) -> business group. A new feature must be
# registered here explicitly: a system must never assemble a plausible-sounding
# credit reason out of an unknown variable name.
BASE_FEATURE_GROUPS = {
    "amt_income_total": "收入及可用性",
    "income_zero": "收入及可用性",
    "amt_credit": "本次贷款规模",
    "amt_annuity": "本次还款金额",
    "bureau_debt_total": "外部债务及可用性",
    "days_birth": "年龄相关信息",
    "age_years": "年龄相关信息",
    "days_employed": "就业时长及可用性",
    "employed_years": "就业时长及可用性",
    "employment_age_inconsistent": "数据一致性",
    "ext_source_1": "外部评分信息",
    "ext_source_2": "外部评分信息",
    "ext_source_3": "外部评分信息",
    "bureau_cnt": "可见征信记录",
    "bureau_has_record": "可见征信记录",
    "credit_to_income_proxy": "贷款与收入相对规模",
    "annuity_to_income_proxy": "还款金额与收入相对规模",
    "bureau_debt_to_income_proxy": "外部债务与收入相对规模",
}

INTERNAL_REASON_CODES = {
    "收入及可用性": "内审码_01",
    "本次贷款规模": "内审码_02",
    "本次还款金额": "内审码_03",
    "外部债务及可用性": "内审码_04",
    "年龄相关信息": "内审码_05",
    "就业时长及可用性": "内审码_06",
    "数据一致性": "内审码_07",
    "外部评分信息": "内审码_08",
    "可见征信记录": "内审码_09",
    "贷款与收入相对规模": "内审码_10",
    "还款金额与收入相对规模": "内审码_11",
    "外部债务与收入相对规模": "内审码_12",
}


def feature_group(feature: str) -> str:
    """Map a known business feature to its explanation group."""
    base = feature

    for suffix in ("_missing", "_invalid", "_sentinel"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    if base not in BASE_FEATURE_GROUPS:
        raise ValueError(
            f"特征 {feature} 尚未登记解释分组，不能自动生成原因。"
        )

    return BASE_FEATURE_GROUPS[base]


def group_contributions(
    contributions: pd.DataFrame,
    feature_groups: dict,
) -> pd.DataFrame:
    """Signed sum inside each business group, preserving the total."""
    values = validate_numeric_frame(contributions)

    if values.isna().any().any():
        raise ValueError("贡献表不能包含缺失值。")

    if set(feature_groups) != set(values.columns):
        raise ValueError("解释分组必须完整对应贡献表的全部特征。")

    if not all(
        isinstance(group, str) and group.strip()
        for group in feature_groups.values()
    ):
        raise ValueError("解释组名称必须是非空字符串。")

    groups = list(dict.fromkeys(feature_groups[col] for col in values.columns))

    result = pd.DataFrame(index=values.index)

    for group in groups:
        members = [
            col for col in values.columns if feature_groups[col] == group
        ]
        result[group] = values.loc[:, members].sum(axis=1)

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("分组贡献出现非有限值。")

    return result


@dataclass
class ExplanationResult:
    prediction: pd.DataFrame
    contributions: pd.DataFrame
    grouped_contributions: pd.DataFrame
    diagnostics: pd.DataFrame
    internal_candidates: pd.DataFrame
    metadata: dict


class InternalModelExplainer:
    """Internal explainer for a frozen model snapshot.

    It does not re-train, does not change the predicted probability and never
    makes an approval decision.
    """

    def __init__(self, model, max_rows: int = 2000):
        if not isinstance(model, (RiskLogisticModel, RiskBoostingModel)):
            raise TypeError("只支持本项目的逻辑回归和梯度提升树流程。")

        check_is_fitted(model, ["selected_features_", "estimator_"])

        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, Integral)
            or max_rows < 1
        ):
            raise ValueError("单批解释样本上限必须是正整数。")

        self.max_rows = int(max_rows)
        self._model = deepcopy(model)
        self._features = list(self._model.selected_features_)

        self._feature_groups = {
            col: feature_group(col) for col in self._features
        }

        estimator = self._model.estimator_

        if not np.array_equal(estimator.classes_, [0, 1]):
            raise ValueError("当前解释器要求类别顺序为零、一。")

        if isinstance(self._model, RiskLogisticModel):
            self._kind = "逻辑回归"
            fitted_order = list(estimator.feature_names_in_)
        else:
            self._kind = "梯度提升树"
            fitted_order = estimator.booster_.feature_name()

        if fitted_order != self._features:
            raise ValueError("解释特征顺序与模型训练顺序不一致。")

    def _reference_description(self) -> str:
        if self._kind == "逻辑回归":
            return (
                "模型截距；各变量证据权重为零的线性参照，不代表平均客户。"
            )

        return (
            "原生树路径统计对应的解释参照；"
            "不是指定业务参照人群，也不是因果基准。"
        )

    def _diagnostics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Data-quality flags over all business fields.

        A flag does not mean the field entered the final model, and it says
        nothing about the direction of its risk contribution.
        """
        features = build_business_features(X)
        out = pd.DataFrame(index=features.index)

        missing_columns = [col for col in features if col.endswith("_missing")]
        invalid_columns = [col for col in features if col.endswith("_invalid")]
        sentinel_columns = [col for col in features if col.endswith("_sentinel")]

        out["raw_missing_count"] = (
            features[missing_columns].sum(axis=1).astype("int64")
        )
        out["invalid_value_count"] = (
            features[invalid_columns].sum(axis=1).astype("int64")
        )
        out["sentinel_count"] = (
            features[sentinel_columns].sum(axis=1).astype("int64")
        )
        out["employment_age_inconsistent"] = features[
            "employment_age_inconsistent"
        ].astype("int64")
        out["income_zero"] = features["income_zero"].astype("int64")

        if self._kind == "逻辑回归":
            bins = self._model.binner_.transform(features)

            unknown = pd.DataFrame(
                {
                    col: ~bins[col].isin(
                        self._model.encoder_.mapping_[col]
                    )
                    for col in self._features
                },
                index=features.index,
            )

            out["unknown_selected_bins"] = unknown.sum(axis=1).astype("Int64")
        else:
            # The tree does not use the scorecard binning, so "no unknown bin"
            # would be a false statement; it is marked as not applicable.
            out["unknown_selected_bins"] = pd.Series(
                pd.NA,
                index=features.index,
                dtype="Int64",
            )

        ordinary_flags = [
            "raw_missing_count",
            "invalid_value_count",
            "sentinel_count",
            "employment_age_inconsistent",
            "income_zero",
        ]

        out["data_review_required"] = (
            out[ordinary_flags].sum(axis=1).gt(0)
            | out["unknown_selected_bins"].fillna(0).gt(0)
        ).astype(bool)

        return out

    def _internal_candidates(
        self,
        grouped: pd.DataFrame,
        diagnostics: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        columns = [
            "sample_index",
            "rank",
            "internal_reason_code",
            "group",
            "log_odds_contribution",
            "data_review_required",
            "description",
        ]
        rows = []

        for sample_index, contributions in grouped.iterrows():
            # Drop numerical noise around zero.
            positive = {
                group: float(value)
                for group, value in contributions.items()
                if value > 1e-10
            }

            ranked = sorted(
                positive.items(), key=lambda item: (-item[1], item[0])
            )[:top_k]

            for rank, (group, value) in enumerate(ranked, start=1):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "rank": rank,
                        "internal_reason_code": INTERNAL_REASON_CODES[group],
                        "group": group,
                        "log_odds_contribution": value,
                        "data_review_required": bool(
                            diagnostics.at[
                                sample_index, "data_review_required"
                            ]
                        ),
                        "description": (
                            f"{group}相对于当前解释基准推高模型风险输出；"
                            "仅供内部复核，不构成正式审批理由。"
                        ),
                    }
                )

        return pd.DataFrame(rows, columns=columns)

    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 3,
    ) -> ExplanationResult:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("解释输入必须是数据表。")
        if len(X) > self.max_rows:
            raise ValueError("超过单批解释样本上限，请分批处理。")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, Integral)
            or top_k < 1
        ):
            raise ValueError("内部候选原因数量必须是正整数。")

        design = self._model.transform_features(X)
        estimator = self._model.estimator_

        if list(design.columns) != self._features:
            raise ValueError("转换后的特征顺序发生变化。")

        if self._kind == "逻辑回归":
            values = design.to_numpy() * estimator.coef_[0]
            base = np.full(len(design), float(estimator.intercept_[0]))
            raw_output = np.asarray(
                estimator.decision_function(design), dtype="float64"
            )
        else:
            # For a binary model the last column is the base term and the rest
            # are contributions to the raw model output.
            packed = np.asarray(
                estimator.booster_.predict(design, pred_contrib=True),
                dtype="float64",
            )

            if packed.shape != (len(design), len(self._features) + 1):
                raise ValueError("树模型贡献输出形状与预期不一致。")

            values = packed[:, :-1]
            base = packed[:, -1]
            raw_output = np.asarray(
                estimator.booster_.predict(design, raw_score=True),
                dtype="float64",
            )

        if values.shape != design.shape:
            raise ValueError("局部贡献数量与模型变量数量不一致。")

        if not all(
            np.isfinite(array).all()
            for array in (values, base, raw_output)
        ):
            raise ValueError("解释输出包含非有限值。")

        contributions = pd.DataFrame(
            values, index=design.index, columns=self._features
        )

        reconstructed = contributions.sum(axis=1).to_numpy() + base

        if not np.allclose(
            reconstructed, raw_output, atol=1e-8, rtol=1e-10
        ):
            raise RuntimeError("解释贡献无法重构模型原始输出。")

        actual_probability = estimator.predict_proba(design)[:, 1]
        reconstructed_probability = expit(reconstructed)

        if not np.allclose(
            reconstructed_probability,
            actual_probability,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise RuntimeError("解释重构概率与模型概率不一致。")

        grouped = group_contributions(contributions, self._feature_groups)

        if not np.allclose(
            grouped.sum(axis=1),
            contributions.sum(axis=1),
            atol=1e-8,
            rtol=1e-10,
        ):
            raise RuntimeError("业务分组改变了总贡献。")

        diagnostics = self._diagnostics(X)
        candidates = self._internal_candidates(
            grouped, diagnostics, int(top_k)
        )

        reconstruction_error = np.abs(reconstructed - raw_output)

        prediction = pd.DataFrame(
            {
                "base_log_odds": base,
                "model_log_odds": raw_output,
                "predicted_bad_probability": actual_probability,
                "reconstruction_error": reconstruction_error,
            },
            index=design.index,
        )

        metadata = {
            "模型类型": self._kind,
            "解释单位": "对数坏好比",
            "解释参照": self._reference_description(),
            "解释样本数": len(design),
            "最大重构误差": float(reconstruction_error.max()),
            "原因候选数量上限": int(top_k),
            "是否重新训练模型": False,
            "是否修改预测概率": False,
            "原因用途": "内部模型与数据复核，不用于自动拒贷或对外告知",
            "限制": [
                "贡献是模型归因，不是因果影响。",
                "不同模型参照不同，贡献大小不能直接横向比较。",
                "业务分组可能发生正负贡献抵消，需保留逐变量明细。",
                "数据复核标记本身不说明风险贡献方向。",
            ],
        }

        return ExplanationResult(
            prediction=prediction,
            contributions=contributions,
            grouped_contributions=grouped,
            diagnostics=diagnostics,
            internal_candidates=candidates,
            metadata=metadata,
        )
