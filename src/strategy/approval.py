"""Step 13: probability-threshold policy and scenario utility on labelled data.

This is offline research on a public sample. It does not make credit decisions
and it does not estimate real profit:

* the unit is an assumed "utility point", not currency;
* the public label is payment difficulty under the dataset's own definition, not
  a verified loss, recovery or regulatory default;
* the sample constraints describe this development sample only — they are not a
  guarantee about a future approved population;
* thresholds selected on a validation set that has already been inspected carry
  policy-selection bias, and Step 10's fixed-model intervals do not cover it.

Rules read probabilities only. Labels are used for offline evaluation and for
choosing a candidate rule, never inside the decision itself.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd


def _finite_number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{description}必须是有限实数。")

    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{description}超出数值范围。") from exc

    if not np.isfinite(number):
        raise ValueError(f"{description}必须是有限实数。")

    return number


def _probabilities(probability: pd.Series) -> pd.Series:
    if not isinstance(probability, pd.Series):
        raise TypeError("预测概率必须是带索引的序列。")
    if probability.empty:
        raise ValueError("预测概率不能为空。")
    if not probability.index.is_unique:
        raise ValueError("预测概率的样本索引不能重复。")

    try:
        numeric = pd.to_numeric(probability, errors="raise")
        if np.iscomplexobj(numeric.to_numpy()):
            raise ValueError("不接受复数。")
        values = numeric.astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("预测概率包含无效数值。") from exc

    if (
        not np.isfinite(values.to_numpy()).all()
        or values.lt(0).any()
        or values.gt(1).any()
    ):
        raise ValueError("预测概率必须位于零到一之间且全部有限。")

    return values


def _labels(y: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(y, pd.Series):
        raise TypeError("标签必须是带索引的序列。")
    if not y.index.equals(index):
        raise ValueError("标签与预测的索引或顺序不一致。")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("标签必须是没有缺失的零或一。")

    # An evaluation subset may contain a single label class; the ratios that
    # become undefined are reported as missing instead of being invented.
    return y.astype("int64")


@dataclass(frozen=True)
class UtilityAssumptions:
    """Scenario utility parameters; they are not real loan economics."""

    non_event_gain: float = 100.0
    event_loss: float = 1000.0
    approval_cost: float = 5.0

    def __post_init__(self):
        specifications = (
            ("non_event_gain", "标签零收益"),
            ("event_loss", "标签一损失"),
            ("approval_cost", "每笔通过成本"),
        )

        for field_name, description in specifications:
            number = _finite_number(getattr(self, field_name), description)
            object.__setattr__(self, field_name, number)

        if self.non_event_gain <= 0:
            raise ValueError("标签零收益必须大于零。")
        if self.event_loss < 0 or self.approval_cost < 0:
            raise ValueError("情景损失和处理成本不能为负数。")


@dataclass(frozen=True)
class ApprovalRule:
    """A simulation rule that can be applied on its own; it never reads labels."""

    mode: str = "概率阈值"
    threshold: float | None = 0.08

    def __post_init__(self):
        if self.mode == "全部拒绝":
            if self.threshold is not None:
                raise ValueError("全部拒绝规则不能同时配置概率阈值。")

        elif self.mode == "概率阈值":
            threshold = _finite_number(self.threshold, "概率阈值")
            if not 0 <= threshold <= 1:
                raise ValueError("概率阈值必须位于零到一之间。")
            object.__setattr__(self, "threshold", threshold)

        else:
            raise ValueError("未知的模拟审批规则。")

    def decide(self, probability: pd.Series) -> pd.Series:
        """Simulated approval: ``p <= threshold``, ties decided together."""
        values = _probabilities(probability)

        if self.mode == "全部拒绝":
            return pd.Series(
                False, index=values.index, name="simulated_approval"
            )

        return values.le(self.threshold).rename("simulated_approval")


@dataclass(frozen=True)
class PolicyRequirements:
    """Development-sample constraints, not a future risk guarantee."""

    min_approval_rate: float = 0.2
    max_approved_event_rate: float = 0.06
    min_approved_count: int = 100
    require_positive_utility: bool = True

    def __post_init__(self):
        for field_name in (
            "min_approval_rate",
            "max_approved_event_rate",
        ):
            number = _finite_number(getattr(self, field_name), field_name)
            if not 0 <= number <= 1:
                raise ValueError("策略比例约束必须位于零到一之间。")
            object.__setattr__(self, field_name, number)

        if (
            isinstance(self.min_approved_count, bool)
            or not isinstance(self.min_approved_count, Integral)
            or self.min_approved_count < 1
        ):
            raise ValueError("最小通过样本数必须是正整数。")

        object.__setattr__(
            self, "min_approved_count", int(self.min_approved_count)
        )

        if not isinstance(self.require_positive_utility, bool):
            raise ValueError("正净效用要求必须是布尔值。")


class NoFeasiblePolicyError(ValueError):
    """No candidate rule satisfies the current development constraints."""


def build_policy_curve(
    y: pd.Series,
    probability: pd.Series,
    thresholds,
    assumptions: UtilityAssumptions,
) -> pd.DataFrame:
    """Threshold curve on fixed predictions.

    The approve-all and reject-all baselines are always included, the sorting is
    done once, and every cumulative quantity is read off that order. Because the
    rule is ``p <= threshold``, tied predictions can never be split.
    """
    if not isinstance(assumptions, UtilityAssumptions):
        raise TypeError("必须提供明确的情景效用配置。")

    values = _probabilities(probability)
    target = _labels(y, values.index)

    try:
        raw_thresholds = list(thresholds)
    except TypeError as exc:
        raise TypeError("阈值网格必须是可迭代序列。") from exc

    if not raw_thresholds:
        raise ValueError("阈值网格不能为空。")

    checked = []
    for value in raw_thresholds:
        threshold = _finite_number(value, "候选阈值")
        if not 0 <= threshold <= 1:
            raise ValueError("候选阈值必须位于零到一之间。")
        checked.append(threshold)

    # The appended one guarantees the approve-all baseline; de-duplicate and sort.
    grid = np.unique(np.asarray(checked + [1.0]))

    order = np.argsort(values.to_numpy(), kind="stable")
    sorted_probability = values.to_numpy()[order]
    sorted_target = target.to_numpy()[order]

    cumulative_events = np.r_[0, np.cumsum(sorted_target, dtype="int64")]
    cumulative_probability = np.r_[
        0.0, np.cumsum(sorted_probability, dtype="float64")
    ]

    # The right-side insertion point is exactly "less than or equal to".
    threshold_counts = np.searchsorted(
        sorted_probability, grid, side="right"
    )
    approved_count = np.r_[0, threshold_counts].astype("int64")

    approved_events = cumulative_events[approved_count]
    approved_non_events = approved_count - approved_events
    probability_sum = cumulative_probability[approved_count]

    total_count = len(target)
    total_events = int(target.sum())
    total_non_events = total_count - total_events

    rejected_events = total_events - approved_events
    rejected_non_events = total_non_events - approved_non_events

    approved_rate = approved_count / total_count

    observed_event_rate = np.full(len(approved_count), np.nan)
    predicted_event_rate = np.full(len(approved_count), np.nan)

    np.divide(
        approved_events,
        approved_count,
        out=observed_event_rate,
        where=approved_count > 0,
    )
    np.divide(
        probability_sum,
        approved_count,
        out=predicted_event_rate,
        where=approved_count > 0,
    )

    if total_events > 0:
        rejected_event_capture = rejected_events / total_events
    else:
        rejected_event_capture = np.full(len(approved_count), np.nan)

    if total_non_events > 0:
        non_event_rejection = rejected_non_events / total_non_events
    else:
        non_event_rejection = np.full(len(approved_count), np.nan)

    with np.errstate(over="ignore", invalid="ignore"):
        scenario_utility = (
            approved_non_events * assumptions.non_event_gain
            - approved_events * assumptions.event_loss
            - approved_count * assumptions.approval_cost
        )

        predicted_utility = (
            (approved_count - probability_sum) * assumptions.non_event_gain
            - probability_sum * assumptions.event_loss
            - approved_count * assumptions.approval_cost
        )

    if not (
        np.isfinite(scenario_utility).all()
        and np.isfinite(predicted_utility).all()
    ):
        raise ValueError("情景效用计算产生非有限值，请检查成本量级。")

    return pd.DataFrame(
        {
            "rule_mode": ["全部拒绝"] + ["概率阈值"] * len(grid),
            "threshold": np.r_[np.nan, grid],
            "approved_count": approved_count,
            "approved_event_count": approved_events,
            "approved_non_event_count": approved_non_events,
            "approval_rate": approved_rate,
            "approved_event_rate": observed_event_rate,
            "predicted_approved_event_rate": predicted_event_rate,
            "rejected_event_capture_rate": rejected_event_capture,
            "non_event_rejection_rate": non_event_rejection,
            "scenario_utility": scenario_utility,
            "scenario_utility_per_applicant": scenario_utility / total_count,
            "predicted_scenario_utility": predicted_utility,
        }
    )


@dataclass
class SelectedPolicy:
    rule: ApprovalRule
    development_result: pd.Series


def select_policy(
    curve: pd.DataFrame,
    requirements: PolicyRequirements,
) -> SelectedPolicy:
    """Maximise labelled scenario utility inside the sample constraints.

    Ties are broken towards fewer approvals and a lower threshold. When nothing
    qualifies, ``NoFeasiblePolicyError`` is raised rather than quietly relaxing a
    constraint or falling back to approve-all.
    """
    if not isinstance(requirements, PolicyRequirements):
        raise TypeError("必须提供明确的策略约束。")

    required_columns = {
        "threshold",
        "rule_mode",
        "approved_count",
        "approval_rate",
        "approved_event_rate",
        "scenario_utility",
    }
    if (
        not isinstance(curve, pd.DataFrame)
        or not curve.columns.is_unique
        or not required_columns.issubset(curve.columns)
    ):
        raise ValueError("策略曲线结构不完整或存在重复列。")

    feasible = (
        curve["rule_mode"].eq("概率阈值")
        & curve["approved_count"].ge(requirements.min_approved_count)
        & curve["approval_rate"].ge(requirements.min_approval_rate)
        & curve["approved_event_rate"].le(
            requirements.max_approved_event_rate
        )
    )

    if requirements.require_positive_utility:
        feasible &= curve["scenario_utility"].gt(0)

    candidates = curve.loc[feasible].copy()

    if candidates.empty:
        raise NoFeasiblePolicyError(
            "没有候选规则满足当前样本约束。"
            "不自动放宽约束，也不自动改为全部通过。"
        )

    winner = candidates.sort_values(
        ["scenario_utility", "approved_count", "threshold"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0].copy()

    return SelectedPolicy(
        rule=ApprovalRule(
            mode="概率阈值", threshold=float(winner["threshold"])
        ),
        development_result=winner,
    )


def evaluate_fixed_rule(
    y: pd.Series,
    probability: pd.Series,
    rule: ApprovalRule,
    assumptions: UtilityAssumptions,
) -> pd.Series:
    """Evaluate one fixed rule under new assumptions; no re-searching."""
    if not isinstance(rule, ApprovalRule):
        raise TypeError("必须提供有效的模拟规则。")

    thresholds = [1.0] if rule.mode == "全部拒绝" else [rule.threshold]

    curve = build_policy_curve(y, probability, thresholds, assumptions)

    selected = curve["rule_mode"].eq(rule.mode)
    if rule.mode == "概率阈值":
        selected &= curve["threshold"].eq(rule.threshold)

    return curve.loc[selected].iloc[0].copy()
