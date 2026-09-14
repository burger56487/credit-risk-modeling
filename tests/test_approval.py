"""Step 13 tests: threshold rules, scenario utility and selection discipline."""
import numpy as np
import pandas as pd
import pytest

from src.strategy.approval import (
    ApprovalRule,
    NoFeasiblePolicyError,
    PolicyRequirements,
    UtilityAssumptions,
    build_policy_curve,
    evaluate_fixed_rule,
    select_policy,
)


@pytest.fixture
def sample():
    index = pd.Index(range(100, 106))

    y = pd.Series([0, 1, 0, 1, 0, 1], index=index)
    probability = pd.Series([0.01, 0.02, 0.02, 0.1, 0.2, 0.8], index=index)

    return y, probability


def test_ties_receive_same_decision(sample):
    _, probability = sample
    rule = ApprovalRule(threshold=0.02)

    assert rule.decide(probability).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_curve_matches_hand_calculation(sample):
    y, probability = sample
    assumptions = UtilityAssumptions(
        non_event_gain=100, event_loss=300, approval_cost=5
    )

    curve = build_policy_curve(y, probability, [0.02], assumptions)
    row = curve.loc[curve["threshold"].eq(0.02)].iloc[0]

    assert row["approved_count"] == 3
    assert row["approved_event_count"] == 1
    assert row["approved_non_event_count"] == 2
    assert row["approval_rate"] == pytest.approx(0.5)
    assert row["approved_event_rate"] == pytest.approx(1 / 3)

    # Two non-events, one event and three processing costs.
    assert row["scenario_utility"] == pytest.approx(-115.0)

    # The approved probability mass is 0.05.
    assert row["predicted_scenario_utility"] == pytest.approx(265.0)
    assert row["scenario_utility_per_applicant"] == pytest.approx(-115.0 / 6)


def test_reject_all_and_approve_all_baselines(sample):
    y, probability = sample

    curve = build_policy_curve(y, probability, [0.02], UtilityAssumptions())

    reject_all = curve.loc[curve["rule_mode"].eq("全部拒绝")].iloc[0]
    approve_all = curve.loc[curve["threshold"].eq(1.0)].iloc[0]

    assert reject_all["approved_count"] == 0
    assert pd.isna(reject_all["approved_event_rate"])
    assert reject_all["scenario_utility"] == 0.0

    assert approve_all["approved_count"] == len(y)
    assert approve_all["approved_event_rate"] == pytest.approx(y.mean())
    assert approve_all["non_event_rejection_rate"] == 0.0


def test_zero_threshold_is_not_the_same_as_reject_all():
    y = pd.Series([0, 1])
    probability = pd.Series([0.0, 1.0])

    curve = build_policy_curve(y, probability, [0.0], UtilityAssumptions())
    zero_threshold = curve.loc[curve["threshold"].eq(0.0)].iloc[0]

    assert zero_threshold["approved_count"] == 1

    decisions = ApprovalRule(mode="全部拒绝", threshold=None).decide(
        probability
    )

    assert not decisions.any()


def test_policy_selection_finds_expected_candidate():
    y = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.01, 0.02, 0.4, 0.8])

    curve = build_policy_curve(
        y, probability, [0.01, 0.02, 0.5], UtilityAssumptions()
    )
    requirements = PolicyRequirements(
        min_approval_rate=0,
        max_approved_event_rate=0.1,
        min_approved_count=1,
    )

    selected = select_policy(curve, requirements)

    assert selected.rule.threshold == 0.02
    assert selected.development_result["scenario_utility"] == pytest.approx(
        190.0
    )
    assert selected.rule.mode == "概率阈值"


def test_tied_utility_prefers_fewer_approvals():
    y = pd.Series([0, 0, 1])
    probability = pd.Series([0.1, 0.2, 0.9])

    assumptions = UtilityAssumptions(
        non_event_gain=10, event_loss=10, approval_cost=10
    )
    curve = build_policy_curve(y, probability, [0.1, 0.2], assumptions)

    requirements = PolicyRequirements(
        min_approval_rate=0,
        max_approved_event_rate=0.5,
        min_approved_count=1,
        require_positive_utility=False,
    )

    selected = select_policy(curve, requirements)

    assert selected.rule.threshold == 0.1


def test_no_feasible_policy_does_not_relax_constraints(sample):
    y, probability = sample
    curve = build_policy_curve(y, probability, [0.02], UtilityAssumptions())

    with pytest.raises(NoFeasiblePolicyError, match="不自动放宽"):
        select_policy(curve, PolicyRequirements(min_approved_count=100))


def test_cost_sensitivity_keeps_the_same_rule(sample):
    y, probability = sample
    rule = ApprovalRule(threshold=0.1)

    first = evaluate_fixed_rule(
        y, probability, rule, UtilityAssumptions(event_loss=100)
    )
    second = evaluate_fixed_rule(
        y, probability, rule, UtilityAssumptions(event_loss=200)
    )

    assert first["approved_count"] == second["approved_count"]
    assert first["approved_event_rate"] == second["approved_event_rate"]

    expected_decrease = first["approved_event_count"] * 100
    assert first["scenario_utility"] - second[
        "scenario_utility"
    ] == pytest.approx(expected_decrease)

    assert rule.threshold == 0.1


def test_single_class_subset_has_undefined_capture_rate():
    y = pd.Series([0, 0])
    probability = pd.Series([0.1, 0.2])

    curve = build_policy_curve(y, probability, [0.1], UtilityAssumptions())

    assert curve["rejected_event_capture_rate"].isna().all()


def test_index_mismatch_rejected(sample):
    y, probability = sample

    with pytest.raises(ValueError, match="索引或顺序"):
        build_policy_curve(
            y.iloc[::-1], probability, [0.1], UtilityAssumptions()
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_probability_rejected(sample, value):
    _, probability = sample
    probability = probability.copy()
    probability.iloc[0] = value

    with pytest.raises(ValueError):
        ApprovalRule().decide(probability)


def test_input_not_modified(sample):
    y, probability = sample
    original_y = y.copy(deep=True)
    original_probability = probability.copy(deep=True)

    build_policy_curve(y, probability, [0.02], UtilityAssumptions())

    pd.testing.assert_series_equal(y, original_y)
    pd.testing.assert_series_equal(probability, original_probability)


def test_extreme_utility_overflow_rejected(sample):
    y, probability = sample

    with pytest.raises(ValueError, match="非有限值"):
        build_policy_curve(
            y,
            probability,
            [1.0],
            UtilityAssumptions(non_event_gain=1e308),
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"non_event_gain": 0},
        {"event_loss": -1},
        {"approval_cost": np.inf},
        {"approval_cost": True},
    ],
)
def test_invalid_utility_assumptions_rejected(parameters):
    with pytest.raises(ValueError):
        UtilityAssumptions(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"min_approval_rate": -0.1},
        {"max_approved_event_rate": 1.5},
        {"min_approved_count": 0},
        {"require_positive_utility": "yes"},
    ],
)
def test_invalid_policy_requirements_rejected(parameters):
    with pytest.raises(ValueError):
        PolicyRequirements(**parameters)


def test_invalid_rule_configuration_rejected():
    with pytest.raises(ValueError, match="不能同时配置"):
        ApprovalRule(mode="全部拒绝", threshold=0.1)

    with pytest.raises(ValueError, match="概率阈值"):
        ApprovalRule(threshold=1.5)

    with pytest.raises(ValueError, match="未知的模拟审批规则"):
        ApprovalRule(mode="按原因码拒绝", threshold=None)


def test_empty_or_invalid_threshold_grid_rejected(sample):
    y, probability = sample

    with pytest.raises(ValueError, match="阈值网格不能为空"):
        build_policy_curve(y, probability, [], UtilityAssumptions())

    with pytest.raises(ValueError, match="候选阈值"):
        build_policy_curve(y, probability, [1.5], UtilityAssumptions())


def test_curve_structure_is_complete(sample):
    y, probability = sample

    curve = build_policy_curve(
        y, probability, [0.0, 0.02, 0.02], UtilityAssumptions()
    )

    # De-duplicated grid plus the appended approve-all baseline.
    assert curve["threshold"].tolist()[1:] == [0.0, 0.02, 1.0]
    assert curve["rule_mode"].iloc[0] == "全部拒绝"
    assert curve["approved_count"].is_monotonic_increasing
    assert (curve["approval_rate"].iloc[1:] >= 0).all()
