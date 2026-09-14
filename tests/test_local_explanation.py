"""Step 12 tests: explanation identities, grouping and reason-code boundaries.

The artificial data checks the explanation identities only; it is not evidence
about model performance.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.exceptions import NotFittedError

from src.explain.local import (
    BASE_FEATURE_GROUPS,
    INTERNAL_REASON_CODES,
    InternalModelExplainer,
    feature_group,
    group_contributions,
)
from src.models.boosting import RiskBoostingModel
from src.models.logistic import RiskLogisticModel
from src.models.scorecard import LogisticScorecard, ScoreScale


@pytest.fixture
def sample():
    n = 160
    index = pd.Index(range(5000, 5000 + n))
    labels = np.arange(n) % 2
    score = np.where(labels == 1, 0.2, 0.8)

    X = pd.DataFrame(
        {
            "sk_id_curr": np.arange(n),
            "amt_income_total": 100000.0,
            "amt_credit": 200000.0,
            "amt_annuity": 10000.0,
            "days_birth": -10957.5,
            "days_employed": -3652.5,
            "ext_source_1": score,
            "ext_source_2": score.copy(),
            "ext_source_3": 0.5,
            "bureau_cnt": 1,
            "bureau_debt_total": 20000.0,
        },
        index=index,
    )

    y = pd.Series(labels, index=index, name="target")
    return X, y


@pytest.fixture(params=["逻辑回归", "梯度提升树"])
def fitted_model(request, sample):
    X, y = sample

    if request.param == "逻辑回归":
        return RiskLogisticModel().fit(X, y)

    return RiskBoostingModel(
        n_estimators=15,
        num_leaves=7,
        min_child_samples=10,
        n_jobs=1,
    ).fit(X, y)


def test_explanation_reconstructs_raw_output_and_probability(sample, fitted_model):
    X, _ = sample
    result = InternalModelExplainer(fitted_model).explain(X)

    reconstructed = (
        result.contributions.sum(axis=1) + result.prediction["base_log_odds"]
    )

    np.testing.assert_allclose(
        reconstructed,
        result.prediction["model_log_odds"],
        atol=1e-8,
        rtol=1e-10,
    )

    np.testing.assert_allclose(
        expit(reconstructed),
        fitted_model.predict_bad_probability(X),
        atol=1e-10,
        rtol=1e-10,
    )

    pd.testing.assert_index_equal(result.contributions.index, X.index)
    assert result.metadata["最大重构误差"] < 1e-8
    assert result.metadata["是否修改预测概率"] is False


def test_grouping_uses_signed_sum_not_absolute_sum():
    contributions = pd.DataFrame({"收入": [0.7], "收入缺失标记": [-0.6]})
    groups = {"收入": "收入组", "收入缺失标记": "收入组"}

    result = group_contributions(contributions, groups)

    assert result.loc[0, "收入组"] == pytest.approx(0.1)


def test_grouping_preserves_total_contribution(sample, fitted_model):
    X, _ = sample
    result = InternalModelExplainer(fitted_model).explain(X)

    np.testing.assert_allclose(
        result.grouped_contributions.sum(axis=1),
        result.contributions.sum(axis=1),
        atol=1e-10,
        rtol=1e-10,
    )


def test_internal_candidates_match_positive_group_contributions(
    sample, fitted_model
):
    X, _ = sample
    result = InternalModelExplainer(fitted_model).explain(X, top_k=3)

    assert not result.internal_candidates.empty

    for row in result.internal_candidates.itertuples(index=False):
        assert row.log_odds_contribution > 0
        assert 1 <= row.rank <= 3

        expected = result.grouped_contributions.loc[
            row.sample_index, row.group
        ]
        assert row.log_odds_contribution == pytest.approx(expected)

    assert not result.internal_candidates.duplicated(
        ["sample_index", "group"]
    ).any()
    assert set(result.internal_candidates["internal_reason_code"]) <= set(
        INTERNAL_REASON_CODES.values()
    )


def test_missing_data_can_be_explained_and_is_flagged(sample, fitted_model):
    X, _ = sample
    future = X.iloc[:2].copy()

    future["ext_source_1"] = np.nan
    future["ext_source_2"] = np.nan
    future["days_employed"] = 365243

    result = InternalModelExplainer(fitted_model).explain(future)

    assert np.isfinite(result.contributions.to_numpy()).all()
    assert result.diagnostics["data_review_required"].all()
    assert result.diagnostics["sentinel_count"].eq(1).all()


def test_unknown_logistic_bin_has_zero_contribution(sample):
    X, y = sample
    model = RiskLogisticModel().fit(X, y)

    future = X.iloc[:1].copy()
    future["ext_source_1"] = np.nan
    future["ext_source_2"] = np.nan

    result = InternalModelExplainer(model).explain(future)

    assert result.contributions.shape[1] == 1
    assert result.contributions.iloc[0, 0] == 0.0
    assert result.diagnostics["unknown_selected_bins"].iloc[0] == 1
    assert result.internal_candidates.empty


def test_logistic_contributions_match_existing_scorecard(sample):
    X, y = sample
    model = RiskLogisticModel().fit(X, y)
    scale = ScoreScale()

    explanation = InternalModelExplainer(model).explain(X)
    scorecard = LogisticScorecard(model, scale)
    score_parts = scorecard.contributions(X)

    # Step 9: points = -factor * log-odds contribution.
    np.testing.assert_allclose(
        score_parts.to_numpy(),
        -scale.factor * explanation.contributions.to_numpy(),
        atol=1e-9,
        rtol=1e-10,
    )


def test_source_model_changes_do_not_change_snapshot(sample, fitted_model):
    X, _ = sample
    explainer = InternalModelExplainer(fitted_model)

    before = explainer.explain(X)

    # Mutating the source object must not leak into the explainer's own copy.
    fitted_model.selected_features_.clear()

    after = explainer.explain(X)

    pd.testing.assert_frame_equal(before.contributions, after.contributions)
    pd.testing.assert_frame_equal(before.prediction, after.prediction)


def test_input_is_not_modified(sample, fitted_model):
    X, _ = sample
    before = X.copy(deep=True)

    InternalModelExplainer(fitted_model).explain(X)

    pd.testing.assert_frame_equal(X, before)


def test_raw_column_order_does_not_change_explanation(sample, fitted_model):
    X, _ = sample
    explainer = InternalModelExplainer(fitted_model)

    first = explainer.explain(X)
    second = explainer.explain(X.loc[:, list(reversed(X.columns))])

    pd.testing.assert_frame_equal(first.contributions, second.contributions)


def test_batch_limit_and_invalid_top_k_rejected(sample, fitted_model):
    X, _ = sample
    explainer = InternalModelExplainer(fitted_model, max_rows=5)

    with pytest.raises(ValueError, match="样本上限"):
        explainer.explain(X.iloc[:6])

    with pytest.raises(ValueError, match="正整数"):
        explainer.explain(X.iloc[:2], top_k=0)


def test_unfitted_and_wrong_model_rejected():
    with pytest.raises(NotFittedError):
        InternalModelExplainer(RiskLogisticModel())

    with pytest.raises(TypeError):
        InternalModelExplainer(object())


def test_incomplete_group_mapping_rejected():
    contributions = pd.DataFrame({"特征甲": [0.1], "特征乙": [0.2]})

    with pytest.raises(ValueError, match="完整对应"):
        group_contributions(contributions, {"特征甲": "业务组"})


def test_every_registered_feature_and_group_is_consistent():
    """Groups, reason codes and the feature registry must line up exactly."""
    assert set(BASE_FEATURE_GROUPS.values()) == set(INTERNAL_REASON_CODES)

    # Suffixed availability flags belong to their base feature's group.
    assert feature_group("amt_income_total_missing") == "收入及可用性"
    assert feature_group("days_employed_sentinel") == "就业时长及可用性"
    assert feature_group("ext_source_1_invalid") == "外部评分信息"

    with pytest.raises(ValueError, match="尚未登记"):
        feature_group("a_new_feature_without_a_group")


def test_group_contributions_rejects_missing_values_and_bad_names():
    contributions = pd.DataFrame({"known": [0.5]})

    with pytest.raises(ValueError, match="不能包含缺失值"):
        group_contributions(
            pd.DataFrame({"known": [np.nan]}), {"known": "收入及可用性"}
        )

    with pytest.raises(ValueError, match="非空字符串"):
        group_contributions(contributions, {"known": "  "})
