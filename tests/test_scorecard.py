"""Step 9 tests: scale identities, boundary behaviour and decomposition.

These check the maths and the boundaries, not predictive performance.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.logistic import RiskLogisticModel
from src.models.scorecard import LogisticScorecard, ScoreScale


@pytest.fixture
def sample():
    n = 80
    index = pd.Index(range(3000, 3000 + n))
    labels = np.arange(n) % 2
    external_score = np.where(labels == 1, 0.2, 0.8)

    X = pd.DataFrame(
        {
            "sk_id_curr": np.arange(n),
            "amt_income_total": 100000.0,
            "amt_credit": 200000.0,
            "amt_annuity": 10000.0,
            "days_birth": -10957.5,
            "days_employed": -3652.5,
            "ext_source_1": external_score,
            "ext_source_2": external_score.copy(),
            "ext_source_3": 0.5,
            "bureau_cnt": 1,
            "bureau_debt_total": 20000.0,
        },
        index=index,
    )

    y = pd.Series(labels, index=index, name="target")
    return X, y


@pytest.fixture
def fitted_model(sample):
    X, y = sample
    return RiskLogisticModel().fit(X, y)


def test_base_anchor_and_odds_doubling():
    """The anchor is defined on odds, so the probability does not double."""
    scale = ScoreScale()

    odds = np.array([1.0 / 50.0, 1.0 / 25.0])
    probabilities = pd.Series(odds / (1.0 + odds))
    scores = scale.scores_from_probabilities(probabilities)

    assert scores.iloc[0] == pytest.approx(600.0)
    assert scores.iloc[1] == pytest.approx(580.0)
    assert probabilities.iloc[0] == pytest.approx(1.0 / 51.0)
    assert probabilities.iloc[1] < 2.0 * probabilities.iloc[0]


def test_higher_probability_means_lower_score():
    scale = ScoreScale()
    probabilities = pd.Series([0.005, 0.02, 0.2])

    scores = scale.scores_from_probabilities(probabilities)

    assert (np.diff(scores.to_numpy()) < 0).all()


def test_probability_score_round_trip():
    scale = ScoreScale()
    probabilities = pd.Series(
        [1e-6, 0.01, 0.08, 0.5, 0.95, 1.0 - 1e-6],
        index=[10, 11, 12, 13, 14, 15],
    )

    scores = scale.scores_from_probabilities(probabilities)
    recovered = scale.probabilities_from_scores(scores)

    np.testing.assert_allclose(
        recovered.to_numpy(),
        probabilities.to_numpy(),
        atol=1e-12,
        rtol=0,
    )
    pd.testing.assert_index_equal(recovered.index, probabilities.index)


@pytest.mark.parametrize(
    "probability",
    [0.0, 1.0, -0.1, 1.1, np.nan, np.inf],
)
def test_invalid_probability_rejected(probability):
    with pytest.raises(ValueError):
        ScoreScale().scores_from_probabilities(pd.Series([probability]))


@pytest.mark.parametrize(
    "parameters",
    [
        {"base_score": np.inf},
        {"base_score": True},
        {"base_bad_good_odds": 0.0},
        {"base_bad_good_odds": -1.0},
        {"points_to_double_odds": 0.0},
        {"points_to_double_odds": -20.0},
    ],
)
def test_invalid_scale_rejected(parameters):
    with pytest.raises(ValueError):
        ScoreScale(**parameters)


def test_extreme_log_odds_can_still_produce_finite_scores():
    scale = ScoreScale()
    log_odds = pd.Series([-1000.0, 1000.0])

    scores = scale.scores_from_log_odds(log_odds)
    recovered = scale.probabilities_from_scores(scores)

    assert np.isfinite(scores.to_numpy()).all()

    # Extreme inputs may saturate the probability; the raw score stays finite.
    assert recovered.iloc[0] == 0.0
    assert recovered.iloc[1] == 1.0


def test_score_matches_original_model_probability(sample, fitted_model):
    X, _ = sample
    scale = ScoreScale()
    scorecard = LogisticScorecard(fitted_model, scale)

    result = scorecard.score(X)
    original_probability = fitted_model.predict_bad_probability(X)

    probability_based_scores = scale.scores_from_probabilities(
        original_probability
    )

    np.testing.assert_allclose(
        result["score_raw"],
        probability_based_scores,
        atol=1e-9,
        rtol=0,
    )
    np.testing.assert_allclose(
        result["predicted_bad_probability"],
        original_probability,
        atol=1e-12,
        rtol=0,
    )


def test_components_reconstruct_total_score(sample, fitted_model):
    X, _ = sample
    scorecard = LogisticScorecard(fitted_model)

    scores = scorecard.score(X)
    parts = scorecard.contributions(X)
    reconstructed = parts.sum(axis=1) + scorecard.base_points

    np.testing.assert_allclose(
        reconstructed,
        scores["score_raw"],
        atol=1e-9,
        rtol=0,
    )
    pd.testing.assert_index_equal(parts.index, X.index)


def test_training_bin_points_match_formula(fitted_model):
    scale = ScoreScale()
    scorecard = LogisticScorecard(fitted_model, scale)

    table = scorecard.training_bin_table()
    expected = -scale.factor * table["coefficient"] * table["woe"]

    np.testing.assert_allclose(
        table["points"],
        expected,
        atol=1e-12,
        rtol=0,
    )


def test_unknown_selected_bin_has_zero_contribution(sample, fitted_model):
    X, _ = sample
    scorecard = LogisticScorecard(fitted_model)

    future = X.iloc[:1].copy()
    future["ext_source_1"] = np.nan
    future["ext_source_2"] = np.nan

    parts = scorecard.contributions(future)
    scores = scorecard.score(future)
    report = scorecard.unknown_bin_report(future)

    # In this artificial sample the only signal is the external score, and the
    # two identical score columns leave exactly one retained variable.
    assert parts.shape[1] == 1
    assert parts.iloc[0, 0] == 0.0
    assert scores["score_raw"].iloc[0] == pytest.approx(scorecard.base_points)

    selected_feature = fitted_model.selected_features_[0]
    assert report.loc[selected_feature, "unknown_count"] == 1
    assert bool(report.loc[selected_feature, "is_selected"]) is True


def test_source_model_changes_do_not_change_snapshot(sample, fitted_model):
    X, _ = sample
    scorecard = LogisticScorecard(fitted_model)
    before = scorecard.score(X)

    # Mutate the source model to prove the scorecard holds its own snapshot.
    fitted_model.estimator_.intercept_[0] += 1.0

    after = scorecard.score(X)
    pd.testing.assert_frame_equal(after, before)


def test_input_unchanged_and_display_rounding(sample, fitted_model):
    X, _ = sample
    before = X.copy(deep=True)
    scorecard = LogisticScorecard(fitted_model)

    scores = scorecard.score(X)
    scorecard.contributions(X)

    pd.testing.assert_frame_equal(X, before)
    np.testing.assert_array_equal(
        scores["score_display"],
        np.rint(scores["score_raw"]),
    )


def test_unfitted_or_wrong_model_rejected(sample, fitted_model):
    with pytest.raises(NotFittedError):
        LogisticScorecard(RiskLogisticModel())

    with pytest.raises(TypeError):
        LogisticScorecard(object())

    with pytest.raises(TypeError, match="评分刻度"):
        LogisticScorecard(fitted_model, scale="not a scale")


def test_duplicate_index_rejected():
    scale = ScoreScale()
    duplicated = pd.Series([0.1, 0.2], index=[1, 1])

    with pytest.raises(ValueError, match="索引不能重复"):
        scale.scores_from_log_odds(duplicated)

    with pytest.raises(ValueError, match="不能为空"):
        scale.scores_from_probabilities(pd.Series([], dtype="float64"))


def test_non_neutral_unknown_policy_rejected(sample, fitted_model):
    fitted_model.encoder_.unknown_policy = "error"

    with pytest.raises(ValueError, match="中性回退"):
        LogisticScorecard(fitted_model)
