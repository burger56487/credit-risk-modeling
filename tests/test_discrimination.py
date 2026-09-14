"""Step 10 tests: ranking metrics, paired resampling and interval construction."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.discrimination import (
    METRIC_NAMES,
    paired_stratified_bootstrap,
    ranking_metrics,
)


def reference_metrics(target, probability, sample_weight=None) -> dict:
    """Independent reference using the weighted scikit-learn metrics."""
    target = np.asarray(target)
    probability = np.asarray(probability, dtype="float64")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        target,
        probability,
        sample_weight=sample_weight,
        drop_intermediate=False,
    )

    return {
        "排序曲线下面积": float(
            roc_auc_score(target, probability, sample_weight=sample_weight)
        ),
        "最大分布差异": float(
            np.max(np.abs(true_positive_rate - false_positive_rate))
        ),
        "平均精确率": float(
            average_precision_score(
                target, probability, sample_weight=sample_weight
            )
        ),
    }


@pytest.fixture
def sample():
    """Two correlated models scored on the same artificial validation rows."""
    rng = np.random.default_rng(2026)
    n = 240
    index = pd.Index(range(7000, 7000 + n))

    latent = rng.normal(size=n)
    labels = (
        rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.1 * latent))
    ).astype("int64")

    baseline = 1.0 / (
        1.0 + np.exp(-(0.9 * latent + rng.normal(scale=0.5, size=n)))
    )
    challenger = 1.0 / (
        1.0 + np.exp(-(1.3 * latent + rng.normal(scale=0.5, size=n)))
    )

    y = pd.Series(labels, index=index, name="target")
    return (
        y,
        pd.Series(baseline, index=index),
        pd.Series(challenger, index=index),
    )


def test_weighted_metrics_match_reference(sample):
    """Weighted metrics equal both the library result and the expanded rows."""
    y, baseline, _ = sample
    rng = np.random.default_rng(3)
    weights = pd.Series(
        rng.integers(1, 4, size=len(y)).astype("float64"),
        index=y.index,
    )

    actual = ranking_metrics(y, baseline, weights)
    expected = reference_metrics(
        y.to_numpy(),
        baseline.to_numpy(),
        sample_weight=weights.to_numpy(),
    )

    for metric in METRIC_NAMES:
        assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)

    # Duplicating rows is the definition the occurrence weights must match.
    counts = weights.to_numpy().astype("int64")
    expanded = reference_metrics(
        np.repeat(y.to_numpy(), counts),
        np.repeat(baseline.to_numpy(), counts),
    )

    for metric in METRIC_NAMES:
        assert actual[metric] == pytest.approx(expanded[metric], abs=1e-12)


def test_tied_predictions_match_reference():
    """Ties must be accumulated as a group, not row by row."""
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probability = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.9, 0.2, 0.9])

    y = pd.Series(target)
    values = pd.Series(probability)

    actual = ranking_metrics(y, values)
    expected = reference_metrics(target, probability)

    for metric in METRIC_NAMES:
        assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)


def test_constant_prediction_has_expected_metrics(sample):
    y, _, _ = sample
    probability = pd.Series(0.2, index=y.index)

    actual = ranking_metrics(y, probability)

    assert actual["排序曲线下面积"] == pytest.approx(0.5)
    assert actual["最大分布差异"] == pytest.approx(0.0)
    assert actual["平均精确率"] == pytest.approx(y.mean())


def test_reverse_ranking_is_not_mistaken_for_good_direction():
    y = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.9, 0.8, 0.2, 0.1])

    actual = ranking_metrics(y, probability)

    assert actual["排序曲线下面积"] == pytest.approx(0.0)
    assert actual["最大分布差异"] == pytest.approx(1.0)


def test_identical_models_have_zero_paired_difference(sample):
    y, baseline, _ = sample

    result = paired_stratified_bootstrap(
        y,
        baseline,
        baseline.copy(),
        n_bootstrap=30,
        random_state=7,
    )

    assert (
        result.summary[["差值", "差值下界", "差值上界"]].eq(0).all().all()
    )

    difference_columns = [f"差值_{metric}" for metric in METRIC_NAMES]
    assert result.draws[difference_columns].eq(0).all().all()


def test_first_draw_matches_independent_paired_resampling(sample):
    y, baseline, challenger = sample
    seed = 17

    result = paired_stratified_bootstrap(
        y,
        baseline,
        challenger,
        n_bootstrap=10,
        random_state=seed,
    )

    target = y.to_numpy()
    good = np.flatnonzero(target == 0)
    bad = np.flatnonzero(target == 1)

    rng = np.random.default_rng(seed)
    sampled = np.concatenate(
        [
            rng.choice(good, len(good), replace=True),
            rng.choice(bad, len(bad), replace=True),
        ]
    )

    # The reference draws actual records instead of using occurrence counts.
    baseline_expected = reference_metrics(
        target[sampled], baseline.to_numpy()[sampled]
    )
    challenger_expected = reference_metrics(
        target[sampled], challenger.to_numpy()[sampled]
    )

    for metric in METRIC_NAMES:
        assert result.draws.loc[0, f"差值_{metric}"] == pytest.approx(
            challenger_expected[metric] - baseline_expected[metric],
            abs=1e-12,
        )


def test_difference_interval_comes_from_difference_draws(sample):
    y, baseline, challenger = sample

    result = paired_stratified_bootstrap(
        y,
        baseline,
        challenger,
        n_bootstrap=80,
        confidence_level=0.9,
    )

    for metric in METRIC_NAMES:
        expected = np.quantile(result.draws[f"差值_{metric}"], [0.05, 0.95])

        assert result.summary.loc[metric, "差值下界"] == pytest.approx(
            expected[0]
        )
        assert result.summary.loc[metric, "差值上界"] == pytest.approx(
            expected[1]
        )

    # The point estimate stays the observed-sample value, not a draw mean.
    point = ranking_metrics(y, challenger)["排序曲线下面积"] - ranking_metrics(
        y, baseline
    )["排序曲线下面积"]
    assert result.summary.loc["排序曲线下面积", "差值"] == pytest.approx(point)
    assert result.summary.loc["排序曲线下面积", "基准值"] == pytest.approx(
        ranking_metrics(y, baseline)["排序曲线下面积"]
    )


def test_fixed_seed_is_reproducible(sample):
    y, baseline, challenger = sample

    first = paired_stratified_bootstrap(
        y, baseline, challenger, n_bootstrap=30, random_state=3
    )
    second = paired_stratified_bootstrap(
        y, baseline, challenger, n_bootstrap=30, random_state=3
    )

    pd.testing.assert_frame_equal(first.draws, second.draws)
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_input_data_are_not_modified(sample):
    y, baseline, challenger = sample
    before = [item.copy(deep=True) for item in (y, baseline, challenger)]

    paired_stratified_bootstrap(y, baseline, challenger, n_bootstrap=10)

    for actual, expected in zip((y, baseline, challenger), before):
        pd.testing.assert_series_equal(actual, expected)


def test_shuffled_prediction_index_rejected(sample):
    y, baseline, challenger = sample

    with pytest.raises(ValueError, match="索引或顺序"):
        paired_stratified_bootstrap(
            y,
            baseline,
            challenger.iloc[::-1],
            n_bootstrap=10,
        )


def test_invalid_probability_rejected(sample):
    y, baseline, challenger = sample
    challenger = challenger.copy()
    challenger.iloc[0] = 1.2

    with pytest.raises(ValueError, match="零到一"):
        paired_stratified_bootstrap(y, baseline, challenger, n_bootstrap=10)


def test_single_observation_class_rejected():
    y = pd.Series([0, 0, 1])
    probability = pd.Series([0.1, 0.2, 0.8])

    with pytest.raises(ValueError, match="每类至少"):
        paired_stratified_bootstrap(
            y, probability, probability, n_bootstrap=10
        )


def test_negative_or_single_class_weights_rejected(sample):
    y, baseline, _ = sample

    negative = pd.Series(1.0, index=y.index)
    negative.iloc[0] = -1.0

    with pytest.raises(ValueError, match="不能为负数"):
        ranking_metrics(y, baseline, negative)

    only_good = y.eq(0).astype("float64")

    with pytest.raises(ValueError, match="同时保留两类样本"):
        ranking_metrics(y, baseline, only_good)


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_bootstrap": 1},
        {"n_bootstrap": True},
        {"confidence_level": 1.0},
        {"confidence_level": np.nan},
        {"random_state": -1},
    ],
)
def test_invalid_configuration_rejected(sample, parameters):
    y, baseline, challenger = sample

    with pytest.raises(ValueError):
        paired_stratified_bootstrap(
            y, baseline, challenger, **parameters
        )


def test_metadata_records_what_the_intervals_do_not_cover(sample):
    y, baseline, challenger = sample

    result = paired_stratified_bootstrap(
        y, baseline, challenger, n_bootstrap=50, random_state=11
    )
    metadata = result.metadata

    assert metadata["差值方向"] == "对照模型减基准模型"
    assert metadata["模型是否重新训练"] is False
    assert metadata["标签比例是否固定"] is True
    assert metadata["随机种子"] == 11
    # Fewer than 1000 draws and a small class both raise a quality note.
    assert metadata["质量提示"]
    assert metadata["标签零样本数"] + metadata["标签一样本数"] == len(y)


def test_large_finite_weights_do_not_overflow():
    """Individually finite weights must not overflow once accumulated."""
    y = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.1, 0.8, 0.4, 0.9])
    weights = pd.Series(1e308, index=y.index)

    actual = ranking_metrics(y, probability, weights)
    expected = ranking_metrics(y, probability)

    for metric in METRIC_NAMES:
        assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)


def test_weights_with_unusable_dynamic_range_rejected():
    """A weight that underflows after normalisation cannot be trusted."""
    y = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.1, 0.8, 0.4, 0.9])
    weights = pd.Series([1e308, 1e-300, 1.0, 1.0], index=y.index)

    with pytest.raises(ValueError, match="下溢"):
        ranking_metrics(y, probability, weights)

    with pytest.raises(ValueError, match="同时保留两类样本"):
        ranking_metrics(y, probability, pd.Series(0.0, index=y.index))
