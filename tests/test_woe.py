"""Step 6 tests: WOE/IV maths, label alignment and unknown-bin handling."""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.features.woe import (
    WOEEncoder,
    candidate_features_by_iv,
    save_woe_reports,
)


@pytest.fixture
def sample():
    """Five low-risk rows (one bad) and five high-risk rows (four bad)."""
    index = pd.Index(range(100, 110))
    X = pd.DataFrame(
        {"risk_bin": ["低风险箱"] * 5 + ["高风险箱"] * 5},
        index=index,
    )
    y = pd.Series(
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        index=index,
        name="target",
    )
    return X, y


def test_matches_hand_calculation(sample):
    """Smoothing, WOE direction and IV match the worked example (ln 3)."""
    X, y = sample
    encoder = WOEEncoder(alpha=0.5).fit(X, y)
    table = encoder.tables_["risk_bin"]

    assert table.loc["低风险箱", "bad_share"] == pytest.approx(0.25)
    assert table.loc["低风险箱", "good_share"] == pytest.approx(0.75)
    assert table.loc["低风险箱", "woe"] == pytest.approx(-np.log(3))
    assert table.loc["高风险箱", "bad_share"] == pytest.approx(0.75)
    assert table.loc["高风险箱", "woe"] == pytest.approx(np.log(3))
    assert encoder.iv_report_.loc["risk_bin", "iv"] == pytest.approx(np.log(3))


def test_smoothed_shares_each_sum_to_one(sample):
    """Both smoothed class distributions stay normalised after smoothing."""
    X, y = sample
    table = WOEEncoder(alpha=0.5).fit(X, y).tables_["risk_bin"]

    assert table["bad_share"].sum() == pytest.approx(1.0)
    assert table["good_share"].sum() == pytest.approx(1.0)


def test_missing_bin_is_learned_not_neutral():
    """A missing bin seen in training is a normal bin, not a zero fallback."""
    index = pd.Index(range(20))
    X = pd.DataFrame(
        {"bin_feature": ["缺失箱"] * 10 + ["数值箱_000"] * 10},
        index=index,
    )
    y = pd.Series([1] * 10 + [0] * 10, index=index, name="target")

    encoder = WOEEncoder(alpha=0.5).fit(X, y)
    missing_woe = encoder.mapping_["bin_feature"]["缺失箱"]

    assert missing_woe > 0
    out = encoder.transform(X)
    assert out.loc[0, "bin_feature"] == pytest.approx(missing_woe)


def test_unknown_bin_neutral_policy_and_report(sample):
    """Unknown bins encode as zero and are counted in the report."""
    X, y = sample
    encoder = WOEEncoder(unknown_policy="neutral").fit(X, y)

    valid = pd.DataFrame(
        {"risk_bin": ["高风险箱", "高风险箱", "新箱"]},
        index=[201, 202, 203],
    )
    out = encoder.transform(valid)

    assert out["risk_bin"].tolist() == pytest.approx(
        [np.log(3), np.log(3), 0.0]
    )

    report = encoder.unknown_report(valid)
    assert report.loc["risk_bin", "unknown_count"] == 1
    assert report.loc["risk_bin", "unknown_rate"] == pytest.approx(1 / 3)


def test_unknown_bin_error_policy_raises(sample):
    X, y = sample
    encoder = WOEEncoder(unknown_policy="error").fit(X, y)

    with pytest.raises(ValueError, match="未知箱"):
        encoder.transform(pd.DataFrame({"risk_bin": ["新箱"]}))


def test_transform_does_not_modify_training_statistics(sample):
    X, y = sample
    encoder = WOEEncoder().fit(X, y)

    table_before = encoder.tables_["risk_bin"].copy(deep=True)
    report_before = encoder.iv_report_.copy(deep=True)

    valid = pd.DataFrame({"risk_bin": ["高风险箱"] * 100 + ["新箱"]})
    encoder.transform(valid)
    encoder.unknown_report(valid)

    pd.testing.assert_frame_equal(encoder.tables_["risk_bin"], table_before)
    pd.testing.assert_frame_equal(encoder.iv_report_, report_before)


def test_shuffled_target_index_is_rejected(sample):
    X, y = sample

    with pytest.raises(ValueError, match="索引或顺序"):
        WOEEncoder().fit(X, y.iloc[::-1])


def test_invalid_target_values_are_rejected(sample):
    X, y = sample
    y = y.copy()
    y.iloc[0] = 2

    with pytest.raises(ValueError, match="只能取零或一"):
        WOEEncoder().fit(X, y)


def test_single_class_target_is_rejected(sample):
    X, y = sample
    y = pd.Series(0, index=y.index)

    with pytest.raises(ValueError, match="同时包含零和一"):
        WOEEncoder().fit(X, y)


def test_raw_missing_and_numeric_bins_are_rejected(sample):
    X, y = sample

    missing = X.copy()
    missing.iloc[0, 0] = None
    with pytest.raises(ValueError, match="缺失箱"):
        WOEEncoder().fit(missing, y)

    numeric = pd.DataFrame({"feature": range(len(X))}, index=X.index)
    with pytest.raises(ValueError, match="先完成分箱"):
        WOEEncoder().fit(numeric, y)


def test_input_unchanged_and_output_aligned(sample):
    X, y = sample
    X_before = X.copy(deep=True)
    y_before = y.copy(deep=True)

    out = WOEEncoder().fit_transform(X, y)

    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)
    pd.testing.assert_index_equal(out.index, X.index)
    assert out.shape == X.shape


def test_refit_replaces_old_feature_state(sample):
    X, y = sample
    encoder = WOEEncoder().fit(X, y)

    renamed = X.rename(columns={"risk_bin": "new_feature"})
    encoder.fit(renamed, y)

    assert set(encoder.mapping_) == {"new_feature"}
    assert set(encoder.tables_) == {"new_feature"}


def test_unfitted_and_schema_change_raise(sample):
    X, y = sample

    with pytest.raises(NotFittedError):
        WOEEncoder().transform(X)

    encoder = WOEEncoder().fit(X, y)
    with pytest.raises(ValueError, match="输入结构变化"):
        encoder.transform(X.rename(columns={"risk_bin": "other"}))


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, True])
def test_invalid_smoothing_parameter_rejected(sample, alpha):
    X, y = sample

    with pytest.raises(ValueError, match="平滑参数"):
        WOEEncoder(alpha=alpha).fit(X, y)


def test_single_class_bin_is_flagged_without_infinite_values():
    """A bin with no bad samples is smoothed, flagged, and stays finite."""
    index = pd.Index(range(20))
    X = pd.DataFrame(
        {"risk_bin": ["全好箱"] * 10 + ["混合箱"] * 10},
        index=index,
    )
    y = pd.Series([0] * 10 + [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  index=index, name="target")

    encoder = WOEEncoder(alpha=0.5, min_bin_samples=5).fit(X, y)
    table = encoder.tables_["risk_bin"]

    assert table.loc["全好箱", "single_class_bin"]
    assert np.isfinite(table["woe"]).all()
    assert np.isfinite(encoder.iv_report_["iv"]).all()


def test_constant_feature_has_zero_iv():
    index = pd.Index(range(10))
    X = pd.DataFrame({"constant": ["数值箱_000"] * 10}, index=index)
    y = pd.Series([0, 1] * 5, index=index, name="target")

    encoder = WOEEncoder().fit(X, y)

    assert encoder.iv_report_.loc["constant", "iv"] == pytest.approx(0.0)
    assert encoder.mapping_["constant"]["数值箱_000"] == pytest.approx(0.0)


def test_candidate_screen_and_report_files(sample, tmp_path):
    """The IV screen returns a candidate list and the reports are written."""
    X, y = sample
    encoder = WOEEncoder().fit(X, y)

    candidates = candidate_features_by_iv(encoder, iv_threshold=0.02)
    assert candidates == ["risk_bin"]

    paths = save_woe_reports(encoder, X, tmp_path, candidates)
    assert set(paths) == {
        "iv_report", "bin_details", "unknown_bins", "candidate_features"
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    with pytest.raises(ValueError, match="没有变量达到"):
        candidate_features_by_iv(encoder, iv_threshold=10.0)
