"""Step 11 tests: calibration diagnostics and reference-binning drift checks."""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.evaluation.calibration_stability import (
    NumericPSIMonitor,
    calibration_diagnostics,
)


def test_grouped_calibration_matches_hand_calculation():
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    probability = pd.Series([0.25] * 4 + [0.75] * 4)

    result = calibration_diagnostics(y, probability, n_bins=2)

    assert result.summary["分箱加权绝对校准误差"] == pytest.approx(0.0)
    assert result.summary["布里尔分数"] == pytest.approx(0.1875)
    assert result.summary["整体预测偏差"] == pytest.approx(0.0)

    np.testing.assert_allclose(result.bins["observed_rate"], [0.25, 0.75])


def test_probability_boundaries_and_empty_bins():
    y = pd.Series([0, 1, 1])
    probability = pd.Series([0.0, 0.5, 1.0])

    result = calibration_diagnostics(y, probability, n_bins=4)

    assert result.bins["n_samples"].tolist() == [1, 1, 0, 1]
    assert result.bins.loc["概率箱_02", "empty_bin"]
    assert pd.isna(result.bins.loc["概率箱_02", "observed_rate"])
    assert result.summary["空箱数"] == 1
    assert result.summary["端点概率样本数"] == 2


def test_zero_observed_events_still_have_positive_upper_bound():
    y = pd.Series([0] * 10)
    probability = pd.Series([0.1] * 10)

    result = calibration_diagnostics(y, probability, n_bins=2)
    first = result.bins.iloc[0]

    assert first["observed_rate"] == 0.0
    assert first["observed_lower"] == pytest.approx(0.0)
    assert first["observed_upper"] > 0.0
    assert np.isfinite(result.summary["对数损失"])


def test_calibration_checks_alignment():
    y = pd.Series([0, 1], index=[10, 11])
    probability = pd.Series([0.2, 0.8], index=[11, 10])

    with pytest.raises(ValueError, match="索引或顺序"):
        calibration_diagnostics(y, probability)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_probability_rejected(value):
    y = pd.Series([0, 1])
    probability = pd.Series([0.2, value])

    with pytest.raises(ValueError):
        calibration_diagnostics(y, probability)


def test_same_proportions_different_sample_sizes_have_zero_psi():
    reference = pd.DataFrame({"value": [0.0, 0.0, 1.0, np.nan]})
    current = pd.concat([reference, reference], ignore_index=True)

    monitor = NumericPSIMonitor(n_bins=2).fit(reference)
    result = monitor.report(current)

    assert result.summary.loc["value", "稳定性指数"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_psi_matches_hand_calculation():
    reference = pd.DataFrame({"value": [0.0] * 8 + [1.0] * 2})
    current = pd.DataFrame({"value": [0.0] * 2 + [1.0] * 8})

    epsilon = 1e-4
    monitor = NumericPSIMonitor(n_bins=2, epsilon=epsilon).fit(reference)
    result = monitor.report(current)

    # Two numeric bins plus the always-present missing bin.
    p = np.array([0.8, 0.2, 0.0])
    q = np.array([0.2, 0.8, 0.0])

    p_smoothed = (p + epsilon) / (1.0 + 3 * epsilon)
    q_smoothed = (q + epsilon) / (1.0 + 3 * epsilon)

    expected = np.sum(
        (q_smoothed - p_smoothed)
        * (np.log(q_smoothed) - np.log(p_smoothed))
    )

    assert result.summary.loc["value", "稳定性指数"] == pytest.approx(
        expected, abs=1e-12
    )
    assert result.settings["占比平滑参数"] == pytest.approx(epsilon)


def test_new_missing_values_are_visible_and_finite():
    reference = pd.DataFrame({"value": [0.0, 1.0, 2.0]})
    current = pd.DataFrame({"value": [np.nan, np.nan]})

    result = NumericPSIMonitor().fit(reference).report(current)

    assert np.isfinite(result.summary.loc["value", "稳定性指数"])
    assert result.summary.loc["value", "当前缺失率"] == 1.0
    assert result.summary.loc["value", "新出现分箱占比"] == 1.0


def test_all_missing_reference_handles_new_observed_values():
    reference = pd.DataFrame({"value": [np.nan] * 3})
    current = pd.DataFrame({"value": [np.nan, 7.0]})

    result = NumericPSIMonitor().fit(reference).report(current)

    assert result.summary.loc["value", "新出现分箱占比"] == pytest.approx(0.5)
    assert result.summary.loc["value", "稳定性指数"] > 0
    assert pd.isna(result.summary.loc["value", "高于参考最大值占比"])


def test_constant_bin_requires_separate_range_check():
    reference = pd.DataFrame({"value": [5.0, 5.0, 5.0]})
    current = pd.DataFrame({"value": [100.0, 100.0]})

    result = NumericPSIMonitor().fit(reference).report(current)

    assert result.summary.loc["value", "稳定性指数"] == pytest.approx(0.0)
    assert result.summary.loc["value", "高于参考最大值占比"] == 1.0
    assert result.summary.loc["value", "低于参考最小值占比"] == 0.0


def test_report_does_not_change_reference_state_or_input():
    reference = pd.DataFrame({"value": np.arange(20, dtype="float64")})
    current = pd.DataFrame({"value": [-100.0, 1e6, np.nan]})

    monitor = NumericPSIMonitor().fit(reference)

    before_edges = monitor.binner_.edges_["value"].copy()
    before_counts = monitor.reference_counts_["value"].copy()
    before_current = current.copy(deep=True)

    monitor.report(current)

    np.testing.assert_array_equal(
        monitor.binner_.edges_["value"], before_edges
    )
    pd.testing.assert_series_equal(
        monitor.reference_counts_["value"], before_counts
    )
    pd.testing.assert_frame_equal(current, before_current)


def test_smoothed_distributions_sum_to_one():
    reference = pd.DataFrame({"value": [0.0, 1.0, np.nan]})
    current = pd.DataFrame({"value": [1.0, 1.0, 3.0]})

    result = NumericPSIMonitor().fit(reference).report(current)
    table = result.bins.xs("value", level="feature")

    assert table["reference_smoothed"].sum() == pytest.approx(1.0)
    assert table["current_smoothed"].sum() == pytest.approx(1.0)
    assert (table["psi_component"] >= -1e-12).all()


def test_unfitted_and_schema_change_rejected():
    data = pd.DataFrame({"value": [0.0, 1.0]})

    with pytest.raises(NotFittedError):
        NumericPSIMonitor().report(data)

    monitor = NumericPSIMonitor().fit(data)

    with pytest.raises(ValueError, match="输入结构"):
        monitor.report(pd.DataFrame({"other": [0.0, 1.0]}))


@pytest.mark.parametrize("epsilon", [0.0, -1.0, 1.0, np.nan, True])
def test_invalid_smoothing_rejected(epsilon):
    data = pd.DataFrame({"value": [0.0, 1.0]})

    with pytest.raises(ValueError, match="平滑参数"):
        NumericPSIMonitor(epsilon=epsilon).fit(data)


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_bins": 0},
        {"n_bins": True},
        {"min_bin_samples": 0},
        {"confidence_level": 1.0},
        {"confidence_level": np.nan},
    ],
)
def test_invalid_calibration_configuration_rejected(parameters):
    y = pd.Series([0, 1])
    probability = pd.Series([0.2, 0.8])

    with pytest.raises(ValueError):
        calibration_diagnostics(y, probability, **parameters)
