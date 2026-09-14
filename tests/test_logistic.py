"""Step 7 tests: end-to-end logistic baseline and probability metrics.

The samples here are artificial. They check program properties (alignment,
state handling, error surfacing), not real model performance.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from src.models.logistic import (
    RiskLogisticModel,
    evaluate_probabilities,
    save_baseline_reports,
)


@pytest.fixture
def sample():
    """Eighty rows with two identical strong signals and several constants."""
    n = 80
    index = pd.Index(range(1000, 1000 + n))
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


def test_probability_shape_range_and_alignment(sample):
    X, y = sample
    model = RiskLogisticModel().fit(X, y)

    probabilities = model.predict_proba(X)
    bad_probability = model.predict_bad_probability(X)

    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
    )
    pd.testing.assert_index_equal(bad_probability.index, X.index)


def test_constant_and_duplicate_features_removed(sample):
    X, y = sample
    model = RiskLogisticModel().fit(X, y)

    retained_scores = set(model.selected_features_) & {
        "ext_source_1",
        "ext_source_2",
    }
    assert len(retained_scores) == 1
    assert len(model.selected_features_) == 1

    reasons = model.selection_report_["selection_reason"]
    assert reasons.str.contains("完全重复").any()
    assert reasons.eq("删除：训练编码为常数").any()


def test_input_data_not_modified(sample):
    X, y = sample
    before_X = X.copy(deep=True)
    before_y = y.copy(deep=True)

    model = RiskLogisticModel().fit(X, y)
    model.predict_bad_probability(X)

    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(y, before_y)


def test_identifier_and_extra_target_not_used(sample):
    """An extra label column next to the raw features must not change scores."""
    X, y = sample
    model = RiskLogisticModel().fit(X, y)
    expected = model.predict_bad_probability(X)

    changed = X.copy()
    changed["sk_id_curr"] = 999999
    changed["target"] = 1 - y

    actual = model.predict_bad_probability(changed)
    pd.testing.assert_series_equal(actual, expected)


def test_prediction_does_not_change_training_state(sample):
    X, y = sample
    model = RiskLogisticModel().fit(X, y)

    before_iv = model.encoder_.iv_report_.copy(deep=True)
    before_coefficients = model.estimator_.coef_.copy()
    before_edges = {
        col: edges.copy() for col, edges in model.binner_.edges_.items()
    }

    future = X.iloc[:5].copy()
    future["amt_income_total"] = 0.0

    model.predict_bad_probability(future)
    report = model.unknown_bin_report(future)

    pd.testing.assert_frame_equal(model.encoder_.iv_report_, before_iv)
    np.testing.assert_array_equal(
        model.estimator_.coef_,
        before_coefficients,
    )

    for col, edges in before_edges.items():
        np.testing.assert_array_equal(model.binner_.edges_[col], edges)

    assert report["unknown_count"].sum() > 0


def test_misaligned_target_rejected(sample):
    X, y = sample

    with pytest.raises(ValueError, match="索引或顺序"):
        RiskLogisticModel().fit(X, y.iloc[::-1])


def test_single_class_training_rejected(sample):
    X, y = sample
    single_class = pd.Series(0, index=y.index)

    with pytest.raises(ValueError, match="同时包含两类"):
        RiskLogisticModel().fit(X, single_class)


def test_empty_feature_selection_rejected(sample):
    X, y = sample

    with pytest.raises(ValueError, match="没有可用特征"):
        RiskLogisticModel(iv_threshold=1e6).fit(X, y)


def test_missing_raw_column_and_unfitted_model_rejected(sample):
    X, y = sample

    with pytest.raises(NotFittedError):
        RiskLogisticModel().predict_proba(X)

    model = RiskLogisticModel().fit(X, y)
    with pytest.raises(ValueError, match="缺少必需字段"):
        model.predict_proba(X.drop(columns=["amt_credit"]))


def test_constant_probability_metrics():
    """A constant 0.5 score is the random-ordering reference point."""
    index = pd.Index([10, 11, 12, 13])
    y = pd.Series([0, 1, 0, 1], index=index)
    probability = pd.Series(0.5, index=index)

    result = evaluate_probabilities(y, probability)

    assert result["样本数"] == 4
    assert result["排序曲线下面积"] == pytest.approx(0.5)
    assert result["基尼系数"] == pytest.approx(0.0)
    assert result["最大分布差异"] == pytest.approx(0.0)
    assert result["平均精确率"] == pytest.approx(0.5)
    assert result["布里尔分数"] == pytest.approx(0.25)
    assert result["对数损失"] == pytest.approx(np.log(2))


def test_evaluation_rejects_wrong_alignment_and_invalid_probability():
    y = pd.Series([0, 1], index=[10, 11])

    reversed_index = pd.Series([0.2, 0.8], index=[11, 10])
    with pytest.raises(ValueError, match="索引或顺序"):
        evaluate_probabilities(y, reversed_index)

    invalid = pd.Series([0.2, 1.2], index=[10, 11])
    with pytest.raises(ValueError, match="零到一"):
        evaluate_probabilities(y, invalid)


def test_nonconvergence_is_not_silently_accepted(sample, monkeypatch):
    X, y = sample

    def fail_fit(*args, **kwargs):
        raise ConvergenceWarning("人工模拟未收敛")

    monkeypatch.setattr(
        "src.models.logistic.LogisticRegression.fit",
        fail_fit,
    )

    with pytest.raises(RuntimeError, match="未收敛"):
        RiskLogisticModel().fit(X, y)


def test_invalid_parameters_rejected(sample):
    X, y = sample

    with pytest.raises(ValueError, match="正则化倒数参数"):
        RiskLogisticModel(C=0.0).fit(X, y)

    with pytest.raises(ValueError, match="信息价值门槛"):
        RiskLogisticModel(iv_threshold=-0.1).fit(X, y)

    with pytest.raises(ValueError, match="最大迭代次数"):
        RiskLogisticModel(max_iter=0).fit(X, y)


def test_saved_reports_match_fitted_state(sample, tmp_path):
    """The exported reports are the fitted tables, not recomputed copies."""
    X, y = sample
    model = RiskLogisticModel().fit(X, y)
    probability = model.predict_bad_probability(X)

    metrics = pd.DataFrame(
        {
            "训练集回代诊断": evaluate_probabilities(y, probability),
        }
    ).T

    paths = save_baseline_reports(
        model,
        metrics,
        model.unknown_bin_report(X),
        tmp_path,
    )

    assert set(paths) == {
        "baseline_metrics",
        "feature_selection",
        "coefficients",
        "unknown_bins",
        "model_summary",
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    selection = pd.read_csv(paths["feature_selection"], index_col=0)
    pd.testing.assert_frame_equal(
        selection,
        model.selection_report_,
        check_dtype=False,
    )
    coefficients = pd.read_csv(paths["coefficients"], index_col=0)
    pd.testing.assert_series_equal(
        coefficients["coefficient"],
        model.coefficient_report_["coefficient"],
        check_names=False,
    )
