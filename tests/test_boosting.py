"""Step 8 tests: boosting pipeline, duplicate detection and comparison wiring.

The samples here are artificial. They check program behaviour, not real
predictive performance.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.boosting import RiskBoostingModel
from src.models.logistic import RiskLogisticModel, evaluate_probabilities


@pytest.fixture
def sample():
    rng = np.random.default_rng(7)
    n = 240
    index = pd.Index(range(2000, 2000 + n))

    external_score = rng.uniform(0.0, 1.0, size=n)
    labels = (external_score < 0.4).astype("int64")

    # Two score fields are exactly identical, including missing positions.
    score_with_missing = external_score.copy()
    score_with_missing[0] = np.nan

    X = pd.DataFrame(
        {
            "sk_id_curr": np.arange(n),
            "amt_income_total": rng.uniform(50000, 200000, size=n),
            "amt_credit": rng.uniform(30000, 300000, size=n),
            "amt_annuity": 10000.0,
            "days_birth": -10957.5,
            "days_employed": -3652.5,
            "ext_source_1": score_with_missing,
            "ext_source_2": score_with_missing.copy(),
            "ext_source_3": np.nan,
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
    return RiskBoostingModel(
        n_estimators=20,
        num_leaves=7,
        min_child_samples=10,
        random_state=7,
        n_jobs=1,
    ).fit(X, y)


def test_probability_shape_range_and_alignment(sample, fitted_model):
    X, _ = sample
    probabilities = fitted_model.predict_proba(X)
    risk = fitted_model.predict_bad_probability(X)

    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
    )
    pd.testing.assert_index_equal(risk.index, X.index)


def test_constant_and_duplicate_columns_removed(fitted_model):
    selected = set(fitted_model.selected_features_)

    assert "ext_source_1" in selected
    assert "ext_source_2" not in selected
    assert "ext_source_3" not in selected

    report = fitted_model.selection_report_
    assert "完全重复" in report.loc["ext_source_2", "selection_reason"]
    assert report.loc[
        "ext_source_3", "selection_reason"
    ] == "删除：训练特征为常数"


def test_missing_values_remain_supported(sample, fitted_model):
    X, _ = sample
    future = X.iloc[:3].copy()

    future["amt_income_total"] = 0.0
    future["ext_source_1"] = np.nan
    future["ext_source_2"] = np.nan
    future["days_employed"] = 365243

    design = fitted_model.transform_features(future)
    probabilities = fitted_model.predict_bad_probability(future)

    assert design.isna().any().any()
    assert np.isfinite(probabilities.to_numpy()).all()


def test_input_is_not_modified(sample):
    X, y = sample
    before_X = X.copy(deep=True)
    before_y = y.copy(deep=True)

    model = RiskBoostingModel(n_estimators=10, min_child_samples=10).fit(X, y)
    model.predict_bad_probability(X)

    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(y, before_y)


def test_identifier_and_extra_label_do_not_change_predictions(
    sample,
    fitted_model,
):
    X, y = sample
    expected = fitted_model.predict_bad_probability(X)

    changed = X.copy()
    changed["sk_id_curr"] = 999999
    changed["target"] = 1 - y

    actual = fitted_model.predict_bad_probability(changed)
    pd.testing.assert_series_equal(actual, expected)


def test_prediction_does_not_modify_model(sample, fitted_model):
    X, _ = sample
    before_model = fitted_model.estimator_.booster_.model_to_string()
    before_features = list(fitted_model.selected_features_)

    fitted_model.predict_bad_probability(X.iloc[:10])

    assert fitted_model.estimator_.booster_.model_to_string() == before_model
    assert fitted_model.selected_features_ == before_features


def test_wrong_target_order_rejected(sample):
    X, y = sample

    with pytest.raises(ValueError, match="索引或顺序"):
        RiskBoostingModel().fit(X, y.iloc[::-1])


def test_single_class_target_rejected(sample):
    X, y = sample
    constant_target = pd.Series(0, index=y.index)

    with pytest.raises(ValueError, match="同时包含两类"):
        RiskBoostingModel().fit(X, constant_target)


def test_no_usable_features_rejected(sample):
    X, y = sample

    # Every raw field except the identifier is forced to a single value.
    constant = X.copy()
    for col in constant.columns:
        if col != "sk_id_curr":
            constant[col] = constant[col].iloc[0]

    with pytest.raises(ValueError, match="没有可用特征"):
        RiskBoostingModel().fit(constant, y)


def test_unfitted_and_missing_raw_column_rejected(sample, fitted_model):
    X, _ = sample

    with pytest.raises(NotFittedError):
        RiskBoostingModel().predict_proba(X)

    with pytest.raises(ValueError, match="缺少必需字段"):
        fitted_model.predict_proba(X.drop(columns=["amt_credit"]))


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"learning_rate": np.nan},
        {"num_leaves": 1},
        {"min_child_samples": 0},
        {"reg_lambda": -1.0},
        {"n_jobs": 0},
        {"random_state": True},
    ],
)
def test_invalid_parameters_rejected(sample, parameters):
    X, y = sample

    with pytest.raises(ValueError):
        RiskBoostingModel(**parameters).fit(X, y)


def test_actual_iterations_and_importance_are_recorded(fitted_model):
    assert 1 <= fitted_model.actual_iterations_ <= fitted_model.n_estimators
    assert set(fitted_model.gain_importance_.index) == set(
        fitted_model.selected_features_
    )
    assert (
        fitted_model.gain_importance_["training_split_gain"] >= 0
    ).all()


def test_both_pipelines_score_the_same_rows(sample):
    """The comparison is only meaningful on one shared split."""
    X, y = sample

    logistic = RiskLogisticModel(iv_threshold=0.02).fit(X, y)
    boosting = RiskBoostingModel(
        n_estimators=20,
        num_leaves=7,
        min_child_samples=10,
        random_state=7,
    ).fit(X, y)

    logistic_risk = logistic.predict_bad_probability(X)
    boosting_risk = boosting.predict_bad_probability(X)
    constant_risk = pd.Series(logistic.training_bad_rate_, index=X.index)

    pd.testing.assert_index_equal(logistic_risk.index, boosting_risk.index)

    comparison = pd.DataFrame(
        {
            "逻辑回归流程": evaluate_probabilities(y, logistic_risk),
            "梯度提升树流程": evaluate_probabilities(y, boosting_risk),
            "训练比例常数基准": evaluate_probabilities(y, constant_risk),
        }
    ).T

    assert comparison.shape == (3, 9)
    assert np.isfinite(
        comparison[["排序曲线下面积", "布里尔分数", "对数损失"]].to_numpy()
    ).all()
    # The comparison is descriptive: no direction is asserted up front.
    assert comparison.loc["训练比例常数基准", "排序曲线下面积"] == pytest.approx(0.5)


def test_duplicate_detection_compares_missing_positions():
    """Two columns matching on values but not on missing rows are not drops."""
    rng = np.random.default_rng(11)
    n = 120
    index = pd.Index(range(n))
    signal = rng.uniform(0.0, 1.0, size=n)
    labels = (signal < 0.5).astype("int64")

    first = signal.copy()
    first[0] = np.nan
    second = signal.copy()
    second[1] = np.nan  # same values, different missing position

    X = pd.DataFrame(
        {
            "amt_income_total": 100000.0,
            "amt_credit": 200000.0,
            "amt_annuity": 10000.0,
            "days_birth": -10957.5,
            "days_employed": -3652.5,
            "ext_source_1": first,
            "ext_source_2": second,
            "ext_source_3": 0.5,
            "bureau_cnt": 1,
            "bureau_debt_total": 20000.0,
        },
        index=index,
    )
    y = pd.Series(labels, index=index, name="target")

    model = RiskBoostingModel(
        n_estimators=10,
        num_leaves=4,
        min_child_samples=10,
        random_state=3,
    ).fit(X, y)

    assert {"ext_source_1", "ext_source_2"} <= set(
        model.selected_features_
    )
    report = model.selection_report_
    for col in ("ext_source_1", "ext_source_2"):
        assert report.loc[col, "selection_reason"] == "保留"
    assert not report["selection_reason"].str.contains("完全重复").any()
