"""Step 5 tests: business-rule features and train-only binning."""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.features.engineering import (
    EMPLOYED_SENTINEL,
    TrainQuantileBinner,
    build_business_features,
)


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "sk_id_curr": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "amt_income_total": [100.0, 0.0, np.nan, 200.0],
            "amt_credit": [200.0, 100.0, 50.0, -20.0],
            "amt_annuity": [10.0, 20.0, np.nan, 25.0],
            "days_birth": [-7305.0, -10957.5, -14610.0, -18262.5],
            "days_employed": [
                -3652.5, EMPLOYED_SENTINEL, np.nan, 10.0
            ],
            "ext_source_1": [0.2, 0.4, np.nan, 0.8],
            "ext_source_2": [0.3, 0.5, 0.6, 0.9],
            "ext_source_3": [0.1, np.nan, 0.7, 0.8],
            "bureau_cnt": [2, 0, 1, 1],
            "bureau_debt_total": [50.0, np.nan, 0.0, -5.0],
        },
        index=[101, 102, 103, 104],
    )


def test_features_preserve_input_and_exclude_metadata(raw):
    before = raw.copy(deep=True)
    out = build_business_features(raw)

    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_index_equal(out.index, raw.index)

    assert "target" not in out.columns
    assert "sk_id_curr" not in out.columns


def test_special_value_and_invalid_employment_are_distinct(raw):
    out = build_business_features(raw)

    assert out.loc[102, "days_employed_sentinel"] == 1
    assert out.loc[102, "days_employed_invalid"] == 0
    assert pd.isna(out.loc[102, "employed_years"])

    assert out.loc[104, "days_employed_sentinel"] == 0
    assert out.loc[104, "days_employed_invalid"] == 1
    assert pd.isna(out.loc[104, "employed_years"])


def test_year_conversion_and_safe_ratios(raw):
    out = build_business_features(raw)

    assert out.loc[101, "age_years"] == pytest.approx(20.0)
    assert out.loc[101, "employed_years"] == pytest.approx(10.0)
    assert out.loc[101, "credit_to_income_proxy"] == pytest.approx(2.0)

    # Zero or missing income must not produce a valid ratio.
    assert pd.isna(out.loc[102, "credit_to_income_proxy"])
    assert pd.isna(out.loc[103, "credit_to_income_proxy"])

    # A negative loan amount is flagged first and never enters a ratio.
    assert out.loc[104, "amt_credit_invalid"] == 1
    assert pd.isna(out.loc[104, "credit_to_income_proxy"])
    assert not np.isinf(out.to_numpy()).any()


def test_no_record_does_not_mean_zero_debt(raw):
    out = build_business_features(raw)

    assert out.loc[102, "bureau_has_record"] == 0
    assert pd.isna(out.loc[102, "bureau_debt_total"])

    assert out.loc[103, "bureau_has_record"] == 1
    assert out.loc[103, "bureau_debt_total"] == 0


def test_missing_required_column_raises(raw):
    with pytest.raises(ValueError, match="缺少必需字段"):
        build_business_features(raw.drop(columns=["amt_credit"]))


def test_validation_cannot_change_training_edges():
    train = pd.DataFrame({"income": np.arange(100, dtype=float)})
    valid = pd.DataFrame({"income": [-1000.0, 1e9, np.nan]})

    binner = TrainQuantileBinner(n_bins=5).fit(train)
    before = binner.edges_["income"].copy()
    out = binner.transform(valid)

    np.testing.assert_array_equal(binner.edges_["income"], before)
    assert out["income"].tolist() == [
        "数值箱_000", "数值箱_004", "缺失箱"
    ]


def test_binary_feature_stays_separated():
    train = pd.DataFrame({"flag": [0.0] * 99 + [1.0]})
    out = TrainQuantileBinner(n_bins=5).fit_transform(train)

    assert out.loc[0, "flag"] != out.loc[99, "flag"]


def test_all_missing_and_constant_columns():
    train = pd.DataFrame({
        "empty": [np.nan, np.nan],
        "constant": [5.0, 5.0],
    })
    valid = pd.DataFrame({
        "empty": [np.nan, 3.0],
        "constant": [5.0, 100.0],
    })

    out = TrainQuantileBinner().fit(train).transform(valid)

    assert out["empty"].tolist() == ["缺失箱", "训练外非缺失箱"]
    assert out["constant"].tolist() == ["数值箱_000", "数值箱_000"]


def test_boundary_is_right_closed():
    train = pd.DataFrame({"value": [0.0, 10.0, 20.0, 30.0]})
    binner = TrainQuantileBinner(n_bins=2).fit(train)

    boundary = binner.edges_["value"][0]
    valid = pd.DataFrame({
        "value": [boundary, np.nextafter(boundary, np.inf)]
    })
    out = binner.transform(valid)

    assert out["value"].tolist() == ["数值箱_000", "数值箱_001"]


def test_refit_replaces_previous_state():
    binner = TrainQuantileBinner().fit(
        pd.DataFrame({"old": [1.0, 2.0]})
    )
    binner.fit(pd.DataFrame({"new": [10.0, 20.0]}))

    assert set(binner.edges_) == {"new"}


def test_schema_change_and_unfitted_use_raise():
    data = pd.DataFrame({"value": [1.0, 2.0]})

    with pytest.raises(NotFittedError):
        TrainQuantileBinner().transform(data)

    binner = TrainQuantileBinner().fit(data)
    with pytest.raises(ValueError, match="输入结构发生变化"):
        binner.transform(pd.DataFrame({"other": [1.0]}))


def test_infinite_input_and_label_column_are_rejected():
    with pytest.raises(ValueError, match="无穷"):
        TrainQuantileBinner().fit(
            pd.DataFrame({"value": [1.0, np.inf]})
        )

    with pytest.raises(ValueError, match="标签或编号"):
        TrainQuantileBinner().fit(
            pd.DataFrame({"target": [0.0, 1.0]})
        )
