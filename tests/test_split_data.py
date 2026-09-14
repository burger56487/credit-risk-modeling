"""Step 16 tests for the revised splitter: membership, ids and immutability."""
import pandas as pd
import pytest

from src.data_layer.split_data import stratified_split


@pytest.fixture
def data():
    n = 120
    return pd.DataFrame(
        {
            "sk_id_curr": range(1000, 1000 + n),
            "feature": range(n),
            "target": [0, 0, 0, 1] * 30,
        }
    )


def test_partitions_are_disjoint_and_complete(data):
    result = stratified_split(data)

    groups = [
        set(result.X_train.index),
        set(result.X_valid.index),
        set(result.X_test.index),
    ]

    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(data["sk_id_curr"])

    for X, y in (
        (result.X_train, result.y_train),
        (result.X_valid, result.y_valid),
        (result.X_test, result.y_test),
    ):
        pd.testing.assert_index_equal(X.index, y.index)
        assert "target" not in X.columns
        assert set(y.unique()) == {0, 1}


def test_input_order_does_not_change_membership(data):
    first = stratified_split(data)
    second = stratified_split(data.sample(frac=1, random_state=9))

    pd.testing.assert_frame_equal(first.membership(), second.membership())


def test_duplicate_application_is_rejected(data):
    changed = data.copy()
    changed.loc[1, "sk_id_curr"] = changed.loc[0, "sk_id_curr"]

    with pytest.raises(ValueError, match="重复"):
        stratified_split(changed)


def test_invalid_ratio_is_rejected(data):
    with pytest.raises(ValueError, match="正比例"):
        stratified_split(data, valid_size=0.6, test_size=0.5)


def test_input_is_not_modified(data):
    before = data.copy(deep=True)
    stratified_split(data)
    pd.testing.assert_frame_equal(data, before)


def test_application_ids_must_be_integers_and_unique(data):
    with pytest.raises(ValueError, match="整数类型"):
        stratified_split(data.assign(sk_id_curr=data["sk_id_curr"].astype(str)))

    with pytest.raises(ValueError, match="缺失"):
        stratified_split(data.assign(sk_id_curr=[None] * len(data)))


def test_single_class_target_is_rejected(data):
    with pytest.raises(ValueError, match="同时存在两类标签"):
        stratified_split(data.assign(target=0))


def test_membership_lists_every_application_once(data):
    membership = stratified_split(data).membership()

    assert set(membership.columns) == {"application_id", "partition"}
    assert len(membership) == len(data)
    assert membership["application_id"].is_unique
    assert set(membership["partition"]) == {"训练", "验证", "测试"}


def test_stratified_ratios_are_respected(data):
    result = stratified_split(data, valid_size=0.25, test_size=0.25)

    assert len(result.X_test) == 30
    assert len(result.X_valid) == 30
    assert len(result.X_train) == 60

    # Stratification keeps the class ratio only up to rounding on small sets.
    for y in (result.y_train, result.y_valid, result.y_test):
        assert y.mean() == pytest.approx(0.25, abs=0.05)
        assert 1 <= int(y.sum()) < len(y)
