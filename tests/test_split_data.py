"""Step 3 tests: correctness and leakage discipline of the data splits."""
import numpy as np
import pandas as pd
import pytest

from src.data_layer.split_data import stratified_split, temporal_split


@pytest.fixture
def mock_df():
    """Mock dataset with an ID, a feature, a timestamp and the target."""
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        "sk_id_curr": range(n),
        "feature_a": rng.normal(size=n),
        "time_col": pd.date_range("2020-01-01", periods=n, freq="D"),
        "target": rng.integers(0, 2, size=n),
    })


def test_stratified_split_sizes(mock_df):
    """The three splits cover the data exactly once, at the requested sizes."""
    split = stratified_split(mock_df, valid_size=0.2, oot_size=0.2,
                             random_state=42)
    total = len(split.y_train) + len(split.y_valid) + len(split.y_oot)
    assert total == len(mock_df)
    assert abs(len(split.y_oot) / len(mock_df) - 0.2) < 0.01
    assert abs(len(split.y_valid) / len(mock_df) - 0.2) < 0.01


def test_stratified_split_preserves_bad_rate(mock_df):
    """Stratification keeps the bad rate close across the three splits."""
    split = stratified_split(mock_df, random_state=42)
    rates = [split.y_train.mean(), split.y_valid.mean(), split.y_oot.mean()]
    assert max(rates) - min(rates) < 0.05


def test_stratified_split_no_overlap(mock_df):
    """Leakage guard: the three splits must not share any customer."""
    split = stratified_split(mock_df, random_state=42)
    train_ids = set(split.X_train["sk_id_curr"])
    valid_ids = set(split.X_valid["sk_id_curr"])
    oot_ids = set(split.X_oot["sk_id_curr"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(oot_ids)
    assert valid_ids.isdisjoint(oot_ids)


def test_stratified_split_keeps_id_but_not_target(mock_df):
    """The ID is kept for traceability; the target never appears in X."""
    split = stratified_split(mock_df, random_state=42)
    assert "sk_id_curr" in split.X_train.columns
    assert "target" not in split.X_train.columns


def test_temporal_split_no_time_travel(mock_df):
    """Train < Valid < OOT in time, verified from the recorded boundaries."""
    split = temporal_split(mock_df, time_col="time_col",
                           valid_size=0.2, oot_size=0.2, verbose=False)
    ranges = split.meta["time_ranges"]
    assert ranges["train"][1] <= ranges["valid"][0]
    assert ranges["valid"][1] <= ranges["oot"][0]


def test_temporal_split_sizes_and_feature_columns(mock_df):
    """Sizes are conserved and the timestamp is not used as a feature."""
    split = temporal_split(mock_df, time_col="time_col", verbose=False)
    total = len(split.y_train) + len(split.y_valid) + len(split.y_oot)
    assert total == len(mock_df)
    assert "time_col" not in split.X_train.columns
