"""Step 3: data splitting into train / validation / test sets.

Two modes are supported:

* ``stratified`` — stratified random holdout, for datasets without an
  application timestamp (Home Credit). This is **not** an out-of-time test and
  is not described as one.
* ``temporal`` — true out-of-time split, for datasets with a timestamp
  (e.g. LendingClub ``issue_d``). Past data trains, the most recent data tests.

The stratified splitter uses ``numpy`` only, so the data layer keeps a light
dependency footprint; it reproduces the class ratios of
``sklearn.model_selection.train_test_split(..., stratify=y)``.

Core discipline: the OOT set is only opened once, at final evaluation. All
tuning, feature selection and threshold choice must use train/validation only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    """Container for the three splits, so they cannot be passed around mixed up.

    ``meta`` records auditable facts about the split (for example the time
    boundaries of a temporal split) without leaking them into the features.
    """

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    meta: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Size and bad rate of each split, for sanity checks."""
        rows = []
        for name, y in [
            ("train", self.y_train),
            ("valid", self.y_valid),
            ("test", self.y_test),
        ]:
            rows.append({
                "dataset": name,
                "n_samples": len(y),
                "bad_rate": round(float(y.mean()), 4),
            })
        return pd.DataFrame(rows)


def _stratified_take(y: np.ndarray, size: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (taken, kept) while preserving class ratios.

    A class with fewer than two members is always kept, so that rare classes do
    not create empty splits.
    """
    taken: list[np.ndarray] = []
    kept: list[np.ndarray] = []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) < 2:
            kept.append(idx)
            continue
        n_take = int(round(size * len(idx)))
        n_take = min(max(n_take, 1), len(idx) - 1)
        taken.append(idx[:n_take])
        kept.append(idx[n_take:])

    taken_idx = (np.concatenate(taken) if taken
                 else np.array([], dtype=int))
    kept_idx = (np.concatenate(kept) if kept
                else np.array([], dtype=int))
    rng.shuffle(taken_idx)
    rng.shuffle(kept_idx)
    return taken_idx, kept_idx


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "target",
    id_col: str = "sk_id_curr",
    valid_size: float = 0.2,
    oot_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """Stratified random split (Home Credit mode).

    The OOT set is cut first, then the validation set is cut from what remains,
    so each split keeps the overall bad rate.

    The ID column is kept in ``X`` for traceability; it must be dropped before
    model training (see Step 7).
    """
    if valid_size + oot_size >= 1.0:
        raise ValueError("valid_size + oot_size must be below 1.0")
    if target_col not in df.columns:
        raise KeyError(f"target column not found: {target_col}")

    rng = np.random.default_rng(random_state)
    y_all = df[target_col].to_numpy()

    test_idx, rest_idx = _stratified_take(y_all, oot_size, rng)

    # valid_size is stated relative to the full dataset, so rescale it to the
    # remaining data after OOT has been removed.
    valid_ratio_in_rest = valid_size / (1.0 - oot_size)
    valid_rel, train_rel = _stratified_take(y_all[rest_idx],
                                            valid_ratio_in_rest, rng)
    train_idx, valid_idx = rest_idx[train_rel], rest_idx[valid_rel]

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    return DataSplit(
        X_train=X.iloc[train_idx], X_valid=X.iloc[valid_idx],
        X_test=X.iloc[test_idx],
        y_train=y.iloc[train_idx], y_valid=y.iloc[valid_idx],
        y_test=y.iloc[test_idx],
        meta={
            "mode": "stratified_holdout",
            "id_col": id_col,
            "random_state": random_state,
            "limitation": (
                "Home Credit has no application timestamp, so this is a "
                "stratified random holdout, not an out-of-time test."
            ),
        },
    )


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    target_col: str = "target",
    valid_size: float = 0.2,
    oot_size: float = 0.2,
    verbose: bool = True,
) -> DataSplit:
    """Out-of-time split: earliest data trains, latest data is the OOT test.

    The time boundaries are recorded in ``meta['time_ranges']`` and the split
    asserts that no time travel occurs (train max <= OOT min).
    """
    if valid_size + oot_size >= 1.0:
        raise ValueError("valid_size + oot_size must be below 1.0")
    for col in (time_col, target_col):
        if col not in df.columns:
            raise KeyError(f"column not found: {col}")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * (1.0 - valid_size - oot_size))
    valid_end = int(n * (1.0 - oot_size))

    train_df = df_sorted.iloc[:train_end]
    valid_df = df_sorted.iloc[train_end:valid_end]
    oot_df = df_sorted.iloc[valid_end:]

    if len(train_df) == 0 or len(valid_df) == 0 or len(oot_df) == 0:
        raise ValueError("split sizes must all be non-empty")

    time_ranges = {
        "train": (train_df[time_col].min(), train_df[time_col].max()),
        "valid": (valid_df[time_col].min(), valid_df[time_col].max()),
        "oot": (oot_df[time_col].min(), oot_df[time_col].max()),
    }

    # No time travel: training data must be strictly in the past of the OOT set.
    assert time_ranges["train"][1] <= time_ranges["oot"][0], "time travel detected"

    if verbose:
        for name, (start, end) in time_ranges.items():
            print(f"[时间划分] {name:5s}: {start} ~ {end}")

    feature_cols = [c for c in df.columns if c not in (target_col, time_col)]
    return DataSplit(
        X_train=train_df[feature_cols],
        X_valid=valid_df[feature_cols],
        X_test=oot_df[feature_cols],
        y_train=train_df[target_col],
        y_valid=valid_df[target_col],
        y_test=oot_df[target_col],
        meta={"mode": "temporal_oot", "time_col": time_col,
              "time_ranges": time_ranges},
    )
