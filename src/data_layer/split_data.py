"""Stratified random train/valid/test splitting; no out-of-time claim.

This module deliberately offers only a stratified random holdout. An earlier
version also exposed a "temporal" split, but this dataset has no reliable
application timestamp, so a time-based split could not be validated and the
interface was removed rather than maintained as an unverified capability.

Membership is decided on the application id, not on row position:

* ids must be unique, integer and never missing;
* rows are sorted by id before splitting, so the same input gives the same
  partition regardless of the order it arrives in;
* the split index is the application id;
* the three partitions are checked to be pairwise disjoint and jointly complete.

Unique application ids still do not prove that applicants are independent; a
future customer identifier would need grouped splitting.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series

    def membership(self) -> pd.DataFrame:
        """Return application-to-partition membership, without labels."""
        rows = []

        for name, frame in (
            ("训练", self.X_train),
            ("验证", self.X_valid),
            ("测试", self.X_test),
        ):
            rows.extend(
                {"application_id": int(identifier), "partition": name}
                for identifier in frame.index
            )

        return (
            pd.DataFrame(rows)
            .sort_values("application_id")
            .reset_index(drop=True)
        )


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "target",
    id_col: str = "sk_id_curr",
    valid_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """Stratified random split whose membership does not depend on row order."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("划分输入必须是非空数据表。")
    if not df.columns.is_unique:
        raise ValueError("输入字段名不能重复。")
    if target_col == id_col:
        raise ValueError("标签列与申请编号列不能相同。")

    missing = {target_col, id_col} - set(df.columns)
    if missing:
        raise ValueError(f"缺少划分字段：{sorted(missing)}")

    for name, value in (
        ("验证集比例", valid_size),
        ("测试集比例", test_size),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name}必须严格位于零与一之间。")

    if valid_size + test_size >= 1:
        raise ValueError("训练集必须保留正比例样本。")

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or not 0 <= random_state < 2**32
    ):
        raise ValueError("随机种子必须是规定范围内的非负整数。")

    identifiers = df[id_col]

    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("申请编号不能缺失或重复。")
    if (
        is_bool_dtype(identifiers.dtype)
        or not is_integer_dtype(identifiers.dtype)
    ):
        raise ValueError("本数据契约要求申请编号使用整数类型。")

    target = df[target_col]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("标签必须是没有缺失的零或一。")
    if target.nunique() != 2:
        raise ValueError("分层划分需要同时存在两类标签。")

    # The input is never modified: order by id and use the id as the index.
    ordered = (
        df.sort_values(id_col, kind="stable")
        .set_index(id_col, drop=False)
        .copy()
    )

    try:
        remaining, test = train_test_split(
            ordered,
            test_size=float(test_size),
            stratify=ordered[target_col],
            random_state=int(random_state),
        )

        train, valid = train_test_split(
            remaining,
            test_size=float(valid_size) / (1.0 - float(test_size)),
            stratify=remaining[target_col],
            random_state=int(random_state),
        )
    except ValueError as exc:
        raise ValueError(
            "当前样本数量、类别数量与比例无法完成分层划分。"
        ) from exc

    partitions = [
        frame.sort_index().copy() for frame in (train, valid, test)
    ]

    if any(
        frame.empty or frame[target_col].nunique() != 2
        for frame in partitions
    ):
        raise ValueError("划分后每个集合都必须同时包含两类样本。")

    id_sets = [set(frame.index) for frame in partitions]

    if any(
        id_sets[left] & id_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("划分结果存在重复申请。")

    if set.union(*id_sets) != set(ordered.index):
        raise RuntimeError("划分结果没有完整覆盖输入申请。")

    feature_frames = [
        frame.drop(columns=[target_col]).copy() for frame in partitions
    ]
    targets = [
        frame[target_col].astype("int64").copy() for frame in partitions
    ]

    return DataSplit(
        X_train=feature_frames[0],
        X_valid=feature_frames[1],
        X_test=feature_frames[2],
        y_train=targets[0],
        y_valid=targets[1],
        y_test=targets[2],
    )
