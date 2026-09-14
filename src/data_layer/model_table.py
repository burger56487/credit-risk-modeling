"""Load the modelling table produced by the Step 2 SQL layer.

Both experiment runners read the same table, so the loading rules and the
integrity guards live here instead of being copied into each script.
"""
import hashlib
from pathlib import Path

import pandas as pd

from src.features.engineering import REQUIRED_COLUMNS


def file_digest(path: Path) -> str:
    """SHA-256 of the input file, so a run can be traced to a data version."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_table(path: Path) -> pd.DataFrame:
    """Read the modelling table and check the columns both models need."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到建模表：{path}。请先完成第二步的数据库聚合，"
            "或把模型表导出为 CSV。"
        )

    frame = (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path)
    )

    required = {"sk_id_curr", "target", *REQUIRED_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"建模表缺少字段：{missing}")

    ids = frame["sk_id_curr"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("建模表申请编号缺失或重复。")

    return frame


def check_split_ids(split) -> None:
    """Application IDs must not overlap; this is not applicant de-duplication."""
    train_ids = set(split.X_train["sk_id_curr"])
    valid_ids = set(split.X_valid["sk_id_curr"])

    if not train_ids.isdisjoint(valid_ids):
        raise ValueError("训练集和验证集存在重复申请编号。")
