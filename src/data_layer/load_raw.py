"""Step 1: raw data loading and integrity checks.

This module only reads the raw CSVs and performs basic validation. It does not
clean or transform the data; cleaning happens in Step 2 and later.
"""
from pathlib import Path

import pandas as pd

# Project root / data/raw
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Expected raw files, used for the integrity check.
EXPECTED_FILES = [
    "application_train.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]


def check_files_exist(data_dir: Path = RAW_DATA_DIR) -> None:
    """Raise FileNotFoundError listing any expected raw file that is missing."""
    missing = [f for f in EXPECTED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"以下原始数据文件缺失：{missing}\n"
            f"请先从 Kaggle 下载并放入 {data_dir}"
        )
    print(f"[OK] 全部 {len(EXPECTED_FILES)} 个原始文件就位。")


def load_main_table(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load the main application table and print basic data-quality diagnostics."""
    df = pd.read_csv(data_dir / "application_train.csv")

    print(f"[主表] 形状：{df.shape[0]} 行 × {df.shape[1]} 列")

    target_dist = df["TARGET"].value_counts(normalize=True).sort_index()
    bad_rate = target_dist.get(1, 0.0)
    print(f"[目标] 违约率（TARGET=1 占比）：{bad_rate:.4f}")
    print(f"[目标] 正负样本比约 1:{(1 - bad_rate) / bad_rate:.1f}")

    missing_rate = df.isnull().mean().sort_values(ascending=False)
    print("\n[缺失率最高的前 10 列]")
    print(missing_rate.head(10).to_string())

    return df


if __name__ == "__main__":
    check_files_exist()
    load_main_table()
