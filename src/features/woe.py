"""Step 6: weight-of-evidence encoding and information value, fitted on train.

Direction convention used throughout: WOE = ln(bad share / good share), where
"bad" is the sample with target 1 and "good" is the sample with target 0. This
is a modelling convention only; it does not change the dataset's own definition
of the label.

Smoothing adds the same total mass to both classes, so the smoothed bad and good
shares each still sum to one. Bins that never appear in training are handled by
an explicit policy (neutral zero, or an error) and are always reported.
"""
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WOEEncoder(TransformerMixin, BaseEstimator):
    """Weight-of-evidence encoder for already-binned string features.

    Direction: bad share / good share.
    Unknown-bin policy: ``neutral`` encodes zero and reports it; ``error``
    raises instead.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_bin_samples: int = 20,
        unknown_policy: str = "neutral",
    ):
        self.alpha = alpha
        self.min_bin_samples = min_bin_samples
        self.unknown_policy = unknown_policy

    def _validate_parameters(self) -> None:
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("平滑参数必须是有限的正数。")

        if (
            isinstance(self.min_bin_samples, bool)
            or not isinstance(self.min_bin_samples, Integral)
            or self.min_bin_samples < 1
        ):
            raise ValueError("小样本箱阈值必须是正整数。")

        if self.unknown_policy not in {"neutral", "error"}:
            raise ValueError("未知箱策略只能选择中性回退或严格报错。")

    @staticmethod
    def _validate_bins(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("分箱输入必须是数据表。")
        if X.empty:
            raise ValueError("分箱输入不能为空。")
        if not X.columns.is_unique:
            raise ValueError("特征列名不能重复。")
        if not X.index.is_unique:
            raise ValueError("样本索引不能重复。")
        if not all(isinstance(col, str) for col in X.columns):
            raise ValueError("特征列名必须是字符串。")

        forbidden = {"target", "sk_id_curr"} & set(X.columns)
        if forbidden:
            raise ValueError(
                f"标签或编号不能进入编码器：{sorted(forbidden)}"
            )

        out = pd.DataFrame(index=X.index)

        for col in X.columns:
            values = X[col]

            if values.isna().any():
                raise ValueError(
                    f"列 {col} 含空值，请先转换成明确的缺失箱。"
                )

            if not values.map(lambda value: isinstance(value, str)).all():
                raise ValueError(
                    f"列 {col} 含非字符串值，请先完成分箱。"
                )

            if values.str.strip().eq("").any():
                raise ValueError(f"列 {col} 含空白箱名。")

            out[col] = values.astype("string")

        return out

    @staticmethod
    def _validate_target(y: pd.Series, index: pd.Index) -> pd.Series:
        if not isinstance(y, pd.Series):
            raise TypeError("标签必须是带索引的序列。")

        if not y.index.equals(index):
            raise ValueError("标签与特征的索引或顺序不一致。")

        if y.isna().any():
            raise ValueError("标签不能包含缺失值。")

        if not y.isin([0, 1]).all():
            raise ValueError("标签只能取零或一。")

        if y.nunique() != 2:
            raise ValueError("训练标签必须同时包含零和一。")

        return y.astype("int64")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn the bin -> WOE mapping and IV table from training data only."""
        self._validate_parameters()
        data = self._validate_bins(X)
        target = self._validate_target(y, data.index)

        total_bad = int(target.sum())
        total_good = len(target) - total_bad
        alpha = float(self.alpha)

        tables = {}
        mappings = {}
        reports = []

        for col in data.columns:
            pairs = pd.DataFrame(
                {
                    "bin": data[col],
                    "bad": target,
                },
                index=data.index,
            )

            table = (
                pairs.groupby("bin", observed=True, sort=True)["bad"]
                .agg(["size", "sum"])
                .rename(columns={"size": "n_samples", "sum": "n_bad"})
            )

            table["n_good"] = table["n_samples"] - table["n_bad"]
            table["bad_rate"] = table["n_bad"] / table["n_samples"]

            n_bins = len(table)
            bad_denominator = total_bad + alpha * n_bins
            good_denominator = total_good + alpha * n_bins

            if not np.isfinite(
                [bad_denominator, good_denominator]
            ).all():
                raise ValueError("平滑参数造成数值溢出，请调整参数。")

            table["bad_share"] = (
                table["n_bad"] + alpha
            ) / bad_denominator

            table["good_share"] = (
                table["n_good"] + alpha
            ) / good_denominator

            shares = table[["bad_share", "good_share"]].to_numpy()
            if (shares <= 0).any() or not np.isfinite(shares).all():
                raise ValueError("平滑后的占比无效，请检查数值范围。")

            # Subtract two logs instead of taking the ratio first, to avoid an
            # extra overflow risk.
            table["woe"] = (
                np.log(table["bad_share"])
                - np.log(table["good_share"])
            )
            table["iv_component"] = (
                table["bad_share"] - table["good_share"]
            ) * table["woe"]

            table["small_bin"] = (
                table["n_samples"] < self.min_bin_samples
            )
            table["single_class_bin"] = (
                table["n_bad"].eq(0) | table["n_good"].eq(0)
            )

            iv = float(table["iv_component"].sum())

            tables[col] = table
            mappings[col] = table["woe"].to_dict()
            reports.append(
                {
                    "feature": col,
                    "iv": iv,
                    "n_bins": n_bins,
                    "small_bins": int(table["small_bin"].sum()),
                    "single_class_bins": int(
                        table["single_class_bin"].sum()
                    ),
                    "review_high_iv": iv >= 0.5,
                }
            )

        # Update the fitted state as a whole, after every column succeeded.
        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.n_features_in_ = data.shape[1]
        self.n_samples_seen_ = len(data)
        self.tables_ = tables
        self.mapping_ = mappings
        self.iv_report_ = (
            pd.DataFrame(reports)
            .set_index("feature")
            .sort_values("iv", ascending=False, kind="stable")
        )

        return self

    def _align_input(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            ["feature_names_in_", "mapping_", "iv_report_"],
        )
        data = self._validate_bins(X)
        expected = list(self.feature_names_in_)

        if set(data.columns) != set(expected):
            missing = sorted(set(expected) - set(data.columns))
            extra = sorted(set(data.columns) - set(expected))
            raise ValueError(
                f"输入结构变化；缺少列：{missing}；新增列：{extra}"
            )

        return data.loc[:, expected]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the training mapping; the label is not read or used."""
        data = self._align_input(X)
        out = pd.DataFrame(index=data.index)

        for col in data.columns:
            encoded = data[col].map(self.mapping_[col])
            unknown = encoded.isna()

            if unknown.any() and self.unknown_policy == "error":
                raise ValueError(
                    f"列 {col} 出现 {int(unknown.sum())} 条未知箱记录。"
                )

            out[col] = encoded.fillna(0.0).astype("float64")

        return out

    def unknown_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Count bins unseen during training; the fitted state is unchanged."""
        data = self._align_input(X)
        rows = []

        for col in data.columns:
            known_bins = list(self.mapping_[col])
            unknown = ~data[col].isin(known_bins)

            rows.append(
                {
                    "feature": col,
                    "n_samples": len(data),
                    "unknown_count": int(unknown.sum()),
                    "unknown_rate": float(unknown.mean()),
                }
            )

        return pd.DataFrame(rows).set_index("feature")


def candidate_features_by_iv(
    encoder: WOEEncoder,
    iv_threshold: float = 0.02,
) -> list[str]:
    """Initial IV screen; a candidate list, not a final variable selection."""
    if iv_threshold < 0:
        raise ValueError("信息价值门槛不能为负数。")
    selected = encoder.iv_report_.index[
        encoder.iv_report_["iv"] >= iv_threshold
    ].tolist()
    if not selected:
        raise ValueError(
            "没有变量达到当前初筛门槛，请检查数据和分箱，"
            "不要直接用空特征集训练模型。"
        )
    return selected


def save_woe_reports(
    encoder: WOEEncoder,
    bins: pd.DataFrame,
    out_dir: str | Path,
    candidate_features: list[str] | None = None,
) -> dict[str, Path]:
    """Write the auditable Step 6 reports (IV, bin detail, unknown bins)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    iv_path = out_dir / "iv_report.csv"
    encoder.iv_report_.to_csv(iv_path, encoding="utf-8-sig")
    paths["iv_report"] = iv_path

    bin_path = out_dir / "bin_details.csv"
    pd.concat(encoder.tables_, names=["feature", "bin"]).to_csv(
        bin_path, encoding="utf-8-sig")
    paths["bin_details"] = bin_path

    unknown_path = out_dir / "unknown_bins.csv"
    encoder.unknown_report(bins).to_csv(unknown_path, encoding="utf-8-sig")
    paths["unknown_bins"] = unknown_path

    if candidate_features is not None:
        cand_path = out_dir / "candidate_features.csv"
        pd.Series(list(candidate_features), name="feature").to_csv(
            cand_path, index=False, encoding="utf-8-sig")
        paths["candidate_features"] = cand_path

    return paths
