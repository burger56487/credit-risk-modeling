"""Step 5: fixed business-rule features and train-only numeric binning.

Two different kinds of operation live here:

* Row-wise rules (sentinel detection, negative-amount handling, ratio
  calculation) are fixed in advance and use no learned statistics.
* Quantile bin edges are *learned parameters*: they are fitted on the training
  split only and then applied unchanged to validation and the holdout set.

Missing values are preserved (never median-imputed); they become their own bin
so the scorecard branch can treat missingness explicitly.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


EMPLOYED_SENTINEL = 365243
DAYS_PER_YEAR = 365.25

# Only these raw fields may enter this module, so IDs and the label cannot leak
# into the feature matrix by accident.
REQUIRED_COLUMNS = (
    "amt_income_total",
    "amt_credit",
    "amt_annuity",
    "days_birth",
    "days_employed",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "bureau_cnt",
    "bureau_debt_total",
)


def validate_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate structure and cast to numeric; keep missing, reject infinities."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是数据表。")
    if df.empty:
        raise ValueError("输入不能没有行或没有列。")
    if not df.columns.is_unique:
        raise ValueError("列名不能重复。")
    if not df.index.is_unique:
        raise ValueError("行索引不能重复，请先修复数据索引。")
    if not all(isinstance(col, str) for col in df.columns):
        raise ValueError("所有列名必须是字符串。")

    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        try:
            values = pd.to_numeric(df[col], errors="raise")
            if np.iscomplexobj(values.to_numpy()):
                raise ValueError("不接受复数。")
            out[col] = values.astype("float64")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"列 {col} 含有不能接受的数值。") from exc

    if np.isinf(out.to_numpy()).any():
        raise ValueError("输入包含正无穷或负无穷，请检查上游计算。")

    return out


def build_business_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive business features from raw (un-imputed) columns.

    No statistic is fitted here, the label is not read, and the input object is
    never modified. Missing values are kept for the scorecard missing bin.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("输入必须是数据表。")
    if not raw.columns.is_unique:
        raise ValueError("原始数据列名不能重复。")

    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"缺少必需字段：{missing}")

    x = validate_numeric_frame(raw.loc[:, list(REQUIRED_COLUMNS)])
    out = pd.DataFrame(index=x.index)

    # Amount fields: mark original missing and negative values separately; do
    # not fill missing with zero.
    amount_columns = (
        "amt_income_total",
        "amt_credit",
        "amt_annuity",
        "bureau_debt_total",
    )

    for col in amount_columns:
        invalid = x[col].lt(0)
        out[col] = x[col].mask(invalid)
        out[f"{col}_missing"] = x[col].isna().astype("int8")
        out[f"{col}_invalid"] = invalid.astype("int8")

    # Birth must precede the application; no hard age limits are imposed.
    birth_invalid = x["days_birth"].ge(0)
    birth = x["days_birth"].mask(birth_invalid)

    out["days_birth_missing"] = x["days_birth"].isna().astype("int8")
    out["days_birth_invalid"] = birth_invalid.astype("int8")
    out["age_years"] = -birth / DAYS_PER_YEAR

    # The special code and other invalid positive values are recorded
    # separately; the special code is not interpreted as "unemployed".
    employed = x["days_employed"]
    sentinel = employed.eq(EMPLOYED_SENTINEL)
    employed_invalid = employed.gt(0) & ~sentinel

    out["days_employed_missing"] = employed.isna().astype("int8")
    out["days_employed_sentinel"] = sentinel.astype("int8")
    out["days_employed_invalid"] = employed_invalid.astype("int8")
    out["employed_years"] = (
        -employed.mask(sentinel | employed_invalid) / DAYS_PER_YEAR
    )

    # Employment longer than age is a consistency flag, not something to
    # silently correct.
    out["employment_age_inconsistent"] = (
        out["employed_years"].gt(out["age_years"])
    ).astype("int8")

    # External standardised scores: keep values inside the documented range.
    for col in ("ext_source_1", "ext_source_2", "ext_source_3"):
        invalid = x[col].lt(0) | x[col].gt(1)
        out[col] = x[col].mask(invalid)
        out[f"{col}_missing"] = x[col].isna().astype("int8")
        out[f"{col}_invalid"] = invalid.astype("int8")

    # Upstream aggregation contract: "no matching record" is already encoded as
    # a count of zero.
    count = x["bureau_cnt"]
    if (
        count.isna().any()
        or count.lt(0).any()
        or count.mod(1).ne(0).any()
    ):
        raise ValueError("征信记录数必须是非缺失的非负整数。")

    out["bureau_cnt"] = count
    out["bureau_has_record"] = count.gt(0).astype("int8")

    # Only strictly positive income may be used as a denominator.
    income = out["amt_income_total"]
    denominator = income.where(income.gt(0))
    out["income_zero"] = income.eq(0).astype("int8")

    ratio_sources = {
        "credit_to_income_proxy": "amt_credit",
        "annuity_to_income_proxy": "amt_annuity",
        "bureau_debt_to_income_proxy": "bureau_debt_total",
    }

    for feature, numerator in ratio_sources.items():
        out[feature] = out[numerator].div(denominator)

    return validate_numeric_frame(out)


class TrainQuantileBinner(TransformerMixin, BaseEstimator):
    """Numeric binner whose edges are learned on fit data only.

    Missing values get their own bin. Low-cardinality variables keep their
    distinct values separate, so binary flags are not merged. High-cardinality
    variables are binned on training quantiles.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y=None):
        """Learn bin edges. The label is not used."""
        if (
            isinstance(self.n_bins, bool)
            or not isinstance(self.n_bins, int)
            or self.n_bins < 2
        ):
            raise ValueError("分箱数必须是大于等于二的整数。")

        data = validate_numeric_frame(X)

        forbidden = {"target", "sk_id_curr"} & set(data.columns)
        if forbidden:
            raise ValueError(f"标签或编号不能进入分箱器：{sorted(forbidden)}")

        # Compute into local variables first, then update state as a whole.
        edges = {}
        all_missing = {}
        ranges = {}

        for col in data.columns:
            observed = data[col].dropna().to_numpy()
            unique = np.unique(observed)

            all_missing[col] = len(unique) == 0

            if len(unique) == 0:
                cuts = np.array([], dtype="float64")
                ranges[col] = (None, None)

            elif len(unique) <= self.n_bins:
                # Right-closed intervals: for a binary flag the cut is zero, so
                # zero lands in the first bin and one in the second.
                cuts = unique[:-1]
                ranges[col] = (float(unique[0]), float(unique[-1]))

            else:
                probabilities = np.linspace(
                    0.0, 1.0, self.n_bins + 1
                )[1:-1]
                cuts = np.unique(np.quantile(observed, probabilities))

                # With many ties the effective number of bins can be lower.
                cuts = cuts[
                    (cuts > unique[0]) & (cuts < unique[-1])
                ]
                ranges[col] = (float(unique[0]), float(unique[-1]))

            edges[col] = np.asarray(cuts, dtype="float64")

        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.n_features_in_ = data.shape[1]
        self.edges_ = edges
        self.all_missing_ = all_missing
        self.training_ranges_ = ranges

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned edges without recomputing any quantile."""
        check_is_fitted(
            self,
            ["feature_names_in_", "edges_", "all_missing_"],
        )

        data = validate_numeric_frame(X)
        expected = list(self.feature_names_in_)

        if set(data.columns) != set(expected):
            missing = sorted(set(expected) - set(data.columns))
            extra = sorted(set(data.columns) - set(expected))
            raise ValueError(
                f"输入结构发生变化；缺少列：{missing}；新增列：{extra}"
            )

        # Output follows the training column order even if the input differs.
        data = data.loc[:, expected]
        out = pd.DataFrame(index=data.index)

        for col in expected:
            values = data[col]
            observed = values.notna()

            labels = pd.Series(
                "缺失箱",
                index=data.index,
                dtype="string",
            )

            if self.all_missing_[col]:
                # No observed value during training: no numeric range exists.
                labels.loc[observed] = "训练外非缺失箱"
            else:
                positions = np.searchsorted(
                    self.edges_[col],
                    values.loc[observed].to_numpy(),
                    side="left",
                )
                labels.loc[observed] = [
                    f"数值箱_{int(position):03d}"
                    for position in positions
                ]

            out[col] = labels

        return out
