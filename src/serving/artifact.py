"""Step 15: artefact export, digest verification and load self-checks.

One release package carries the scorecard (which already owns the fitted
logistic snapshot), its scale and the frozen policy together, so a fitted
probability, a score and a simulated decision can never come from three
different versions.

Loading is deliberately strict:

* the manifest digest must be supplied from an already verified source, not read
  from the directory being loaded;
* the manifest, the model bytes, the environment and the project source digest
  are all checked **before** anything is deserialised;
* the same in-memory bytes that passed the digest check are the bytes that get
  restored, so the file cannot change between checking and loading;
* any source change invalidates older packages, because custom preprocessing
  classes serialise state rather than frozen program behaviour.

A digest proves content, not publisher identity, and the canary check proves
consistency, not model quality or the absence of malicious code. Loading only
trusted, internally exported artefacts remains a precondition.
"""
import hashlib
import io
import json
import platform
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import REQUIRED_COLUMNS
from src.models.scorecard import LogisticScorecard
from src.strategy.approval import ApprovalRule


OUTPUT_COLUMNS = [
    "predicted_bad_probability",
    "score_raw",
    "score_display",
    "unknown_selected_bins",
    "input_review_required",
    "simulated_approval",
]


@dataclass(frozen=True)
class ReleaseInfo:
    artifact_id: str
    model_version: str
    strategy_version: str
    input_schema_version: str = "贷款申请数值字段第一版"


@dataclass
class ServingBundle:
    release: ReleaseInfo
    scorecard: LogisticScorecard
    rule: ApprovalRule | None = None

    def validate(self) -> None:
        if not isinstance(self.release, ReleaseInfo):
            raise TypeError("发布信息类型错误。")
        if not isinstance(self.scorecard, LogisticScorecard):
            raise TypeError("发布包必须包含本项目评分卡。")
        if self.rule is not None and not isinstance(
            self.rule, ApprovalRule
        ):
            raise TypeError("发布包中的策略类型错误。")

        for name, value in asdict(self.release).items():
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 128
            ):
                raise ValueError(f"发布字段 {name} 无效。")

        if self.rule is None and self.release.strategy_version != "未配置":
            raise ValueError("没有策略时，策略版本必须明确标记未配置。")

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """The one prediction entry point used offline and by the API."""
        self.validate()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("评分输入必须是数据表。")
        if not X.columns.is_unique:
            raise ValueError("评分字段不能重复。")
        if set(X.columns) != set(REQUIRED_COLUMNS):
            raise ValueError("评分输入必须严格匹配当前数值字段约定。")

        ordered = X.loc[:, list(REQUIRED_COLUMNS)]

        scores = self.scorecard.score(ordered)
        diagnostics = self.scorecard.input_diagnostics(ordered)

        pd.testing.assert_index_equal(scores.index, diagnostics.index)

        out = scores.join(diagnostics)

        if self.rule is None:
            # No policy means no decision; nothing is invented in its place.
            out["simulated_approval"] = pd.Series(
                [None] * len(out), index=out.index, dtype="object"
            )
        else:
            out["simulated_approval"] = self.rule.decide(
                out["predicted_bad_probability"]
            )

        return out.loc[:, OUTPUT_COLUMNS]


def canary_frame() -> pd.DataFrame:
    """Fixed artificial self-check rows; no real customer record is used."""
    base = {
        "amt_income_total": 100000.0,
        "amt_credit": 200000.0,
        "amt_annuity": 10000.0,
        "days_birth": -10957.5,
        "days_employed": -3652.5,
        "ext_source_1": 0.7,
        "ext_source_2": 0.6,
        "ext_source_3": 0.5,
        "bureau_cnt": 1,
        "bureau_debt_total": 20000.0,
    }

    rows = [
        base,
        {
            **base,
            "ext_source_1": None,
            "ext_source_2": None,
            "days_employed": 365243.0,
        },
        {
            **base,
            "amt_income_total": 0.0,
            "bureau_cnt": 0,
            "bureau_debt_total": None,
        },
    ]

    return pd.DataFrame(rows).loc[:, list(REQUIRED_COLUMNS)]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _environment() -> dict:
    return {
        "解释器": platform.python_version(),
        "数值计算": version("numpy"),
        "数据处理": version("pandas"),
        "科学计算": version("scipy"),
        "机器学习": version("scikit-learn"),
        "序列化": version("joblib"),
    }


def _source_digest() -> str:
    """Digest of the project source files.

    It is not a supply-chain guarantee, but it catches the common case of a
    package being loaded in front of code that has since changed.
    """
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()

    files = sorted(source_root.rglob("*.py"))
    if not files:
        raise RuntimeError("未找到需要校验的项目源码。")

    for path in files:
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("工件必须是本地普通文件，不能是符号链接。")

    with path.open("rb") as handle:
        contents = handle.read(limit + 1)

    if len(contents) > limit:
        raise ValueError("工件文件超过当前允许大小。")

    return contents


def export_bundle(bundle: ServingBundle, destination: Path) -> str:
    """Export to a brand-new directory and return the manifest digest.

    The manifest is written last: a directory that failed midway is not a
    successful release.
    """
    bundle.validate()
    expected = bundle.predict_frame(canary_frame())

    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    model_bytes = buffer.getvalue()

    if len(model_bytes) > 64 * 1024 * 1024:
        raise ValueError("当前研究发布包超过允许大小。")

    manifest = {
        "manifest_schema": 1,
        "release": asdict(bundle.release),
        "model_file": "bundle.joblib",
        "model_sha256": _sha256(model_bytes),
        "environment": _environment(),
        "source_sha256": _source_digest(),
        "canary_outputs": expected.to_dict("records"),
    }

    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    (destination / "bundle.joblib").write_bytes(model_bytes)
    (destination / "manifest.json").write_bytes(manifest_bytes)

    return _sha256(manifest_bytes)


def load_bundle(
    directory: Path,
    trusted_manifest_sha256: str,
) -> ServingBundle:
    """Verify bytes and versions first, then deserialise the trusted artefact."""
    if (
        not isinstance(trusted_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", trusted_manifest_sha256) is None
    ):
        raise ValueError("必须提供事先核验的发布清单摘要。")

    directory = Path(directory)

    manifest_bytes = _read_bounded(
        directory / "manifest.json", limit=1024 * 1024
    )

    if _sha256(manifest_bytes) != trusted_manifest_sha256:
        raise ValueError("发布清单摘要不匹配，拒绝加载。")

    manifest = json.loads(manifest_bytes)

    if (
        manifest.get("manifest_schema") != 1
        or manifest.get("model_file") != "bundle.joblib"
    ):
        raise ValueError("发布清单结构版本或工件名称不受支持。")

    if (
        manifest["environment"] != _environment()
        or manifest["source_sha256"] != _source_digest()
    ):
        raise ValueError("运行环境或项目源码版本不匹配。")

    model_bytes = _read_bounded(
        directory / "bundle.joblib", limit=64 * 1024 * 1024
    )

    if _sha256(model_bytes) != manifest["model_sha256"]:
        raise ValueError("模型工件摘要不匹配，拒绝加载。")

    # Restore exactly the bytes that passed the digest check; reopening the path
    # could load different content from the one that was verified.
    bundle = joblib.load(io.BytesIO(model_bytes))

    if not isinstance(bundle, ServingBundle):
        raise TypeError("恢复对象不是预期的评分发布包。")

    bundle.validate()

    if asdict(bundle.release) != manifest["release"]:
        raise ValueError("发布包内部版本与清单不一致。")

    actual = bundle.predict_frame(canary_frame()).reset_index(drop=True)
    expected = pd.DataFrame(
        manifest["canary_outputs"], columns=OUTPUT_COLUMNS
    )

    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            atol=1e-9,
            rtol=1e-10,
        )
    except AssertionError as exc:
        raise RuntimeError("加载自检预测与导出时不一致。") from exc

    return bundle
