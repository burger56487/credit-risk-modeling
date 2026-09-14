"""End-to-end test of the Step 15 release export and load round trip."""
import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

from src.serving.artifact import canary_frame, load_bundle


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_step_15_release.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_step_15_release", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def released(runner, model_table_builder, tmp_path_factory):
    table = tmp_path_factory.mktemp("data") / "model_table.csv"
    model_table_builder(n=1200, separation=3.0).to_csv(table, index=False)

    releases = tmp_path_factory.mktemp("releases")
    runner.main(
        [
            "--model-table",
            str(table),
            "--releases-dir",
            str(releases),
            "--release",
            "研究包第一版",
            "--threshold",
            "0.08",
        ]
    )

    return table, releases


def test_release_directory_contains_a_complete_package(released):
    _, releases = released
    directory = releases / "研究包第一版"

    assert {path.name for path in directory.iterdir()} == {
        "bundle.joblib",
        "manifest.json",
    }
    assert (directory / "bundle.joblib").stat().st_size > 0


def test_reloaded_bundle_reproduces_the_canary_predictions(released):
    _, releases = released
    directory = releases / "研究包第一版"

    import json

    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    digest = __import__("hashlib").sha256(
        (directory / "manifest.json").read_bytes()
    ).hexdigest()

    assert manifest["manifest_schema"] == 1
    assert manifest["release"]["strategy_version"] == "开发候选规则第一版"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["model_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["source_sha256"])
    assert manifest["environment"]["序列化"]

    restored = load_bundle(directory, digest)
    frame = restored.predict_frame(canary_frame())

    assert len(frame) == 3
    assert frame["predicted_bad_probability"].between(0, 1).all()
    assert frame["input_review_required"].iloc[1]  # missing sources + sentinel
    assert restored.release.artifact_id == "研究包第一版"


def test_release_does_not_overwrite_and_needs_a_new_name(released, runner):
    table, releases = released

    with pytest.raises(FileExistsError):
        runner.main(
            [
                "--model-table",
                str(table),
                "--releases-dir",
                str(releases),
                "--release",
                "研究包第一版",
                "--skip-self-request",
            ]
        )


def test_release_without_a_threshold_is_marked_unconfigured(
    runner, model_table_builder, tmp_path
):
    table = tmp_path / "no_rule_table.csv"
    model_table_builder(n=900, separation=3.0).to_csv(table, index=False)

    releases = tmp_path / "releases"
    runner.main(
        [
            "--model-table",
            str(table),
            "--releases-dir",
            str(releases),
            "--release",
            "无策略包",
            "--skip-self-request",
        ]
    )

    import hashlib
    import json

    directory = releases / "无策略包"
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(
        (directory / "manifest.json").read_bytes()
    ).hexdigest()

    assert manifest["release"]["strategy_version"] == "未配置"

    restored = load_bundle(directory, digest)
    frame = restored.predict_frame(canary_frame())

    assert frame["simulated_approval"].isna().all()
    assert frame["predicted_bad_probability"].between(0, 1).all()


def test_release_script_sets_the_research_purpose_flag(runner, tmp_path):
    """The script must not label its own output as a real credit decision."""
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "研究仿真" not in source or "非真实授信" in source
    assert "secrets.token_urlsafe" in source
    assert "print(token)" not in source
