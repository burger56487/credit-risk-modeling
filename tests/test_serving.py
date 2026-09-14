"""Step 15 tests: artefact verification, strict input and online==offline."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.logistic import RiskLogisticModel
from src.models.scorecard import LogisticScorecard, ScoreScale
from src.serving.api import create_app
from src.serving.artifact import (
    OUTPUT_COLUMNS,
    ReleaseInfo,
    ServingBundle,
    canary_frame,
    export_bundle,
    load_bundle,
)
from src.strategy.approval import ApprovalRule


TEST_TOKEN = "t" * 43


@pytest.fixture
def bundle():
    n = 80
    labels = np.arange(n) % 2

    # Keep every field's original type, especially the integer bureau count.
    base = canary_frame().to_dict("records")[0]
    X = pd.DataFrame([base.copy() for _ in range(n)])

    X["ext_source_1"] = np.where(labels == 1, 0.2, 0.8)
    X["ext_source_2"] = X["ext_source_1"]
    y = pd.Series(labels, index=X.index)

    model = RiskLogisticModel().fit(X, y)

    return ServingBundle(
        release=ReleaseInfo(
            artifact_id="测试发布包",
            model_version="测试模型一",
            strategy_version="测试策略一",
        ),
        scorecard=LogisticScorecard(model),
        rule=ApprovalRule(threshold=0.1),
    )


@pytest.fixture
def exported(bundle, tmp_path):
    directory = tmp_path / "release"
    digest = export_bundle(bundle, directory)
    return directory, digest


def request_payload():
    application = canary_frame().to_dict("records")[0]
    application["application_id"] = "人工申请一"

    return {"request_id": "请求一", "applications": [application]}


def auth_headers():
    return {"Authorization": "Bearer " + TEST_TOKEN}


def test_export_load_preserves_predictions(bundle, exported):
    directory, digest = exported
    restored = load_bundle(directory, digest)

    pd.testing.assert_frame_equal(
        restored.predict_frame(canary_frame()),
        bundle.predict_frame(canary_frame()),
    )


def test_export_does_not_overwrite_existing_directory(bundle, exported):
    directory, _ = exported

    with pytest.raises(FileExistsError):
        export_bundle(bundle, directory)


@pytest.mark.parametrize("filename", ["manifest.json", "bundle.joblib"])
def test_tampering_is_rejected_before_deserialization(
    exported, monkeypatch, filename
):
    directory, digest = exported
    path = directory / filename
    path.write_bytes(path.read_bytes() + b"changed")

    called = []

    def forbidden_load(*args, **kwargs):
        called.append(True)
        raise AssertionError("不应进入反序列化。")

    monkeypatch.setattr(
        "src.serving.artifact.joblib.load", forbidden_load
    )

    with pytest.raises(ValueError, match="摘要"):
        load_bundle(directory, digest)

    assert not called


def test_source_mismatch_prevents_loading(exported, monkeypatch):
    directory, digest = exported

    monkeypatch.setattr(
        "src.serving.artifact._source_digest", lambda: "0" * 64
    )

    with pytest.raises(ValueError, match="源码版本"):
        load_bundle(directory, digest)


def test_wrong_pin_prevents_application_start(exported):
    directory, _ = exported

    with pytest.raises(ValueError, match="摘要"):
        create_app(directory, "0" * 64, TEST_TOKEN)


def test_api_matches_offline_result(bundle, exported):
    directory, digest = exported
    app = create_app(directory, digest, TEST_TOKEN)
    payload = request_payload()

    with TestClient(app) as client:
        response = client.post(
            "/v1/score", json=payload, headers=auth_headers()
        )

    assert response.status_code == 200
    actual = response.json()["results"][0]

    raw = {
        key: value
        for key, value in payload["applications"][0].items()
        if key != "application_id"
    }
    expected = bundle.predict_frame(pd.DataFrame([raw])).iloc[0]

    for column in (
        "predicted_bad_probability",
        "score_raw",
        "score_display",
    ):
        assert actual[column] == pytest.approx(expected[column], abs=1e-9)

    assert actual["simulated_approval"] == bool(
        expected["simulated_approval"]
    )
    assert response.json()["purpose"] == "研究仿真，非真实授信决定"
    assert response.json()["artifact_id"] == "测试发布包"


def test_missing_token_rejected(exported):
    directory, digest = exported

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.post("/v1/score", json=request_payload())

    assert response.status_code == 401


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", 1),
        ("amt_income_total", "敏感输入占位"),
        ("amt_credit", True),
        ("bureau_cnt", 1.2),
    ],
)
def test_strict_input_rejects_invalid_fields_without_echo(
    exported, field, value
):
    directory, digest = exported
    payload = request_payload()
    payload["applications"][0][field] = value

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.post(
            "/v1/score", json=payload, headers=auth_headers()
        )

    assert response.status_code == 422
    assert "敏感输入占位" not in response.text
    assert "amt_income_total" not in response.text


def test_duplicate_application_identifiers_rejected(exported):
    directory, digest = exported
    payload = request_payload()
    payload["applications"].append(
        payload["applications"][0].copy()
    )

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.post(
            "/v1/score", json=payload, headers=auth_headers()
        )

    assert response.status_code == 422


def test_explicit_missing_fields_are_supported_and_flagged(exported):
    directory, digest = exported
    payload = request_payload()

    payload["applications"][0]["ext_source_1"] = None
    payload["applications"][0]["ext_source_2"] = None
    payload["applications"][0]["days_employed"] = 365243.0

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.post(
            "/v1/score", json=payload, headers=auth_headers()
        )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["input_review_required"] is True
    assert result["unknown_selected_bins"] >= 1


def test_oversized_request_body_rejected(exported):
    directory, digest = exported

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.post(
            "/v1/score",
            content=b"x" * 65537,
            headers={
                **auth_headers(),
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 413


def test_unconfigured_policy_returns_no_decision(bundle, tmp_path):
    no_policy = ServingBundle(
        release=ReleaseInfo(
            artifact_id="无策略测试包",
            model_version="测试模型一",
            strategy_version="未配置",
        ),
        scorecard=bundle.scorecard,
        rule=None,
    )

    directory = tmp_path / "no_policy"
    digest = export_bundle(no_policy, directory)

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.post(
            "/v1/score", json=request_payload(), headers=auth_headers()
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["simulated_approval"] is None


def test_health_endpoint_needs_no_token(exported):
    directory, digest = exported

    with TestClient(
        create_app(directory, digest, TEST_TOKEN)
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "就绪"}


def test_offline_frame_returns_the_documented_columns(bundle):
    frame = bundle.predict_frame(canary_frame())

    assert list(frame.columns) == OUTPUT_COLUMNS
    assert len(frame) == len(canary_frame())
    assert frame["predicted_bad_probability"].between(0, 1).all()
    assert frame["input_review_required"].iloc[0] is np.False_ or (
        frame["input_review_required"].iloc[0] == False  # noqa: E712
    )


def test_bundle_rejects_wrong_input_and_inconsistent_release(bundle, tmp_path):
    with pytest.raises(ValueError, match="严格匹配"):
        bundle.predict_frame(canary_frame().drop(columns=["ext_source_3"]))

    broken_columns = canary_frame()
    broken_columns["extra"] = 1.0
    with pytest.raises(ValueError, match="严格匹配"):
        bundle.predict_frame(broken_columns)

    with pytest.raises(ValueError, match="未配置"):
        ServingBundle(
            release=ReleaseInfo(
                artifact_id="不一致包",
                model_version="测试模型一",
                strategy_version="测试策略一",
            ),
            scorecard=bundle.scorecard,
            rule=None,
        ).validate()

    with pytest.raises(TypeError):
        ServingBundle(
            release=bundle.release,
            scorecard=object(),
        ).validate()


def test_scoring_uses_the_raw_probability_not_the_display_score(tmp_path):
    """Rounding for display must not move the simulated decision."""
    n = 80
    labels = np.arange(n) % 2
    base = canary_frame().to_dict("records")[0]
    X = pd.DataFrame([base.copy() for _ in range(n)])
    X["ext_source_1"] = np.where(labels == 1, 0.2, 0.8)
    X["ext_source_2"] = X["ext_source_1"]
    model = RiskLogisticModel().fit(X, pd.Series(labels, index=X.index))

    scorecard = LogisticScorecard(
        model,
        ScoreScale(
            base_score=600.0,
            base_bad_good_odds=1.0 / 50.0,
            points_to_double_odds=20.0,
        ),
    )
    frame = scorecard.score(canary_frame())
    probability = frame["predicted_bad_probability"]

    # A threshold exactly between the raw probability and the rounded score
    # boundary must follow the probability.
    rule = ApprovalRule(threshold=float(probability.iloc[0]))
    assert rule.decide(probability).iloc[0]

    bundle = ServingBundle(
        release=ReleaseInfo(
            artifact_id="阈值边界包",
            model_version="测试模型一",
            strategy_version="边界测试策略",
        ),
        scorecard=scorecard,
        rule=rule,
    )
    digest = export_bundle(bundle, tmp_path / "boundary")
    restored = load_bundle(tmp_path / "boundary", digest)

    result = restored.predict_frame(canary_frame())
    assert bool(result["simulated_approval"].iloc[0]) is True
