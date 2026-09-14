"""Step 16 cross-module acceptance: raw rows to the scoring API.

It runs split -> training -> policy selection -> scorecard -> artefact export
and verified load -> API request, then compares the API output against the
offline probability, score and simulated decision. Artificial rows only, so no
public dataset download and no final holdout is touched.

Passing this proves the modules connect, the artefact restores and the service
does not change the model's numbers. It does not prove that real data imports
correctly, that the model has business value, or that the service is safe under
concurrency.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data_layer.split_data import stratified_split
from src.features.engineering import REQUIRED_COLUMNS
from src.models.logistic import RiskLogisticModel
from src.models.scorecard import LogisticScorecard
from src.serving.api import create_app
from src.serving.artifact import (
    ReleaseInfo,
    ServingBundle,
    canary_frame,
    export_bundle,
)
from src.strategy.approval import (
    PolicyRequirements,
    UtilityAssumptions,
    build_policy_curve,
    select_policy,
)


@pytest.mark.integration
def test_training_to_api_is_consistent(tmp_path):
    # The strong artificial signal only keeps the connection check stable; it is
    # not a model result.
    base = canary_frame().to_dict("records")[0]
    rows = []

    for number in range(120):
        label = int(number % 4 == 0)

        rows.append(
            {
                **base,
                "sk_id_curr": 10000 + number,
                "target": label,
                "ext_source_1": 0.2 if label == 1 else 0.8,
                "ext_source_2": 0.6,
            }
        )

    raw = pd.DataFrame(rows)
    split = stratified_split(raw)

    model = RiskLogisticModel().fit(split.X_train, split.y_train)

    valid_probability = model.predict_bad_probability(split.X_valid)

    curve = build_policy_curve(
        split.y_valid,
        valid_probability,
        thresholds=[0.05, 0.1, 0.2, 0.5],
        assumptions=UtilityAssumptions(),
    )

    selected = select_policy(
        curve,
        PolicyRequirements(
            min_approval_rate=0,
            max_approved_event_rate=0.5,
            min_approved_count=1,
            require_positive_utility=True,
        ),
    )

    bundle = ServingBundle(
        release=ReleaseInfo(
            artifact_id="端到端测试包",
            model_version="人工模型一",
            strategy_version="人工策略一",
        ),
        scorecard=LogisticScorecard(model),
        rule=selected.rule,
    )

    directory = tmp_path / "release"
    digest = export_bundle(bundle, directory)

    serving_input = split.X_valid.loc[:, list(REQUIRED_COLUMNS)].head(8)

    expected = bundle.predict_frame(serving_input)
    original_probability = model.predict_bad_probability(
        split.X_valid
    ).loc[serving_input.index]

    token = "a" * 43
    app = create_app(directory, digest, token)

    applications = [
        {"application_id": str(identifier), **values}
        for identifier, values in zip(
            serving_input.index, serving_input.to_dict("records")
        )
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/score",
            headers={"Authorization": "Bearer " + token},
            json={
                "request_id": "端到端请求一",
                "applications": applications,
            },
        )

    assert response.status_code == 200

    result = response.json()
    assert result["artifact_id"] == "端到端测试包"
    assert result["purpose"] == "研究仿真，非真实授信决定"

    for position, item in enumerate(result["results"]):
        assert item["predicted_bad_probability"] == pytest.approx(
            original_probability.iloc[position], abs=1e-12
        )
        assert item["score_raw"] == pytest.approx(
            expected["score_raw"].iloc[position], abs=1e-9
        )
        assert item["simulated_approval"] == bool(
            expected["simulated_approval"].iloc[position]
        )

    # This test never evaluates or selects anything on the final test split.
    assert not set(serving_input.index) & set(split.X_test.index)
