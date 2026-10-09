from __future__ import annotations

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from credit_default.api import create_app
from credit_default.schema import MODEL_FEATURES, RAW_FEATURES
from credit_default.features import engineer_features


def write_test_artifact(path) -> None:
    training_frame = pd.DataFrame(
        [{name: 1.0 for name in RAW_FEATURES}, {name: 2.0 for name in RAW_FEATURES}]
    )
    model_frame = engineer_features(training_frame)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="constant", constant=1)),
        ]
    )
    pipeline.fit(model_frame.loc[:, MODEL_FEATURES], [0, 1])
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": list(MODEL_FEATURES),
            "model_version": "test-model-v1",
        },
        path,
    )


def sample_payload() -> dict[str, float]:
    return {name: 1.0 for name in RAW_FEATURES}


def test_health_and_prediction_load_versioned_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"
    write_test_artifact(artifact_path)
    app = create_app(str(artifact_path))

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=sample_payload())

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert prediction.status_code == 200
    assert prediction.json() == {
        "default_probability": 1.0,
        "predicted_default": True,
        "model_version": "test-model-v1",
    }


def test_unavailable_model_is_not_ready(tmp_path) -> None:
    app = create_app(str(tmp_path / "missing.joblib"))

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=sample_payload())

    assert health.status_code == 503
    assert prediction.status_code == 503


def test_prediction_rejects_unknown_fields(tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"
    write_test_artifact(artifact_path)
    app = create_app(str(artifact_path))
    payload = {**sample_payload(), "unexpected": 5}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
