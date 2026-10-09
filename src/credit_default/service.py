"""Loading and running the serialized prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_default.features import engineer_features
from credit_default.schema import MODEL_FEATURES


class PredictionService:
    """A loaded, immutable model bundle used by the API process."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        self.pipeline: Any | None = None
        self.model_version: str | None = None

    def load(self) -> None:
        """Load a trusted joblib model artifact and verify its feature contract."""
        if not self.artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")
        artifact = joblib.load(self.artifact_path)
        if not isinstance(artifact, dict) or not {
            "pipeline",
            "feature_columns",
            "model_version",
        }.issubset(artifact):
            raise ValueError("Model artifact has an unsupported format")
        if artifact["feature_columns"] != list(MODEL_FEATURES):
            raise ValueError("Model artifact feature columns do not match this service")
        self.pipeline = artifact["pipeline"]
        self.model_version = str(artifact["model_version"])

    def predict(self, raw_features: pd.DataFrame) -> float:
        """Return the probability of serious delinquency for one borrower."""
        if self.pipeline is None:
            raise RuntimeError("Prediction model has not been loaded")
        features = engineer_features(raw_features)
        probability = self.pipeline.predict_proba(features.loc[:, MODEL_FEATURES])[:, 1]
        return float(probability[0])
