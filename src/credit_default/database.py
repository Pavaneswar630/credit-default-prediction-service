"""CSV ingestion and SQL feature extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from credit_default.features import FEATURE_QUERY
from credit_default.schema import RAW_FEATURES, RAW_TABLE, TARGET


def make_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for SQLite or PostgreSQL."""
    return create_engine(database_url, future=True)


def ingest_csv(csv_path: str | Path, engine: Engine) -> int:
    """Load a labeled Kaggle training CSV into the fixed raw table."""
    frame = pd.read_csv(csv_path)
    required = set(RAW_FEATURES) | {TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, [*RAW_FEATURES, TARGET]]
    frame.to_sql(RAW_TABLE, engine, if_exists="replace", index=False)
    return len(frame)


def read_training_features(engine: Engine) -> tuple[pd.DataFrame, pd.Series]:
    """Execute SQL-derived features and return the model matrix and labels."""
    with engine.connect() as connection:
        frame = pd.read_sql_query(text(FEATURE_QUERY), connection)
    if TARGET not in frame:
        raise ValueError(f"The {RAW_TABLE!r} table must contain {TARGET!r} labels")
    labels = pd.to_numeric(frame.pop(TARGET), errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only non-null 0/1 labels")
    ignored = {"age_band", "age_cohort_size", "ID", "Unnamed: 0"}
    features = frame.drop(columns=[name for name in ignored if name in frame])
    missing = sorted(set(RAW_FEATURES).difference(features.columns))
    if missing:
        raise ValueError(f"SQL feature query is missing columns: {', '.join(missing)}")
    return features, labels.astype("int64")
