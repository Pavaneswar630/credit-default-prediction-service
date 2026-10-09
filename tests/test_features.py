from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_default.database import make_engine, read_training_features
from credit_default.features import engineer_features
from credit_default.schema import MODEL_FEATURES, RAW_FEATURES, TARGET


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.2, 42, 1, 0.3, 60000, 5, 0, 1, 0, 1],
            [1.1, 68, 2, 0.8, 0, 3, 1, 2, 1, 2],
            [0.0, None, None, 0.0, None, None, None, None, None, None],
        ],
        columns=RAW_FEATURES,
    )


def test_engineered_features_are_stable_and_keep_nulls() -> None:
    features = engineer_features(sample_rows())

    assert tuple(features.columns) == MODEL_FEATURES
    assert features.loc[0, "TotalPastDueEvents"] == 1
    assert features.loc[1, "MaxPastDueEvents"] == 2
    assert features.loc[0, "TotalCreditLines"] == 6
    assert features.loc[0, "MonthlyIncomePerDependent"] == 30000
    assert pd.isna(features.loc[2, "TotalPastDueEvents"])
    assert pd.isna(features.loc[2, "MonthlyIncomePerDependent"])


def test_sql_features_match_online_features() -> None:
    engine = make_engine("sqlite://")
    raw = sample_rows()
    raw[TARGET] = [0, 1, 0]
    try:
        raw.to_sql("raw_credit_data", engine, index=False)
        sql_features, labels = read_training_features(engine)
    finally:
        engine.dispose()

    expected = engineer_features(raw).loc[:, MODEL_FEATURES]
    assert_frame_equal(
        sql_features.loc[:, MODEL_FEATURES].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert labels.tolist() == [0, 1, 0]


def test_engineering_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required feature columns"):
        engineer_features(pd.DataFrame({"age": [42]}))
