"""Deterministic borrower-level features shared by offline and online paths."""

from __future__ import annotations

import pandas as pd

from credit_default.schema import (
    DELINQUENCY_FEATURES,
    DERIVED_FEATURES,
    MODEL_FEATURES,
    RAW_FEATURES,
)

FEATURE_QUERY = '''
WITH borrower_rows AS (
    SELECT
        *,
        CASE
            WHEN "age" IS NULL THEN 'unknown'
            ELSE CAST(CAST("age" / 10 AS INTEGER) AS TEXT)
        END AS "age_band"
    FROM "raw_credit_data"
),
engineered AS (
    SELECT
        *,
        CASE
            WHEN "NumberOfTime30-59DaysPastDueNotWorse" IS NULL
             AND "NumberOfTime60-89DaysPastDueNotWorse" IS NULL
             AND "NumberOfTimes90DaysLate" IS NULL THEN NULL
            ELSE COALESCE("NumberOfTime30-59DaysPastDueNotWorse", 0)
               + COALESCE("NumberOfTime60-89DaysPastDueNotWorse", 0)
               + COALESCE("NumberOfTimes90DaysLate", 0)
        END AS "TotalPastDueEvents",
        CASE
            WHEN "NumberOfTime30-59DaysPastDueNotWorse" IS NULL
             AND "NumberOfTime60-89DaysPastDueNotWorse" IS NULL
             AND "NumberOfTimes90DaysLate" IS NULL THEN NULL
            ELSE CASE
                WHEN "NumberOfTime30-59DaysPastDueNotWorse" IS NOT NULL
                 AND ("NumberOfTime60-89DaysPastDueNotWorse" IS NULL
                      OR "NumberOfTime30-59DaysPastDueNotWorse"
                         >= "NumberOfTime60-89DaysPastDueNotWorse")
                 AND ("NumberOfTimes90DaysLate" IS NULL
                      OR "NumberOfTime30-59DaysPastDueNotWorse"
                         >= "NumberOfTimes90DaysLate")
                    THEN "NumberOfTime30-59DaysPastDueNotWorse"
                WHEN "NumberOfTime60-89DaysPastDueNotWorse" IS NOT NULL
                 AND ("NumberOfTimes90DaysLate" IS NULL
                      OR "NumberOfTime60-89DaysPastDueNotWorse"
                         >= "NumberOfTimes90DaysLate")
                    THEN "NumberOfTime60-89DaysPastDueNotWorse"
                ELSE "NumberOfTimes90DaysLate"
            END
        END AS "MaxPastDueEvents",
        CASE
            WHEN "NumberOfOpenCreditLinesAndLoans" IS NULL
             AND "NumberRealEstateLoansOrLines" IS NULL THEN NULL
            ELSE COALESCE("NumberOfOpenCreditLinesAndLoans", 0)
               + COALESCE("NumberRealEstateLoansOrLines", 0)
        END AS "TotalCreditLines",
        "MonthlyIncome" / NULLIF(COALESCE("NumberOfDependents", 0) + 1, 0)
            AS "MonthlyIncomePerDependent"
    FROM borrower_rows
)
SELECT
    engineered.*,
    COUNT(*) OVER (PARTITION BY "age_band") AS "age_cohort_size"
FROM engineered
'''


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model features from raw Kaggle columns, preserving input row order."""
    missing = sorted(set(RAW_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    result = frame.loc[:, RAW_FEATURES].copy()
    for column in RAW_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["TotalPastDueEvents"] = result.loc[:, DELINQUENCY_FEATURES].sum(
        axis=1, min_count=1
    )
    result["MaxPastDueEvents"] = result.loc[:, DELINQUENCY_FEATURES].max(axis=1)
    result["TotalCreditLines"] = result.loc[
        :, ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]
    ].sum(axis=1, min_count=1)

    denominator = result["NumberOfDependents"].fillna(0) + 1
    denominator = denominator.mask(denominator == 0)
    result["MonthlyIncomePerDependent"] = result["MonthlyIncome"] / denominator
    return result.loc[:, MODEL_FEATURES]
