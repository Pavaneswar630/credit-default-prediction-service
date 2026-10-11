from __future__ import annotations

import sqlite3

import pandas as pd

from scripts.db_setup import load_dataset


def test_load_dataset_creates_engineered_features_view(tmp_path) -> None:
    csv_path = tmp_path / "cs-training.csv"
    database_path = tmp_path / "credit_data.db"
    pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.2,
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfTimes90DaysLate": 3,
                "MonthlyIncome": 6000,
                "DebtRatio": 0.5,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfDependents": 1,
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.8,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfTimes90DaysLate": 0,
                "MonthlyIncome": 3000,
                "DebtRatio": 0.25,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfOpenCreditLinesAndLoans": 5,
                "NumberOfDependents": 2,
            },
        ]
    ).to_csv(csv_path, index=False)

    assert load_dataset(csv_path, database_path) == 2

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT total_past_due_events, estimated_monthly_debt,
                   real_estate_ratio, income_per_capita,
                   revolving_utilization_ratio_rank
            FROM engineered_credit_features
            ORDER BY RevolvingUtilizationOfUnsecuredLines
            """
        ).fetchall()

    assert rows == [(6, 3000.0, 0.5, 3000.0, 0.0), (0, 750.0, 0.2, 1000.0, 1.0)]
