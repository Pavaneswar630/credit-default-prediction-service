DROP VIEW IF EXISTS engineered_credit_features;

CREATE VIEW engineered_credit_features AS
WITH delinquency_events AS (
    SELECT
        rowid AS source_rowid,
        "NumberOfTime30-59DaysPastDueNotWorse" AS event_count
    FROM raw_credit_data

    UNION ALL

    SELECT
        rowid AS source_rowid,
        "NumberOfTime60-89DaysPastDueNotWorse" AS event_count
    FROM raw_credit_data

    UNION ALL

    SELECT
        rowid AS source_rowid,
        "NumberOfTimes90DaysLate" AS event_count
    FROM raw_credit_data
),
delinquency_totals AS (
    SELECT
        source_rowid,
        CASE
            WHEN COUNT(event_count) = 0 THEN NULL
            ELSE SUM(event_count)
        END AS total_past_due_events
    FROM delinquency_events
    GROUP BY source_rowid
)
SELECT
    raw_credit_data.*,
    delinquency_totals.total_past_due_events,
    "MonthlyIncome" * "DebtRatio" AS estimated_monthly_debt,
    CASE
        WHEN "NumberOfOpenCreditLinesAndLoans" IS NULL
          OR "NumberOfOpenCreditLinesAndLoans" = 0
            THEN NULL
        ELSE 1.0 * "NumberRealEstateLoansOrLines"
            / "NumberOfOpenCreditLinesAndLoans"
    END AS real_estate_ratio,
    CASE
        WHEN "NumberOfDependents" IS NULL
          OR "NumberOfDependents" + 1 = 0
            THEN NULL
        ELSE 1.0 * "MonthlyIncome" / ("NumberOfDependents" + 1)
    END AS income_per_capita,
    PERCENT_RANK() OVER (
        ORDER BY "RevolvingUtilizationOfUnsecuredLines"
    ) AS revolving_utilization_ratio_rank
FROM raw_credit_data
JOIN delinquency_totals
    ON raw_credit_data.rowid = delinquency_totals.source_rowid;
