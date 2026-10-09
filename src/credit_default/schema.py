"""Canonical input and feature names for Give Me Some Credit."""

RAW_FEATURES = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)
TARGET = "SeriousDlqin2yrs"
DELINQUENCY_FEATURES = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
)
DERIVED_FEATURES = (
    "TotalPastDueEvents",
    "MaxPastDueEvents",
    "TotalCreditLines",
    "MonthlyIncomePerDependent",
)
MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES
RAW_TABLE = "raw_credit_data"
