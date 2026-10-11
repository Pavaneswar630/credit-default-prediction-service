# Credit Default Prediction Service

A reproducible training and inference starter for Kaggle's [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset. Python 3.10 is the supported runtime. The service keeps the raw Kaggle schema at its API boundary and derives the same borrower-level features used during SQL-backed training.

## Setup

Download Kaggle's labeled `cs-training.csv` and place it at `data/cs-training.csv`. Create a Python 3.10 virtual environment, then install the project and development dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For Bash, activate with `source .venv/bin/activate`. Dependencies are pinned in `pyproject.toml` for repeatable environments.

## Ingest And Train

By default, ingestion creates `data/credit_default.db` and replaces its `raw_credit_data` table. The input must be Kaggle's labeled training CSV; identifiers are intentionally excluded from model features.

```powershell
credit-default ingest --csv data/cs-training.csv
credit-default train --trials 30 --model xgboost
```

### Standalone SQLite feature view

The SQL feature pipeline can also be loaded into a standalone `credit_data.db` at the repository root. It reads the default `data/cs-training.csv` path and creates/replaces `raw_credit_data` plus the `engineered_credit_features` view:

```powershell
python scripts/db_setup.py
```

Pass `--csv` and `--database` to use different paths. Query the engineered rows with `SELECT * FROM engineered_credit_features`; the view definition is in `sql/features.sql`.

Choose `--model lightgbm` to tune LightGBM. Training reserves a stratified 20% holdout, performs seeded stratified 5-fold ROC-AUC tuning with Optuna on the remaining data, reports holdout ROC-AUC, average precision, and Brier score, logs the run to local MLflow (`./mlruns`), writes a SHAP feature-importance plot, and exports `models/credit_default.joblib`. At least ten rows from each target class are required. The exported model is refit on all labeled rows after the holdout evaluation.

SQLite is the default. PostgreSQL can be used by passing a SQLAlchemy URL, for example `postgresql+psycopg://user:password@host:5432/credit`:

```powershell
credit-default ingest --csv data/cs-training.csv --database-url "postgresql+psycopg://user:password@host:5432/credit"
credit-default train --database-url "postgresql+psycopg://user:password@host:5432/credit"
```

SQL derives per-borrower delinquency and credit-line aggregates. Its age-band `COUNT(*) OVER (...)` is retained as a dataset audit feature and excluded from the model because its value would change with each inference batch. This source dataset has no customer-history key or time dimension, so the project does not invent temporal aggregates.

## Serve Locally

After training, the API loads the joblib artifact from `MODEL_PATH` (default `models/credit_default.joblib`):

```powershell
uvicorn credit_default.api:app --host 127.0.0.1 --port 8000
```

`GET /health` reports readiness and returns HTTP 503 if the artifact is missing. `POST /predict` accepts the ten raw Kaggle predictor fields and returns a default probability, a 0.5-threshold prediction, and the model run ID. Interactive schema docs are at `/docs`.

Example request:

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.35,
  "age": 42,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.4,
  "MonthlyIncome": 6500,
  "NumberOfOpenCreditLinesAndLoans": 8,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 1
}
```

## Docker And Render

Build the image and mount the trained artifact read-only at runtime:

```powershell
docker build -t credit-default .
docker run --rm -p 8000:8000 -e MODEL_PATH=/models/credit_default.joblib -v "${PWD}/models:/models:ro" credit-default
```

For Render, create a Docker web service, set `PORT` to `8000`, and make the trained artifact available at the configured `MODEL_PATH` using the deployment's persistent storage or artifact-delivery mechanism. A deployment without a readable model intentionally remains unready. Do not accept joblib artifacts from untrusted sources; loading one can execute code.

## Checks

```powershell
pytest
```

GitHub Actions runs this test suite on Python 3.10. Tests cover deterministic feature construction, SQL/serving parity, API readiness, artifact loading, and request validation. Training on the full Kaggle dataset and PostgreSQL integration are workload/deployment checks and are not run in CI.
