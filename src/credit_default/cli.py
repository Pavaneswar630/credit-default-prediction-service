"""Command-line entry points for data ingestion and model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.engine import make_url

from credit_default.database import ingest_csv, make_engine
from credit_default.training import train_model


def _ensure_sqlite_parent(database_url: str) -> None:
    url = make_url(database_url)
    if url.drivername.startswith("sqlite") and url.database not in (None, ":memory:"):
        Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="credit-default")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Load the Kaggle training CSV into SQL")
    ingest.add_argument("--csv", required=True, type=Path)
    ingest.add_argument(
        "--database-url", default="sqlite:///data/credit_default.db"
    )

    train = commands.add_parser("train", help="Tune, evaluate, and export a model")
    train.add_argument("--database-url", default="sqlite:///data/credit_default.db")
    train.add_argument("--model-output", default="models/credit_default.joblib")
    train.add_argument("--tracking-uri", default="./mlruns")
    train.add_argument("--trials", type=int, default=30)
    train.add_argument("--model", choices=("xgboost", "lightgbm"), default="xgboost")

    args = parser.parse_args()
    _ensure_sqlite_parent(args.database_url)
    if args.command == "ingest":
        engine = make_engine(args.database_url)
        try:
            row_count = ingest_csv(args.csv, engine)
        finally:
            engine.dispose()
        print(f"Loaded {row_count} rows into raw_credit_data")
        return

    metrics = train_model(
        database_url=args.database_url,
        model_output=args.model_output,
        n_trials=args.trials,
        model_name=args.model,
        tracking_uri=args.tracking_uri,
    )
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")


if __name__ == "__main__":
    main()
