"""Load the Give Me Some Credit CSV into the local SQLite feature database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "cs-training.csv"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "credit_data.db"
FEATURES_SQL_PATH = PROJECT_ROOT / "sql" / "features.sql"


def load_dataset(csv_path: Path, database_path: Path) -> int:
    """Replace the raw table and create the engineered-features view."""
    frame = pd.read_csv(csv_path)

    with sqlite3.connect(database_path) as connection:
        frame.to_sql("raw_credit_data", connection, if_exists="replace", index=False)
        connection.executescript(FEATURES_SQL_PATH.read_text(encoding="utf-8"))

    return len(frame)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load the Give Me Some Credit CSV into SQLite."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Input CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"SQLite database path (default: {DEFAULT_DATABASE_PATH})",
    )
    args = parser.parse_args(argv)

    row_count = load_dataset(args.csv, args.database)
    print(f"Loaded {row_count} rows into {args.database}")


if __name__ == "__main__":
    main()
