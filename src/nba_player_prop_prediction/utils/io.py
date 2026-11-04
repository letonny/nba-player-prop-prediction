"""Atomic local writes; source files are never edited in place."""

import json
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd


def configure_logging():
    logging.basicConfig(
        level=os.getenv("NBA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def atomic_write(path: Path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False)))


def write_parquet(path: Path, frame: pd.DataFrame):
    atomic_write(path, lambda p: frame.to_parquet(p, index=False))


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Only CSV and Parquet files are supported")
