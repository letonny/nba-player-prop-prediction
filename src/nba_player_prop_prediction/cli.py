"""Local commands; collection failures return a nonzero exit status."""

import argparse
import calendar
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.config import Settings
from nba_player_prop_prediction.ingestion.collector import TYPES, Collector, import_file
from nba_player_prop_prediction.preprocessing.clean import clean_games
from nba_player_prop_prediction.reporting import quality_report
from nba_player_prop_prediction.utils.io import (
    configure_logging,
    read_table,
    write_json,
    write_parquet,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--start-season", type=int, required=True, help="Starting year, e.g. 2023")
    ingest.add_argument("--end-season", type=int, help="Inclusive starting year")
    ingest.add_argument("--season-type", choices=TYPES, default=TYPES[0])
    ingest.add_argument("--start-date")
    ingest.add_argument("--end-date")
    ingest.add_argument("--refresh", action="store_true")
    imp = sub.add_parser("import")
    imp.add_argument("path", type=Path)
    imp.add_argument("--source", required=True)
    clean = sub.add_parser("preprocess")
    clean.add_argument("paths", type=Path, nargs="+")
    clean.add_argument("--season")
    clean.add_argument("--season-type", choices=TYPES)
    clean.add_argument("--output", type=Path)
    report = sub.add_parser("report")
    report.add_argument("path", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    configure_logging()
    if args.command == "ingest":
        end = args.end_season if args.end_season is not None else args.start_season
        if end < args.start_season:
            parser.error("end-season must be >= start-season")
        if bool(args.start_date) != bool(args.end_date):
            parser.error("Supply both start-date and end-date")
        if args.start_date and end != args.start_season:
            parser.error("Explicit dates require a single season")
        collector = Collector(settings)
        failures = []
        for year in range(args.start_season, end + 1):
            season = f"{year}-{(year + 1) % 100:02}"
            windows = (
                [(args.start_date, args.end_date)]
                if args.start_date
                else [
                    (f"{y}-{m:02}-01", f"{y}-{m:02}-{calendar.monthrange(y, m)[1]}")
                    for y, m in [(year, m) for m in range(7, 13)]
                    + [(year + 1, m) for m in range(1, 11)]
                ]
            )
            # Extended season windows also cover the 2019-20 delayed season.
            for start, stop in windows:
                try:
                    print(collector.collect(season, args.season_type, start, stop, args.refresh))
                except Exception as exc:
                    failures.append(
                        {"season": season, "start": start, "end": stop, "error": str(exc)}
                    )
        if failures:
            write_json(settings.artifacts_dir / "ingestion_failures.json", failures)
            raise SystemExit(1)
    elif args.command == "import":
        print(import_file(args.path, settings, args.source))
    elif args.command == "preprocess":
        output = args.output or settings.data_dir / "processed" / "games.parquet"
        if output.resolve() in [p.resolve() for p in args.paths]:
            parser.error("Output must differ from inputs")
        raw = pd.concat([read_table(p) for p in args.paths], ignore_index=True)
        result = clean_games(raw, args.season, args.season_type)
        result.audit["sources"] = [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in args.paths
        ]
        write_parquet(output, result.data)
        write_parquet(output.with_suffix(".rejected.parquet"), result.rejected)
        write_json(output.with_suffix(".audit.json"), result.audit)
        logging.info("Preprocessing: %s", result.audit)
        print(output)
        if result.rejected.shape[0]:
            raise SystemExit(2)
    else:
        audit_path = args.path.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
        quality_report(
            read_table(args.path),
            args.output or settings.artifacts_dir / "quality",
            audit,
            args.plots,
        )


if __name__ == "__main__":
    main()
