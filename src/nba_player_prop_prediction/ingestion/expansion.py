"""Bounded, resumable season collection and latest-snapshot assembly."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.config import Settings
from nba_player_prop_prediction.ingestion.collector import Collector
from nba_player_prop_prediction.preprocessing.clean import clean_games
from nba_player_prop_prediction.utils.io import read_table, write_json, write_parquet


def expand(
    settings: Settings,
    years: list[int],
    max_requests: int = 4,
    budget_seconds: float = 180,
    refresh: bool = False,
) -> dict:
    """One whole-season request per year; completed windows reuse M1's cache.

    The budget is checked between requests; an in-flight request can exceed it
    by its configured timeout/retries. Current season uses today's date.
    """
    if max_requests < 1 or budget_seconds <= 0:
        raise ValueError("Request/time budgets must be positive")
    collector = Collector(settings)
    start = time.monotonic()
    report = {"completed": [], "failed": [], "pending": []}
    today = pd.Timestamp.today().date()
    for index, year in enumerate(years):
        if index >= max_requests or time.monotonic() - start >= budget_seconds:
            report["pending"].extend(years[index:])
            break
        try:
            end = min(pd.Timestamp(year=year + 1, month=6, day=30).date(), today)
            path = collector.collect(
                f"{year}-{(year + 1) % 100:02}",
                "Regular Season",
                f"{year}-07-01",
                str(end),
                refresh,
            )
            report["completed"].append(str(path))
        except Exception as exc:
            report["failed"].append({"year": year, "error": str(exc)})
        write_json(settings.artifacts_dir / "expansion.json", report)
    write_json(settings.artifacts_dir / "expansion.json", report)
    return report


def assemble(settings: Settings, output: Path) -> dict:
    """Use only latest manifests; newest retrieval wins for overlapping windows.

    Raw snapshots remain immutable. Revised rows are counted in the audit.
    """
    frames, sources = [], []
    for manifest in sorted((settings.data_dir / "raw" / "nba_api").glob("*/latest.json")):
        meta = json.loads(manifest.read_text())
        path = manifest.parent / meta["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {path}")
        frame = read_table(path)
        if frame.empty:
            continue
        frame["snapshot_time"] = meta["fetched_at"]
        frames.append(frame)
        sources.append({"path": str(path), "sha256": meta["sha256"]})
    if not frames:
        raise ValueError("No nonempty cached NBA snapshots available")
    combined = pd.concat(frames, ignore_index=True).sort_values("snapshot_time", kind="stable")
    overlap = int(combined.duplicated(["PLAYER_ID", "GAME_ID"]).sum())
    combined = combined.drop_duplicates(["PLAYER_ID", "GAME_ID"], keep="last")
    result = clean_games(combined)
    write_parquet(output, result.data)
    write_parquet(output.with_suffix(".rejected.parquet"), result.rejected)
    summary = coverage(result.data)
    summary.update(sources=sources, overlapping_rows_replaced=overlap, cleaning=result.audit)
    write_json(output.with_suffix(".coverage.json"), summary)
    return summary


def coverage(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "players": int(frame.player_id.nunique()),
        "games": int(frame.game_id.nunique()),
        "start": str(frame.game_date.min()),
        "end": str(frame.game_date.max()),
        "by_season": {
            str(s): {
                "rows": len(g),
                "players": int(g.player_id.nunique()),
                "games": int(g.game_id.nunique()),
                "dates": int(g.game_date.nunique()),
            }
            for s, g in frame.groupby("season")
        },
        "missing_percent": frame.isna().mean().mul(100).fillna(0).to_dict(),
    }
