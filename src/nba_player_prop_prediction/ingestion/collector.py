"""Content-addressed raw snapshots and resumable, date-window API collection."""

import hashlib
import json
import logging
import re
import shutil
import time
from datetime import UTC, date, datetime
from pathlib import Path

from nba_player_prop_prediction.config import Settings
from nba_player_prop_prediction.utils.io import atomic_write, read_table, write_json, write_parquet

log = logging.getLogger(__name__)
TYPES = ("Regular Season", "Playoffs")


def validate_season(season: str):
    if not re.fullmatch(r"\d{4}-\d{2}", season):
        raise ValueError("Season must be YYYY-YY")
    if int(season[-2:]) != (int(season[:4]) + 1) % 100:
        raise ValueError("Season years must be consecutive")


def nba_fetch(**kwargs):
    from nba_api.stats.endpoints.playergamelogs import PlayerGameLogs

    return PlayerGameLogs(**kwargs).get_data_frames()[0]


class Collector:
    def __init__(self, settings: Settings, fetch=nba_fetch, sleep=time.sleep):
        self.settings, self.fetch, self.sleep = settings, fetch, sleep

    def collect(
        self, season: str, season_type: str, start: str, end: str, refresh: bool = False
    ) -> Path:
        validate_season(season)
        if season_type not in TYPES:
            raise ValueError(f"season_type must be one of {TYPES}")
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        if first > last:
            raise ValueError("start must not be after end")
        params = dict(
            season_nullable=season,
            season_type_nullable=season_type,
            date_from_nullable=first.strftime("%m/%d/%Y"),
            date_to_nullable=last.strftime("%m/%d/%Y"),
            per_mode_simple_nullable="Totals",
        )
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        folder = self.settings.data_dir / "raw" / "nba_api" / key
        manifest = folder / "latest.json"
        if manifest.exists() and not refresh:
            cached = json.loads(manifest.read_text())
            path = folder / cached["file"]
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == cached["sha256"]:
                log.info("Cache hit: %s", path)
                return path
            log.warning("Missing/corrupt cache; requesting again: %s", manifest)
        for attempt in range(self.settings.retries + 1):
            self.sleep(self.settings.pacing)
            try:
                frame = self.fetch(**params, timeout=self.settings.timeout)
                if not {"PLAYER_ID", "GAME_ID", "GAME_DATE"}.issubset(frame.columns):
                    raise ValueError("API returned an unexpected schema")
                if frame.empty:
                    log.warning("No games returned for %s; refresh to recheck this window", params)
                break
            except Exception as exc:
                if attempt == self.settings.retries:
                    log.error("Collection failed for %s: %s", params, exc)
                    raise
                delay = self.settings.backoff * 2**attempt
                response = getattr(exc, "response", None)
                if response is not None:
                    try:
                        delay = max(delay, float(response.headers.get("Retry-After", 0)))
                    except (TypeError, ValueError):
                        pass
                log.warning("Request failed (%s); retrying in %.1fs", exc, delay)
                self.sleep(delay)
        # Context columns supplement the response; no source columns are overwritten.
        frame = frame.assign(source_season=season, source_season_type=season_type)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / f"{stamp}.parquet"
        write_parquet(path, frame)
        metadata = dict(
            source="nba_api.stats.PlayerGameLogs",
            parameters=params,
            fetched_at=stamp,
            file=path.name,
            rows=len(frame),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        write_json(path.with_suffix(".json"), metadata)
        write_json(manifest, metadata)
        log.info("Saved %d rows to %s", len(frame), path)
        return path


def import_file(source: Path, settings: Settings, provenance: str) -> Path:
    """Preserve original bytes; record an explicitly supplied source description."""
    if not provenance.strip():
        raise ValueError("A source description or URL is required")
    read_table(source)  # Validate supported format before copying.
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    path = settings.data_dir / "raw" / "imports" / f"{digest}{source.suffix.lower()}"
    if not path.exists():
        atomic_write(path, lambda p: shutil.copyfile(source, p))
        write_json(
            path.with_suffix(path.suffix + ".json"),
            {
                "source": provenance,
                "original_name": source.name,
                "sha256": digest,
                "imported_at": datetime.now(UTC).isoformat(),
            },
        )
    return path
