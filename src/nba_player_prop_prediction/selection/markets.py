"""Optional The Odds API adapter. No credentials or market access are assumed."""

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import requests

from nba_player_prop_prediction.utils.io import write_json


class MarketProvider(Protocol):
    def snapshot(self, event_id: str, markets: list[str]) -> dict: ...


class OddsAPI:
    def __init__(self, cache: Path, key=None):
        self.cache = cache
        self.key = key or os.getenv("ODDS_API_KEY")

    def snapshot(self, event_id, markets):
        if not self.key:
            return {"available": False, "reason": "No ODDS_API_KEY configured"}
        if not event_id.isalnum():
            raise ValueError("Invalid provider event ID")
        identity = hashlib.sha256(json.dumps([event_id, sorted(markets)]).encode()).hexdigest()
        latest = self.cache / (identity + ".json")
        if latest.exists() and time.time() - latest.stat().st_mtime < 60:
            return json.loads(latest.read_text())
        time.sleep(1.5)
        try:
            response = requests.get(
                f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds",
                params={"apiKey": self.key, "regions": "us", "markets": ",".join(markets)},
                timeout=20,
            )
            if response.status_code != 200:
                return {"available": False, "reason": f"Provider HTTP {response.status_code}"}
            stamp = datetime.now(UTC).isoformat()
            result = {
                "available": True,
                "source": "the-odds-api.com",
                "snapshot_at": stamp,
                "remaining_credits": response.headers.get("x-requests-remaining"),
                "data": response.json(),
            }
            # Immutable snapshot; player/event ID reconciliation is caller work.
            path = self.cache / (identity + "-" + str(time.time_ns()) + ".json")
            write_json(path, result)
            result["snapshot_reference"] = str(path)
            write_json(latest, result)
            return result
        except (requests.RequestException, ValueError):
            return {"available": False, "reason": "Provider unavailable or invalid response"}


def confirmed_market(snapshot, player_id, game_id, target, line, event, cutoff):
    """Accept only an explicit normalized market record with provider provenance."""
    import pandas as pd

    required = {
        "source",
        "snapshot_at",
        "sportsbook",
        "market_id",
        "player_id",
        "game_id",
        "target",
        "line",
        "event",
        "odds",
        "available",
    }
    if not required.issubset(snapshot):
        return False
    observed = pd.Timestamp(snapshot["snapshot_at"])
    limit = pd.Timestamp(cutoff)
    if observed.tzinfo is None or limit.tzinfo is None:
        return False
    return (
        observed <= limit
        and limit - observed <= pd.Timedelta(minutes=5)
        and all(snapshot[k] for k in ["source", "sportsbook", "market_id"])
        and snapshot["available"] is True
        and all(
            snapshot[k] == v
            for k, v in {
                "player_id": player_id,
                "game_id": game_id,
                "target": target,
                "line": line,
                "event": event,
            }.items()
        )
    )
