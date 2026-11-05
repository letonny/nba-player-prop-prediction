"""Append-only forecast payloads and one-time grading in SQLite."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def stamp(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return t


def _now():
    return pd.Timestamp.now(tz="UTC")


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, "
            "payload TEXT NOT NULL, created TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS grades (id TEXT PRIMARY KEY REFERENCES predictions(id), "
            "status TEXT NOT NULL, actual REAL, evidence TEXT NOT NULL, graded TEXT NOT NULL)"
        )
        self.db.executescript("""
        CREATE TRIGGER IF NOT EXISTS immutable_prediction_update BEFORE UPDATE ON predictions
        BEGIN SELECT RAISE(ABORT,'immutable prediction'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_prediction_delete BEFORE DELETE ON predictions
        BEGIN SELECT RAISE(ABORT,'immutable prediction'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_grade_update BEFORE UPDATE ON grades
        BEGIN SELECT RAISE(ABORT,'immutable grade'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_grade_delete BEFORE DELETE ON grades
        BEGIN SELECT RAISE(ABORT,'immutable grade'); END;
        """)

    def record(self, payload):
        now = _now()
        tipoff = stamp(payload["scheduled_tipoff"])
        cutoff = stamp(payload["source_cutoff"])
        generated = stamp(payload["generated_at"])
        if not cutoff <= generated <= now < tipoff:
            raise ValueError("Forward forecasts must be recorded before tipoff with valid cutoffs")
        if payload.get("mode") != "forward":
            raise ValueError("Only forward forecasts enter this ledger")
        if any(k in payload for k in ["hit", "actual", "result", "label"]):
            raise ValueError("Outcome fields forbidden in forecast")
        for key in [
            "player_id",
            "game_id",
            "target",
            "line",
            "event",
            "model_version",
            "policy",
            "probability",
        ]:
            if key not in payload:
                raise ValueError(f"Missing {key}")
        if not 0 <= payload["probability"] <= 1:
            raise ValueError("Invalid probability")
        text = json.dumps(payload, sort_keys=True, allow_nan=False)
        identity = {
            k: payload[k]
            for k in [
                "player_id",
                "game_id",
                "target",
                "line",
                "event",
                "model_version",
                "source_cutoff",
                "policy",
            ]
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        existing = self.db.execute("SELECT payload FROM predictions WHERE id=?", (key,)).fetchone()
        if existing:
            if existing[0] != text:
                raise ValueError("Conflicting existing prediction")
            return key
        with self.db:
            self.db.execute("INSERT INTO predictions VALUES (?,?,?)", (key, text, now.isoformat()))
        return key

    def grade(self, key, actual, participation, completed_at, evidence, market_settlement=None):
        row = self.db.execute("SELECT payload FROM predictions WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown prediction")
        p = json.loads(row[0])
        completed = stamp(completed_at)
        if not stamp(p["scheduled_tipoff"]) < completed <= _now():
            raise ValueError("Outcome is not a completed game")
        if not evidence:
            raise ValueError("Outcome source required")
        if p.get("market_available"):
            if market_settlement not in ["hit", "miss", "void", "unable_to_grade"]:
                raise ValueError("Market-backed grading requires documented provider settlement")
            status = market_settlement
        elif participation != "played" or actual is None:
            status = "unable_to_grade"
        else:
            if actual < 0 or int(actual) != actual:
                raise ValueError("Invalid count outcome")
            if p["event"] in ["over", "under"] and actual == p["line"]:
                status = "void"
            elif p["event"] == "at_least":
                status = "hit" if actual >= p["line"] else "miss"
            elif p["event"] == "over":
                status = "hit" if actual > p["line"] else "miss"
            elif p["event"] == "under":
                status = "hit" if actual < p["line"] else "miss"
            else:
                raise ValueError("Unknown event")
        previous = self.db.execute(
            "SELECT status,actual,evidence FROM grades WHERE id=?", (key,)
        ).fetchone()
        values = (status, actual, evidence)
        if previous:
            if previous != values:
                raise ValueError("Grade cannot be overwritten")
            return status
        with self.db:
            self.db.execute(
                "INSERT INTO grades VALUES (?,?,?,?,?)",
                (key, status, actual, evidence, datetime.now(UTC).isoformat()),
            )
        return status

    def export(self):
        rows = []
        for key, payload, created, status, actual in self.db.execute(
            "SELECT p.id,p.payload,p.created,g.status,g.actual FROM predictions p "
            "LEFT JOIN grades g ON p.id=g.id"
        ):
            rows.append(
                dict(
                    json.loads(payload),
                    prediction_id=key,
                    recorded_at=created,
                    grading_status=status or "pending",
                    actual=actual,
                )
            )
        return rows
