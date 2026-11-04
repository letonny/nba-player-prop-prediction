"""Conservative box-score normalization with explicit row quarantine."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_player_prop_prediction.ingestion.collector import TYPES

ALIASES = {
    "min": "minutes",
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "threes_made",
    "fga": "field_goal_attempts",
    "fg3a": "three_point_attempts",
    "fta": "free_throw_attempts",
    "tov": "turnovers",
    "pf": "personal_fouls",
    "wl": "result",
    "season_year": "season",
}
STATS = [
    "points",
    "rebounds",
    "assists",
    "threes_made",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "turnovers",
    "personal_fouls",
]
TARGETS = ["points", "rebounds", "assists", "threes_made", "pra"]
IDENTITY = ["player_id", "game_id"]
REQUIRED = {
    "player_id",
    "player_name",
    "team_id",
    "game_id",
    "game_date",
    "minutes",
    "points",
    "rebounds",
    "assists",
    "threes_made",
}


def parse_minutes(value) -> float:
    if pd.isna(value) or str(value).strip().upper() in {"", "DNP", "DND", "INACTIVE"}:
        return np.nan
    text = str(value).strip()
    if ":" in text:
        if not re.fullmatch(r"\d+:\d{2}", text):
            raise ValueError("Minutes must be numeric or MM:SS")
        minutes, seconds = map(int, text.split(":"))
        if seconds >= 60:
            raise ValueError("Seconds must be below 60")
        return minutes + seconds / 60
    return float(text)


def normalize_date(value):
    if pd.isna(value) or isinstance(value, (int, float)):
        return pd.NaT
    try:
        return pd.Timestamp(value).normalize().tz_localize(None)
    except (ValueError, TypeError):
        return pd.NaT


@dataclass
class CleanResult:
    data: pd.DataFrame
    rejected: pd.DataFrame
    audit: dict


def clean_games(
    raw: pd.DataFrame, season: str | None = None, season_type: str | None = None
) -> CleanResult:
    frame = raw.copy(deep=True)
    frame.columns = [ALIASES.get(str(c).lower(), str(c).lower()) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Ambiguous duplicate column aliases")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.reset_index(drop=True)
    for col, fallback in [("season", season), ("season_type", season_type)]:
        if col not in frame:
            frame[col] = frame.get("source_" + col, fallback)
        if frame[col].isna().any():
            raise ValueError(f"Supply {col} explicitly; never infer competition type")
    errors = [[] for _ in range(len(frame))]

    def flag(mask, reason):
        for index in frame.index[pd.Series(mask, index=frame.index).fillna(False)]:
            errors[index].append(reason)

    for col in ["player_id", "team_id", "game_id"]:
        values = frame[col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        valid = values.str.fullmatch(r"\d+").fillna(False)
        flag(~valid | values.str.fullmatch(r"0+").fillna(False), f"invalid_{col}")
        if col == "game_id":
            flag(values.str.len() > 10, "invalid_game_id_length")
            frame[col] = values.str.zfill(10)
        else:
            frame[col] = values.str.lstrip("0")
    frame["game_date"] = frame["game_date"].map(normalize_date)
    flag(frame.game_date.isna(), "invalid_game_date")
    flag(
        frame.player_name.isna() | frame.player_name.astype(str).str.strip().eq(""),
        "missing_player_name",
    )
    valid_season = frame.season.astype(str).str.fullmatch(r"\d{4}-\d{2}")
    flag(~valid_season, "invalid_season")
    for idx in frame.index[valid_season]:
        value = str(frame.at[idx, "season"])
        if int(value[-2:]) != (int(value[:4]) + 1) % 100:
            errors[idx].append("invalid_season")
    flag(~frame.season_type.isin(TYPES), "invalid_season_type")
    minute_text = frame.minutes.astype("string").str.upper().str.strip()
    explicit_dnp = minute_text.isin(["DNP", "DND", "INACTIVE"])
    if "participation" in frame:
        explicit_dnp |= frame.participation.eq("dnp")
    parsed = []
    for idx, value in frame.minutes.items():
        try:
            parsed.append(parse_minutes(value))
        except (ValueError, TypeError):
            errors[idx].append("invalid_minutes")
            parsed.append(np.nan)
    frame["minutes"] = pd.Series(parsed, dtype="float64")
    flag(
        frame.minutes.notna() & (~np.isfinite(frame.minutes) | (frame.minutes < 0)),
        "invalid_minutes",
    )
    for col in STATS + ["plus_minus"]:
        original = frame[col] if col in frame else pd.Series(np.nan, index=frame.index)
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = (original.notna() & numeric.isna()) | (
            numeric.notna() & (~np.isfinite(numeric) | (numeric % 1 != 0))
        )
        if col != "plus_minus":
            invalid |= numeric < 0
        flag(invalid, f"invalid_{col}")
        frame[col] = numeric.mask(invalid).astype("Int64")
    flag(frame.threes_made > frame.three_point_attempts, "threes_exceed_attempts")
    flag(frame.three_point_attempts > frame.field_goal_attempts, "three_attempts_exceed_fga")
    flag(frame.points < frame.threes_made * 3, "points_below_made_threes")
    positive_stats = frame[STATS].gt(0).any(axis=1)
    flag(explicit_dnp & ((frame.minutes > 0) | positive_stats), "dnp_has_activity")
    frame["participation"] = "unknown"
    frame.loc[explicit_dnp, "participation"] = "dnp"
    frame.loc[~explicit_dnp & ((frame.minutes > 0) | positive_stats), "participation"] = "played"
    frame["pra"] = frame.points + frame.rebounds + frame.assists
    # Team context belongs to the game, never to a player's current roster.
    for col in ["team_name", "opponent", "home_away", "result"]:
        if col not in frame:
            frame[col] = pd.NA
    if "matchup" in frame:
        match = frame.matchup.astype("string").str.extract(r"^([A-Z]{2,3}) (vs\.|@) ([A-Z]{2,3})$")
        frame["opponent"] = frame.opponent.fillna(match[2])
        frame["home_away"] = frame.home_away.fillna(match[1].map({"vs.": "home", "@": "away"}))
        flag(frame.matchup.notna() & match[0].isna(), "invalid_matchup")
    flag(frame.result.notna() & ~frame.result.isin(["W", "L"]), "invalid_result")
    flag(frame.home_away.notna() & ~frame.home_away.isin(["home", "away"]), "invalid_home_away")
    # Opponent ID from the other team's observed game row; absent is unknown.
    teams = frame.groupby("game_id").team_id.agg(lambda s: sorted(set(s.dropna())))
    frame["opponent_id"] = [
        next((t for t in teams[g] if t != team), pd.NA) if len(teams[g]) == 2 else pd.NA
        for g, team in zip(frame.game_id, frame.team_id, strict=True)
    ]
    duplicates = int(frame.duplicated(IDENTITY).sum())
    compare = list(
        REQUIRED
        | set(STATS)
        | {
            "season",
            "season_type",
            "plus_minus",
            "result",
            "opponent",
            "home_away",
            "participation",
        }
    )
    conflicting = (
        frame.groupby(IDENTITY, dropna=False)[compare]
        .transform("nunique", dropna=False)
        .gt(1)
        .any(axis=1)
    )
    flag(conflicting, "conflicting_player_game")
    frame["validation_errors"] = [";".join(e) for e in errors]
    rejected = frame.loc[frame.validation_errors.ne("")].copy()
    accepted = frame.loc[frame.validation_errors.eq("")].drop(columns="validation_errors")
    accepted = (
        accepted.drop_duplicates(IDENTITY)
        .sort_values(["game_date", "game_id", "player_id"])
        .reset_index(drop=True)
    )
    accepted["player_game_id"] = accepted.player_id + ":" + accepted.game_id
    return CleanResult(
        accepted,
        rejected,
        {
            "input_rows": len(raw),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "duplicate_rows": duplicates,
            "removed_duplicate_rows": len(frame) - len(rejected) - len(accepted),
            "missing_percent": {
                c: float(v) for c, v in accepted.isna().mean().mul(100).fillna(0).items()
            },
        },
    )
