"""Versioned pregame feature allowlist with strict date cutoffs.

Game dates are NBA-local calendar dates, not UTC timestamps. Historical features
exclude the entire cutoff day. Context must come from a known pregame schedule.
"""

import numpy as np
import pandas as pd

from nba_player_prop_prediction.preprocessing.clean import STATS

VERSION = "pregame-v2.1"
CATEGORICAL = ["player_id", "team_id", "opponent_id", "home_away", "season_type"]
CONTEXT = ["player_game_id", "game_id", "game_date", "season", *CATEGORICAL]
WINDOWS = (3, 5, 10, 20)


def _lookup(history, contexts, values):
    """Select aggregates ending strictly before each context's cutoff date."""
    indexes = (
        np.searchsorted(history.game_date.to_numpy(), contexts.cutoff_date.to_numpy(), side="left")
        - 1
    )
    return values.reindex(indexes).set_axis(contexts.index)


def team_history(history: pd.DataFrame) -> pd.DataFrame:
    """Observed team totals; partial box scores are excluded, never zero-filled."""
    rows = []
    for (game, team), group in history.groupby(["game_id", "team_id"], sort=True):
        minutes = group.minutes.sum(min_count=len(group))
        # Five players * (48 + 5 * OT). Tolerance accommodates rounded source minutes.
        if (
            pd.isna(minutes)
            or minutes < 238
            or abs((minutes - 240) / 25 - round((minutes - 240) / 25)) > 0.08
        ):
            continue
        row = {
            "game_id": game,
            "team_id": team,
            "game_date": group.game_date.iloc[0],
            "total_minutes": minutes,
        }
        for col in ["points", "rebounds", "assists"]:
            row[col] = group[col].sum(min_count=len(group))
        inputs = ["field_goal_attempts", "free_throw_attempts", "turnovers", "oreb"]
        totals = {
            c: pd.to_numeric(group[c], errors="coerce").sum(min_count=len(group))
            if c in group
            else np.nan
            for c in inputs
        }
        row["possessions_proxy"] = (
            totals["field_goal_attempts"]
            - totals["oreb"]
            + totals["turnovers"]
            + 0.44 * totals["free_throw_attempts"]
        )
        rows.append(row)
    totals = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "team_id",
            "game_date",
            "total_minutes",
            "possessions_proxy",
            "points",
            "rebounds",
            "assists",
        ],
    )
    paired = totals.merge(totals, on="game_id", suffixes=("", "_other"))
    paired = paired.loc[paired.team_id.ne(paired.team_id_other)]
    counts = totals.groupby("game_id").size()
    paired = paired.loc[paired.game_id.map(counts).eq(2)]
    paired = paired.loc[(paired.total_minutes - paired.total_minutes_other).abs() <= 2]
    for col in ["points", "rebounds", "assists"]:
        paired[f"{col}_allowed"] = paired[f"{col}_other"]
    possessions = (paired.possessions_proxy + paired.possessions_proxy_other) / 2
    possessions = possessions.where(possessions > 0)
    paired["estimated_pace"] = possessions * 48 / (paired.total_minutes / 5)
    paired["estimated_offensive_rating"] = paired.points * 100 / possessions
    paired["estimated_defensive_rating"] = paired.points_other * 100 / possessions
    paired["win"] = (paired.points > paired.points_other).astype(float)
    return paired.sort_values(["game_date", "game_id"])


def build_features(history: pd.DataFrame, contexts: pd.DataFrame | None = None) -> pd.DataFrame:
    """No labels or target-game box scores are returned.

    With contexts omitted, reconstruct historical schedule context from logs.
    Such evaluation is conditional on appearing; it cannot evaluate inactive risk.
    """
    if history.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Duplicate player-game history")
    source = history.copy()
    source["game_date"] = pd.to_datetime(source.game_date).dt.normalize()
    if source.game_date.isna().any():
        raise ValueError("Missing historical game dates")
    context = (history if contexts is None else contexts).copy()
    missing = set(CONTEXT) - set(context.columns)
    if missing:
        raise ValueError(f"Missing game context: {sorted(missing)}")
    result = context[CONTEXT].copy()
    result["game_date"] = pd.to_datetime(result.game_date).dt.normalize()
    result["cutoff_date"] = pd.to_datetime(
        context.get("cutoff_date", result.game_date)
    ).dt.normalize()
    if result.cutoff_date.isna().any() or (result.cutoff_date > result.game_date).any():
        raise ValueError("Cutoff must be present and no later than game date")
    result = result.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    for col in CATEGORICAL:
        result[col] = result[col].astype("string").fillna("unknown")
    season_start = pd.to_datetime(result.season.str[:4] + "-07-01")
    result["season_day"] = (result.game_date - season_start).dt.days.astype(float)
    played = source.loc[source.participation.eq("played")].sort_values(
        ["game_date", "game_id", "player_id"]
    )
    groups = {str(k): g.reset_index(drop=True) for k, g in played.groupby("player_id")}
    pieces = []
    for player, targets in result.groupby("player_id", sort=True):
        h = groups.get(str(player), played.iloc[:0].reset_index(drop=True))
        stats = h[["minutes", *STATS]].astype(float)
        values = pd.DataFrame(index=h.index)
        for window in WINDOWS:
            rolled = stats.rolling(window, min_periods=1)
            for col in stats:
                values[f"prior_{window}_{col}_mean"] = rolled[col].mean()
            values[f"prior_{window}_minutes_median"] = rolled.minutes.median()
            values[f"prior_{window}_minutes_std"] = stats.minutes.rolling(
                window, min_periods=2
            ).std(ddof=1)
        values["previous_minutes"] = stats.minutes
        values["historical_minutes_median"] = stats.minutes.expanding().median()
        values["historical_minutes_min"] = stats.minutes.expanding().min()
        values["historical_minutes_max"] = stats.minutes.expanding().max()
        values["historical_minutes_std"] = stats.minutes.expanding(min_periods=2).std()
        values["ewm_minutes"] = stats.minutes.ewm(span=10, adjust=False).mean()
        values["minutes_change"] = values.prior_3_minutes_mean - values.prior_10_minutes_mean
        values["minutes_consistency"] = values.prior_10_minutes_std / (
            values.prior_10_minutes_mean.replace(0, np.nan)
        )
        values["prior_games"] = np.arange(1, len(h) + 1)
        selected = _lookup(h, targets, values)
        selected["prior_games"] = selected.prior_games.fillna(0)
        dates = h.game_date.to_numpy()
        cutoff = targets.cutoff_date.to_numpy()
        before = np.searchsorted(dates, cutoff, side="left")
        last_dates = h.game_date.reindex(before - 1).set_axis(targets.index)
        selected["days_since_previous_game"] = (targets.game_date - last_dates).dt.days
        selected["rest_days"] = (selected.days_since_previous_game - 1).clip(lower=0)
        selected["back_to_back"] = (
            selected.days_since_previous_game.eq(1).astype(float).where(last_dates.notna())
        )
        for days in (7, 14):
            begin = (targets.game_date - np.timedelta64(days, "D")).to_numpy()
            selected[f"appearances_previous_{days}_days"] = np.maximum(
                0, before - np.searchsorted(dates, begin, side="left")
            )
        selected["prior_vs_opponent_minutes"] = np.nan
        for opponent, target_group in targets.groupby("opponent_id"):
            match = h.loc[h.opponent_id.astype("string").eq(opponent)].reset_index(drop=True)
            aggregate = pd.DataFrame({"mean": match.minutes.astype(float).expanding().mean()})
            selected.loc[target_group.index, "prior_vs_opponent_minutes"] = _lookup(
                match, target_group, aggregate
            )["mean"]
        pieces.append(selected)
    numeric = pd.concat(pieces).sort_index() if pieces else pd.DataFrame(index=result.index)
    result = pd.concat([result, numeric], axis=1)
    teams = team_history(played)
    for prefix, identifier in [("team", "team_id"), ("opponent", "opponent_id")]:
        columns = [
            "points",
            "rebounds",
            "assists",
            "points_allowed",
            "rebounds_allowed",
            "assists_allowed",
            "win",
            "estimated_pace",
            "estimated_offensive_rating",
            "estimated_defensive_rating",
        ]
        for col in columns:
            result[f"{prefix}_prior_10_{col}"] = np.nan
        for team, targets in result.groupby(identifier):
            h = teams.loc[teams.team_id.eq(team)].reset_index(drop=True)
            values = h.reindex(columns=columns).astype(float).rolling(10, min_periods=1).mean()
            selected = _lookup(h, targets, values)
            for col in columns:
                result.loc[targets.index, f"{prefix}_prior_10_{col}"] = selected[col]
    return result.drop(columns="cutoff_date")


def model_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"player_game_id", "game_id", "game_date", "season"}
    return [c for c in features if c not in excluded]


def availability_at(
    statuses: pd.DataFrame, player_id: str, game_id: str, cutoff: str
) -> dict | None:
    """Optional trusted feed adapter; never derive injury status from box scores."""
    required = {"player_id", "game_id", "available_at", "source", "status", "trusted"}
    if not required.issubset(statuses):
        raise ValueError(f"Availability feed requires {sorted(required)}")
    timestamp = pd.Timestamp(cutoff)
    if timestamp.tzinfo is None:
        raise ValueError("Availability cutoff must have a timezone")
    eligible = statuses.loc[
        (statuses.player_id.astype(str) == str(player_id))
        & (statuses.game_id.astype(str) == str(game_id))
        & statuses.trusted.eq(True)
        & statuses.source.notna()
    ].copy()
    eligible["available_at"] = pd.to_datetime(eligible.available_at, utc=True, errors="coerce")
    eligible = eligible.loc[eligible.available_at < timestamp].sort_values("available_at")
    return None if eligible.empty else eligible.iloc[-1].to_dict()
