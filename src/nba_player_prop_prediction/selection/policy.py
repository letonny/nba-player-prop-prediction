"""Deterministic probability ranking; risk restrictions never alter probabilities."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    minimum: float = 0.85
    max_total: int = 100
    max_per_game: int = 10
    max_per_player_game: int = 1
    min_calibration: int = 500
    require_market: bool = False
    require_available: bool = False
    min_history: int = 0
    exclude_restriction: bool = True
    exclude_back_to_back: bool = False
    minimum_minutes_change: float | None = None

    def __post_init__(self):
        if (
            not 0 <= self.minimum <= 1
            or min(
                self.max_total,
                self.max_per_game,
                self.max_per_player_game,
                self.min_calibration,
                self.min_history,
            )
            < 0
        ):
            raise ValueError("Invalid selection policy")


def rank(candidates: pd.DataFrame, policy: Policy | None = None) -> pd.DataFrame:
    policy = policy or Policy()
    frame = candidates.copy()
    required = {
        "candidate_id",
        "player_game_id",
        "game_id",
        "game_date",
        "probability",
        "target",
        "line",
        "event",
        "calibration_n",
        "market_available",
        "availability",
        "prior_games",
        "minutes_restricted",
    }
    if not required.issubset(frame):
        raise ValueError(f"Missing candidate fields: {sorted(required - set(frame))}")
    if frame.empty:
        return frame.assign(
            selected=pd.Series(dtype=bool), reason=pd.Series(dtype=str), policy=pd.Series(dtype=str)
        )
    if frame.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidates")
    if not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
        raise ValueError("Invalid probabilities")
    frame = frame.sort_values(["probability", "candidate_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    frame["selected"] = False
    frame["reason"] = ""
    total = {}
    games = {}
    players = {}
    for i, r in frame.iterrows():
        reason = None
        if r.probability < policy.minimum:
            reason = "below_probability"
        elif r.calibration_n < policy.min_calibration:
            reason = "insufficient_calibration"
        elif r.availability in ["out", "inactive"]:
            reason = "inactive"
        elif policy.require_available and r.availability != "available":
            reason = "uncertain_availability"
        elif policy.exclude_restriction and r.minutes_restricted:
            reason = "minutes_restriction"
        elif policy.exclude_back_to_back and r.get("back_to_back", None) != 0:
            reason = "back_to_back_or_unknown"
        elif policy.minimum_minutes_change is not None and (
            pd.isna(r.get("minutes_change"))
            or r.get("minutes_change") < policy.minimum_minutes_change
        ):
            reason = "minutes_decline_or_unknown"
        elif r.prior_games < policy.min_history:
            reason = "limited_history"
        elif policy.require_market and not r.market_available:
            reason = "market_unavailable"
        elif total.get(str(r.game_date), 0) >= policy.max_total:
            reason = "daily_limit"
        elif games.get(r.game_id, 0) >= policy.max_per_game:
            reason = "game_limit"
        elif players.get(r.player_game_id, 0) >= policy.max_per_player_game:
            reason = "player_game_limit"
        if reason is None:
            frame.at[i, "selected"] = True
            total[str(r.game_date)] = total.get(str(r.game_date), 0) + 1
            games[r.game_id] = games.get(r.game_id, 0) + 1
            players[r.player_game_id] = players.get(r.player_game_id, 0) + 1
        frame.at[i, "reason"] = reason or "selected"
    frame["eligible"] = frame.reason.isin(
        ["selected", "daily_limit", "game_limit", "player_game_limit"]
    )
    frame["correlated_research"] = policy.max_per_player_game > 1
    frame["policy"] = str(asdict(policy))
    return frame
