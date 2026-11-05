"""Strictly historical statistical baselines and OOF joins."""

import numpy as np
import pandas as pd

from nba_player_prop_prediction.probabilistic.distributions import TARGETS


def enrich(history, features, oof):
    if oof.player_game_id.duplicated().any():
        raise ValueError("Duplicate OOF minutes")
    data = features.merge(oof.drop(columns="game_date"), on="player_game_id", validate="one_to_one")
    if not (pd.to_datetime(data.minutes_trained_through) < data.game_date).all():
        raise ValueError("Minutes training overlaps predicted game")
    if not (pd.to_datetime(data.prediction_asof) <= data.game_date).all():
        raise ValueError("Invalid minutes prediction cutoff")
    labels = history.set_index("player_game_id")
    for target in TARGETS:
        data[f"label_{target}"] = data.player_game_id.map(labels[target]).astype(float)
    data["appearance"] = data.player_game_id.map(labels.participation).eq("played")
    groups = {
        str(p): h.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        for p, h in history.loc[history.participation.eq("played")].groupby("player_id")
    }
    for target in TARGETS:
        data[f"season_mean_{target}"] = np.nan
        data[f"rate_{target}"] = np.nan
        data[f"prior_std_{target}"] = np.nan
    for player, contexts in data.groupby("player_id"):
        h = groups.get(str(player))
        if h is None:
            continue
        idx = (
            np.searchsorted(h.game_date.to_numpy(), contexts.game_date.to_numpy(), side="left") - 1
        )
        for target in TARGETS:
            vals = h[target].astype(float)
            # Ratios use matched valid observations; zero minutes are excluded.
            valid = vals.notna() & h.minutes.gt(0)
            numerator = vals.where(valid).fillna(0).cumsum()
            denominator = h.minutes.where(valid).fillna(0).cumsum().replace(0, np.nan)
            rate = (numerator / denominator).reindex(idx).to_numpy()
            data.loc[contexts.index, f"rate_{target}"] = rate
            data.loc[contexts.index, f"prior_std_{target}"] = (
                vals.rolling(20, min_periods=2).std().reindex(idx).to_numpy()
            )
        for season, section in contexts.groupby("season"):
            sh = h.loc[h.season.eq(season)].reset_index(drop=True)
            ids = (
                np.searchsorted(sh.game_date.to_numpy(), section.game_date.to_numpy(), side="left")
                - 1
            )
            for target in TARGETS:
                data.loc[section.index, f"season_mean_{target}"] = (
                    sh[target].astype(float).expanding().mean().reindex(ids).to_numpy()
                )
    return data.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def partitions(data, validation="2024-07-01", calibration="2025-01-01", test="2025-07-01"):
    v, c, t = map(pd.Timestamp, [validation, calibration, test])
    if not v < c < t:
        raise ValueError("Require train < validation < calibration < test")
    return {
        "train": data.loc[data.game_date < v],
        "validation": data.loc[(data.game_date >= v) & (data.game_date < c)],
        "calibration": data.loc[(data.game_date >= c) & (data.game_date < t)],
        "test": data.loc[data.game_date >= t],
    }
