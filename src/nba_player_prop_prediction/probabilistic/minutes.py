"""Season-block expanding minutes forecasts, never fitted on a predicted block."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from nba_player_prop_prediction.features.pregame import model_columns
from nba_player_prop_prediction.models.minutes.training import matrix
from nba_player_prop_prediction.utils.io import write_json, write_parquet


def walk_forward(
    history, features, output: Path, boundaries=None, iterations=100, min_train=5000, min_dates=60
):
    boundaries = boundaries or ["2023-07-01", "2024-07-01", "2025-07-01", "2026-07-01"]
    dates = pd.to_datetime(boundaries)
    if len(dates) < 2 or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Boundaries must strictly increase")
    if output.exists():
        raise ValueError("Use a new OOF output directory")
    output.mkdir(parents=True)
    columns = model_columns(features)
    labels = history.set_index("player_game_id")
    y = features.player_game_id.map(labels.minutes)
    eligible = features.player_game_id.map(labels.participation).eq("played") & y.notna()
    predictions, folds = [], []
    for start, end in zip(dates[:-1], dates[1:], strict=True):
        train = eligible & (features.game_date < start)
        predict = (features.game_date >= start) & (features.game_date < end)
        if train.sum() < min_train or features.loc[train, "game_date"].nunique() < min_dates:
            folds.append({"start": str(start), "status": "skipped_insufficient_history"})
            continue
        # Final test block freezes training at its beginning; no test-year refits.
        model = CatBoostRegressor(
            iterations=iterations,
            depth=5,
            learning_rate=0.06,
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
            has_time=True,
        )
        model.fit(
            matrix(features.loc[train], columns),
            y.loc[train],
            cat_features=[
                c
                for c in columns
                if c in ["player_id", "team_id", "opponent_id", "home_away", "season_type"]
            ],
        )
        name = str(start.date())
        model.save_model(str(output / f"{name}.cbm"))
        part = features.loc[predict, ["player_game_id", "game_date"]].copy()
        part["predicted_minutes"] = np.maximum(
            0, model.predict(matrix(features.loc[predict], columns))
        )
        # Historical volatility is a pregame uncertainty proxy, not a calibrated interval.
        part["minutes_uncertainty"] = features.loc[predict, "prior_10_minutes_std"]
        cutoff = features.loc[train, "game_date"].max()
        part["minutes_trained_through"] = cutoff
        part["prediction_asof"] = part.game_date
        part["minutes_model_frozen_at"] = start
        part["generated_at"] = datetime.now(UTC).isoformat()
        part["minutes_model_version"] = hashlib.sha256(
            (output / f"{name}.cbm").read_bytes()
        ).hexdigest()
        assert (part.minutes_trained_through < part.game_date).all()
        predictions.append(part)
        folds.append(
            {
                "start": name,
                "end_exclusive": str(end.date()),
                "n_train": int(train.sum()),
                "n_predict": len(part),
                "trained_through": str(cutoff),
                "model": f"{name}.cbm",
            }
        )
    if not predictions:
        raise ValueError("No eligible out-of-fold predictions")
    result = pd.concat(predictions, ignore_index=True)
    write_parquet(output / "minutes.parquet", result)
    write_json(
        output / "manifest.json",
        {
            "folds": folds,
            "columns": columns,
            "iterations": iterations,
            "warmup": "Rows before first boundary omitted",
            "uncertainty": "Prior ten-appearance sample standard deviation; not predictive SD",
        },
    )
    return result
