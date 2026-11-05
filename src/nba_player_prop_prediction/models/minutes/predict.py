"""Pregame inference from local history and an explicit scheduled game context."""

from datetime import UTC, datetime

import pandas as pd

from nba_player_prop_prediction.features.pregame import availability_at, build_features
from nba_player_prop_prediction.models.minutes.training import MinutesModel


def predict_player_minutes(
    model: MinutesModel,
    history: pd.DataFrame,
    player_id: str,
    game_context: dict,
    prediction_cutoff: str,
    availability: pd.DataFrame | None = None,
) -> dict:
    cutoff = pd.Timestamp(prediction_cutoff)
    tipoff = pd.Timestamp(game_context["scheduled_tipoff"])
    if cutoff.tzinfo is None or tipoff.tzinfo is None or cutoff >= tipoff:
        raise ValueError("Timezone-aware cutoff must precede scheduled tipoff")
    # NBA schedule dates are interpreted in Eastern time; exclude cutoff day entirely.
    date = cutoff.tz_convert("America/New_York").date()
    if pd.Timestamp(model.metadata["trained_through"]).date() >= date:
        raise ValueError("Model used labels unavailable at this prediction cutoff")
    context = dict(game_context, player_id=str(player_id), cutoff_date=str(date))
    context["player_game_id"] = f"{player_id}:{context['game_id']}"
    if pd.Timestamp(context["game_date"]).date() != tipoff.tz_convert("America/New_York").date():
        raise ValueError("game_date must match the scheduled NBA Eastern calendar date")
    features = build_features(history, pd.DataFrame([context]))
    predicted = float(model.predict(features)[0])
    radius = model.metadata["interval_radius"]
    warnings = [
        "Conditional on appearing; inactive risk is not modeled.",
        "Same-day history excluded; historical publication timestamps unavailable.",
    ]
    if features.prior_games.iloc[0] < 5:
        warnings.append("Fewer than five historical appearances.")
    missing = features[model.metadata["columns"]].isna().sum(axis=1).iloc[0]
    if missing:
        warnings.append(f"{missing} numeric/context features are missing.")
    status = (
        None
        if availability is None
        else availability_at(
            availability, str(player_id), str(context["game_id"]), prediction_cutoff
        )
    )
    if status is None:
        warnings.append("No trusted timestamped availability record; injury features omitted.")
    return {
        "player_id": str(player_id),
        "game_id": str(context["game_id"]),
        "prediction_cutoff": cutoff.isoformat(),
        "prediction_timestamp": datetime.now(UTC).isoformat(),
        "predicted_minutes": predicted,
        "model_version": model.metadata["version"],
        "interval": {
            "lower": max(0.0, predicted - radius),
            "upper": predicted + radius,
            "nominal_coverage": 1 - model.metadata["interval_alpha"],
            "method": "validation absolute-residual interval; not guaranteed",
        },
        "availability_status": None if status is None else str(status["status"]),
        "warnings": warnings,
    }
