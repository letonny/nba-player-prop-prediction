"""Distribution and threshold inference from verified pregame feature rows."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nba_player_prop_prediction.probabilistic.experiment import load_bundle


def predict(folder, features, line, event="at_least", method="sigmoid"):
    bundle, report = load_bundle(folder)
    if len(features) != 1:
        raise ValueError("Supply exactly one pregame feature row")
    if pd.Timestamp(bundle["trained_through"]) >= pd.Timestamp(features.game_date.iloc[0]):
        raise ValueError("Prediction date overlaps model/calibrator fitting")
    if not (pd.to_datetime(features.minutes_trained_through) < features.game_date).all():
        raise ValueError("Invalid minutes cutoff")
    dist = bundle["model"].predict(features)
    if isinstance(dist, np.ndarray):
        return {
            "quantiles": dist[0].tolist(),
            "levels": [0.05, 0.5, 0.95],
            "probability": None,
            "conditional_on_appearance": True,
            "model_version": report["version"],
        }
    raw = dist.probability(line, event)
    # Calibrate survival once; under is its complement, not a second mapping.
    cal = bundle["calibrators"][method]
    probability = 1 - cal.predict(1 - raw) if event == "under" else cal.predict(raw)
    return {
        "player_game_id": str(features.player_game_id.iloc[0]),
        "target": bundle["target"],
        "expected_value": float(dist.expected[0]),
        "variance": float(dist.variance[0]),
        "distribution_parameters": {
            "base_mean": float(dist.mean[0]),
            "nb2_dispersion": float(dist.dispersion[0]),
            "zero_gate": float(dist.gate[0]),
        },
        "line": line,
        "event": event,
        "raw_probability": float(raw[0]),
        "probability": float(probability[0]),
        "calibration": method,
        "model_version": report["version"],
        "prediction_asof": str(features.game_date.iloc[0]),
        "generated_at": datetime.now(UTC).isoformat(),
        "conditional_on_appearance": True,
        "warnings": [
            "Historical test forecast; not a live recommendation or confirmed market.",
            "No inactive-risk, timestamped injury or lineup model; void rules vary.",
            "Raw distribution moments are not recalibrated by threshold calibration.",
            f"Missing features: {int(features[bundle['model'].columns].isna().sum().sum())}",
        ],
    }


def predict_game(
    folder, history, game_context, cutoff, minutes_folder, line, event="at_least", method="sigmoid"
):
    """Local future-game inference using a frozen earlier OOF minutes model.

    Context must supply player_id, game_id, NBA Eastern game_date, season,
    season_type, team_id, opponent_id, home_away and scheduled_tipoff.
    """
    import json
    from pathlib import Path

    from catboost import CatBoostRegressor

    from nba_player_prop_prediction.features.pregame import build_features
    from nba_player_prop_prediction.models.minutes.training import matrix
    from nba_player_prop_prediction.probabilistic.data import enrich

    cutoff = pd.Timestamp(cutoff)
    tipoff = pd.Timestamp(game_context["scheduled_tipoff"])
    if cutoff.tzinfo is None or tipoff.tzinfo is None or cutoff >= tipoff:
        raise ValueError("Timezone-aware cutoff must precede scheduled tipoff")
    cutoff_day = pd.Timestamp(cutoff.tz_convert("America/New_York").date())
    if (
        pd.Timestamp(game_context["game_date"]).date()
        != tipoff.tz_convert("America/New_York").date()
    ):
        raise ValueError("Game date must match scheduled Eastern tipoff date")
    historical = history.loc[history.game_date < cutoff_day].copy()
    context = dict(game_context)
    context["player_game_id"] = f"{context['player_id']}:{context['game_id']}"
    context["cutoff_date"] = cutoff_day
    features = build_features(historical, pd.DataFrame([context]))
    manifest = json.loads((Path(minutes_folder) / "manifest.json").read_text())
    folds = [
        f
        for f in manifest["folds"]
        if "trained_through" in f
        and pd.Timestamp(f["trained_through"]) < cutoff_day
        and pd.Timestamp(f["start"]) <= cutoff_day
    ]
    if not folds:
        raise ValueError("No eligible frozen minutes model")
    fold = max(folds, key=lambda f: f["start"])
    model = CatBoostRegressor()
    model.load_model(str(Path(minutes_folder) / fold["model"]))
    oof = features[["player_game_id", "game_date"]].copy()
    oof["predicted_minutes"] = np.maximum(0, model.predict(matrix(features, manifest["columns"])))
    oof["minutes_uncertainty"] = features.prior_10_minutes_std
    oof["minutes_trained_through"] = pd.Timestamp(fold["trained_through"])
    oof["prediction_asof"] = cutoff_day
    enriched = enrich(historical, features, oof)
    bundle, _ = load_bundle(folder)
    if pd.Timestamp(bundle["trained_through"]) >= cutoff_day:
        raise ValueError("Stat model/calibrator unavailable at cutoff")
    result = predict(folder, enriched, line, event, method)
    result["prediction_asof"] = cutoff.isoformat()
    result["minutes_model_trained_through"] = fold["trained_through"]
    result["warnings"] = [
        "Prepared schedule context is caller-supplied; no live source was fetched.",
        "Conditional on appearing; injuries, starters and void rules are unmodeled.",
        "Minutes volatility is a proxy, not integrated uncertainty propagation.",
    ]
    return result
