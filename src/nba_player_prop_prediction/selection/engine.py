"""Integrate validation-selected models with supplied pregame schedule context."""

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
from catboost import CatBoostRegressor

from nba_player_prop_prediction.features.pregame import availability_at, build_features
from nba_player_prop_prediction.models.minutes.training import matrix
from nba_player_prop_prediction.probabilistic.distributions import TARGETS, CountDistribution
from nba_player_prop_prediction.probabilistic.experiment import load_bundle
from nba_player_prop_prediction.probabilistic.prediction import predict_game
from nba_player_prop_prediction.selection.markets import confirmed_market


def generate(
    history, contexts, cutoff, root, minutes_folder, availability=None, markets=None, lines=None
):
    if not contexts:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff needs timezone")
    day = pd.Timestamp(cutoff.tz_convert("America/New_York").date())
    historical = history.loc[history.game_date < day]
    manifest = json.loads((minutes_folder / "manifest.json").read_text())
    folds = [
        f
        for f in manifest["folds"]
        if "trained_through" in f
        and pd.Timestamp(f["trained_through"]) < day
        and pd.Timestamp(f["start"]) <= day
    ]
    if not folds:
        raise ValueError("No eligible minutes model")
    fold = max(folds, key=lambda f: f["start"])
    minutes = CatBoostRegressor()
    minutes.load_model(str(minutes_folder / fold["model"]))
    rows = []
    for context in contexts:
        if pd.Timestamp(context["scheduled_tipoff"]) <= cutoff:
            raise ValueError("Cutoff must precede tipoff")
        pg = f"{context['player_id']}:{context['game_id']}"
        f = build_features(
            historical, pd.DataFrame([dict(context, player_game_id=pg, cutoff_date=day)])
        )
        predicted_minutes = max(0, float(minutes.predict(matrix(f, manifest["columns"]))[0]))
        status = (
            None
            if availability is None
            else availability_at(
                availability, context["player_id"], context["game_id"], cutoff.isoformat()
            )
        )
        for target in TARGETS:
            selected = json.loads((root / target / "validation_selection.json").read_text())[
                "selected"
            ]
            folder = root / target / selected
            result = predict_game(folder, history, context, cutoff.isoformat(), minutes_folder, 1)
            bundle, report = load_bundle(folder)
            params = result["distribution_parameters"]
            dist = CountDistribution(
                [params["base_mean"]], params["nb2_dispersion"], params["zero_gate"]
            )
            specifications = (lines or {}).get(
                target, [{"line": line, "event": "at_least"} for line in TARGETS[target]]
            )
            for spec in specifications:
                line, event = spec["line"], spec["event"]
                raw = float(dist.probability(line, event)[0])
                cal = bundle["calibrators"]["sigmoid"]
                probability = float(
                    1 - cal.predict([1 - raw])[0] if event == "under" else cal.predict([raw])[0]
                )
                market = next(
                    (
                        m
                        for m in (markets or [])
                        if confirmed_market(
                            m,
                            context["player_id"],
                            context["game_id"],
                            target,
                            line,
                            event,
                            cutoff.isoformat(),
                        )
                    ),
                    None,
                )
                rows.append(
                    dict(
                        context,
                        candidate_id=hashlib.sha256(
                            f"{pg}|{target}|{line}|{event}".encode()
                        ).hexdigest(),
                        player_game_id=pg,
                        target=target,
                        line=line,
                        event=event,
                        probability=probability,
                        raw_probability=raw,
                        expected_value=result["expected_value"],
                        distribution=params,
                        model=selected,
                        model_version=report["version"],
                        predicted_minutes=predicted_minutes,
                        minutes_uncertainty=None
                        if pd.isna(f.prior_10_minutes_std.iloc[0])
                        else float(f.prior_10_minutes_std.iloc[0]),
                        uncertainty_kind="historical_minutes_volatility_not_probability_error",
                        calibration_n=report["periods"]["calibration"]["rows"],
                        prior_games=int(f.prior_games.iloc[0]),
                        back_to_back=None
                        if pd.isna(f.back_to_back.iloc[0])
                        else float(f.back_to_back.iloc[0]),
                        minutes_change=None
                        if pd.isna(f.minutes_change.iloc[0])
                        else float(f.minutes_change.iloc[0]),
                        availability="unknown" if status is None else str(status["status"]).lower(),
                        minutes_restricted=False
                        if status is None
                        else bool(status.get("minutes_restricted", False)),
                        market_available=market is not None,
                        market_snapshot=market,
                        conditional_on_appearance=True,
                        missing_feature_count=int(f.isna().sum().sum()),
                        warnings=result["warnings"],
                        generated_at=datetime.now(UTC).isoformat(),
                        source_cutoff=cutoff.isoformat(),
                        mode="forward",
                    )
                )
    return pd.DataFrame(rows)
