"""Deterministic synthetic tests, never used for production performance claims."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nba_player_prop_prediction.features.pregame import availability_at, build_features
from nba_player_prop_prediction.models.minutes.predict import predict_player_minutes
from nba_player_prop_prediction.models.minutes.training import MinutesModel, TrainConfig, train
from nba_player_prop_prediction.preprocessing.clean import STATS


@pytest.fixture
def games():
    rows = []
    for day in range(30):
        for player in range(4):
            row = {
                "player_id": str(player),
                "game_id": str(day),
                "player_game_id": f"{player}:{day}",
                "team_id": str(player // 2),
                "opponent_id": str(1 - player // 2),
                "home_away": "home",
                "game_date": pd.date_range("2023-10-01", periods=30)[day],
                "season": "2023-24",
                "season_type": "Regular Season",
                "participation": "played",
                "minutes": float(10 + player * 5 + day % 7),
            }
            row.update({col: float(day % 5 + 1) for col in STATS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_rolls_schedule_order_and_reproducibility(games):
    f = build_features(games)
    row = f.loc[f.player_game_id.eq("0:3")].iloc[0]
    assert row.prior_3_minutes_mean == 11
    assert row.prior_3_minutes_median == 11
    assert row.prior_3_minutes_std == 1
    assert row.previous_minutes == 12
    assert row.rest_days == 0 and row.back_to_back == 1
    assert row.appearances_previous_7_days == 3
    assert row.prior_games == 3
    assert pd.isna(f.prior_3_minutes_std.iloc[0])
    assert_frame_equal(f, build_features(games.sample(frac=1, random_state=9)))
    assert "minutes" not in f and "points" not in f and "result" not in f


def test_future_target_and_same_day_leakage(games):
    before = build_features(games)
    changed = games.copy()
    changed.loc[changed.game_date >= "2023-10-10", ["minutes", *STATS]] = 100
    after = build_features(changed)
    assert_frame_equal(
        before.loc[before.game_date <= "2023-10-10"], after.loc[after.game_date <= "2023-10-10"]
    )
    same_day = games.iloc[:8].copy()
    same_day["game_date"] = pd.Timestamp("2023-10-01")
    assert build_features(same_day).prior_games.eq(0).all()


def test_trade_season_rookie_missing_and_rest(games):
    selected = games.loc[games.player_id.eq("0")].iloc[:3].copy()
    selected.iloc[1, selected.columns.get_loc("game_date")] = pd.Timestamp("2024-10-01")
    selected.iloc[2, selected.columns.get_loc("game_date")] = pd.Timestamp("2024-10-10")
    selected.loc[selected.game_id.isin(["1", "2"]), "season"] = "2024-25"
    selected.loc[selected.game_id.eq("2"), "team_id"] = "9"
    selected.loc[selected.game_id.eq("0"), "points"] = np.nan
    f = build_features(selected)
    assert f.prior_games.tolist() == [0, 1, 2]
    assert f.rest_days.iloc[2] == 8
    assert f.appearances_previous_7_days.iloc[2] == 0
    assert f.appearances_previous_14_days.iloc[2] == 1
    assert f.prior_3_minutes_mean.iloc[2] == 10.5
    assert pd.isna(f.prior_3_points_mean.iloc[1])


def test_team_history_excludes_target_and_partial(games):
    games["minutes"] = 120.0  # Two synthetic players per team, complete team minutes.
    f = build_features(games)
    assert f.team_prior_10_points.iloc[4] == 2
    changed = games.copy()
    changed.loc[changed.game_date >= "2023-10-05", "points"] = 100
    ff = build_features(changed)
    assert_frame_equal(f.loc[f.game_date <= "2023-10-05"], ff.loc[ff.game_date <= "2023-10-05"])
    games.loc[games.game_id.eq("0"), "minutes"] = 10
    assert build_features(games).team_prior_10_points.iloc[:8].isna().all()


def test_timestamped_availability():
    status = pd.DataFrame(
        [
            dict(
                player_id="1",
                game_id="g",
                available_at="2024-01-01T12:00Z",
                status="out",
                trusted=True,
                source="fixture",
            )
        ]
    )
    assert availability_at(status, "1", "g", "2024-01-01T12:00Z") is None
    assert availability_at(status, "1", "g", "2024-01-01T13:00Z")["status"] == "out"
    status["trusted"] = False
    assert availability_at(status, "1", "g", "2024-01-01T13:00Z") is None


def test_insufficient_training(games, tmp_path):
    report = train(games, tmp_path)
    assert report["status"] == "skipped"
    assert not (tmp_path / "model.cbm").exists()
    assert "metrics" not in report
    with pytest.raises(ValueError, match="Validation"):
        TrainConfig(validation_start="2025-01-01", test_start="2024-01-01")


def test_model_roundtrip_and_prediction(games, tmp_path):
    config = TrainConfig(
        validation_start="2023-10-15",
        test_start="2023-10-23",
        min_train=20,
        min_validation=10,
        min_test=10,
        min_train_dates=5,
        min_eval_dates=3,
        min_players=2,
        iterations=8,
        depth=2,
        threads=1,
    )
    report = train(games, tmp_path, config)
    assert report["status"] == "trained"
    assert {m["n"] for m in report["metrics"]["test"].values()} == {32}
    model = MinutesModel.load(tmp_path)
    original = pd.read_parquet(tmp_path / "predictions.parquet")
    f = build_features(games)
    expected = original.loc[original.split.eq("test") & original.model.eq("catboost"), "predicted"]
    np.testing.assert_allclose(model.predict(f.loc[f.game_date >= "2023-10-23"]), expected)
    context = dict(
        game_id="future",
        game_date="2023-11-01",
        team_id="0",
        opponent_id="1",
        season="2023-24",
        season_type="Regular Season",
        home_away="away",
        scheduled_tipoff="2023-11-01T19:00:00-04:00",
    )
    result = predict_player_minutes(model, games, "0", context, "2023-11-01T12:00:00-04:00")
    assert result["predicted_minutes"] >= 0
    assert result["interval"]["lower"] <= result["predicted_minutes"]
    assert result["warnings"]
    with pytest.raises(ValueError, match="Model used labels"):
        predict_player_minutes(model, games, "0", context, "2023-10-01T12:00:00-04:00")
    # Guard persisted feature schema/version.
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["feature_version"] = "bad"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        MinutesModel.load(tmp_path)


def test_possession_proxies_and_unknown_context(games):
    games["minutes"] = 120.0
    games["oreb"] = 1.0
    games["field_goal_attempts"] = 40.0
    games["free_throw_attempts"] = 10.0
    games["turnovers"] = 5.0
    features = build_features(games)
    assert features.team_prior_10_estimated_pace.iloc[4] == pytest.approx(96.8)
    context = games.iloc[:1].copy()
    context["player_id"] = "rookie"
    context["player_game_id"] = "rookie:new"
    context["game_id"] = "new"
    result = build_features(games, context)
    assert result.prior_games.iloc[0] == 0
    assert pd.isna(result.previous_minutes.iloc[0])
    context["cutoff_date"] = "2024-01-01"
    with pytest.raises(ValueError, match="Cutoff"):
        build_features(games, context)
